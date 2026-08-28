"""JSON-backed conversation sessions and resumable run-state storage."""

import asyncio
import hashlib
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

from agents import (
    Agent,
    RunContextWrapper,
    RunState,
    SessionSettings,
    TResponseInputItem,
)

from .event_mapping import approval_id
from .models import AgentContext


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


class JsonSession:
    """Minimal single-process implementation of the Agents SDK Session protocol."""

    def __init__(
        self,
        session_id: str,
        owner_subject_id: str,
        path: Path,
        lock: asyncio.Lock,
        session_settings: SessionSettings | None = None,
    ):
        self.session_id = session_id
        self._owner_subject_id = owner_subject_id
        self.session_settings: SessionSettings | None = (
            session_settings or SessionSettings()
        )
        self._path = path
        self._lock = lock

    def _read(self) -> list[TResponseInputItem]:
        if not self._path.exists():
            return []
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        if payload.get("owner_subject_id") != self._owner_subject_id:
            raise PermissionError("Session owner does not match caller identity")
        items = payload.get("items", [])
        if not isinstance(items, list):
            raise TypeError(f"Invalid session data in {self._path}")
        return cast(list[TResponseInputItem], items)

    def _write(self, items: list[TResponseInputItem]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(f".{uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "session_id": self.session_id,
                    "owner_subject_id": self._owner_subject_id,
                    "items": items,
                },
                default=_json_default,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, self._path)

    async def get_items(
        self,
        limit: int | None = None,
    ) -> list[TResponseInputItem]:
        async with self._lock:
            items = self._read()
            effective_limit = (
                limit
                if limit is not None
                else self.session_settings.limit
                if self.session_settings is not None
                else None
            )
            if effective_limit is None:
                return items
            if effective_limit < 0:
                raise ValueError("Session item limit cannot be negative")
            return items[-effective_limit:] if effective_limit else []

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        async with self._lock:
            existing = self._read()
            existing.extend(items)
            self._write(existing)

    async def pop_item(self) -> TResponseInputItem | None:
        async with self._lock:
            items = self._read()
            if not items:
                return None
            item = items.pop()
            self._write(items)
            return item

    async def clear_session(self) -> None:
        async with self._lock:
            self._write([])


class SessionManager:
    """Own JSON sessions and durable HITL checkpoints."""

    def __init__(self, data_dir: Path):
        self._session_dir = data_dir / "sessions"
        self._run_state_dir = data_dir / "run_states"
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._run_state_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, asyncio.Lock] = {}
        self._run_state_locks: dict[UUID, asyncio.Lock] = {}

    @staticmethod
    def from_environment() -> "SessionManager":
        return SessionManager(Path(os.getenv("AGENT_DATA_DIR", ".agent-data")))

    def session(self, session_id: str, owner_subject_id: str) -> JsonSession:
        owner_key = f"{owner_subject_id}:{session_id}"
        digest = hashlib.sha256(owner_key.encode("utf-8")).hexdigest()
        lock = self._locks.setdefault(owner_key, asyncio.Lock())
        return JsonSession(
            session_id=session_id,
            owner_subject_id=owner_subject_id,
            path=self._session_dir / f"{digest}.json",
            lock=lock,
        )

    @staticmethod
    def _serialize_context(context: AgentContext) -> dict[str, Any]:
        return {
            "identity": {
                "subject_id": context.identity.subject_id,
                "actor_id": context.identity.actor_id,
                "roles": sorted(context.identity.roles),
            },
            "request_id": context.request_id,
            "session_id": context.session_id,
            "deadline_epoch_seconds": context.deadline_epoch_seconds,
        }

    async def save_run_state(
        self,
        state: RunState[Any],
        *,
        session_id: str,
        owner_subject_id: str,
    ) -> UUID:
        token = uuid4()
        payload = {
            "session_id": session_id,
            "owner_subject_id": owner_subject_id,
            "state": state.to_json(
                context_serializer=self._serialize_context,
                strict_context=True,
            ),
        }
        path = self._run_state_dir / f"{token}.json"
        temporary = path.with_suffix(f".{uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(payload, default=_json_default),
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return token

    async def load_run_state(
        self,
        token: UUID,
        *,
        agent: Agent[AgentContext],
        context: AgentContext,
    ) -> tuple[RunState[Any], str]:
        path = self._run_state_dir / f"{token}.json"
        if not path.exists():
            raise FileNotFoundError(f"Unknown resume token: {token}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("owner_subject_id") != context.identity.subject_id:
            raise PermissionError("Resume token owner does not match caller identity")
        state = await RunState.from_json(
            agent,
            payload["state"],
            context_override=RunContextWrapper(context=context),
            context_deserializer=lambda _: context,
            strict_context=True,
        )
        return state, str(payload["session_id"])

    def run_state_session_id(self, token: UUID, owner_subject_id: str) -> str:
        path = self._run_state_dir / f"{token}.json"
        if not path.exists():
            raise FileNotFoundError(f"Unknown resume token: {token}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("owner_subject_id") != owner_subject_id:
            raise PermissionError("Resume token owner does not match caller identity")
        return str(payload["session_id"])

    @asynccontextmanager
    async def lock_run_state(self, token: UUID) -> AsyncIterator[None]:
        """Prevent concurrent execution of the same resumable checkpoint."""

        lock = self._run_state_locks.setdefault(token, asyncio.Lock())
        async with lock:
            yield

    @staticmethod
    def apply_decisions(
        state: RunState[Any],
        decisions: dict[str, str],
    ) -> None:
        interruptions = state.get_interruptions()
        known_ids: set[str] = set()
        for index, item in enumerate(interruptions):
            item_id = approval_id(item, index)
            known_ids.add(item_id)
            decision = decisions.get(item_id)
            if decision == "approve":
                state.approve(item)
            elif decision == "reject":
                state.reject(
                    item,
                    rejection_message="The caller rejected this tool invocation.",
                )
        unknown = set(decisions) - known_ids
        if unknown:
            raise ValueError(f"Unknown interruption IDs: {sorted(unknown)}")

    async def delete_run_state(self, token: UUID) -> None:
        path = self._run_state_dir / f"{token}.json"
        if path.exists():
            path.unlink()
