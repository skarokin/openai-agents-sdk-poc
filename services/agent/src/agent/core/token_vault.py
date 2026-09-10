"""JSON-backed per-subject tokens for JIT tool SSO."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from agent.core.models import IdentityContext


@dataclass(frozen=True, slots=True)
class StoredToken:
    access_token: str


@dataclass(frozen=True, slots=True)
class AuthChallenge:
    service: str
    authorization_url: str


class TokenVault:
    """Own demo tokens on disk. A real deployment calls an external vault instead."""

    def __init__(self, data_dir: Path):
        self._token_dir = data_dir / "tokens"
        self._token_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, asyncio.Lock] = {}

    @classmethod
    def from_environment(cls) -> TokenVault:
        return cls(Path(os.getenv("AGENT_DATA_DIR", ".agent-data")))

    def challenge(self, service: str) -> AuthChallenge:
        return AuthChallenge(
            service=service,
            authorization_url=f"agent://vault/{service}",
        )

    async def get(
        self,
        identity: IdentityContext,
        service: str,
    ) -> StoredToken | None:
        async with self._lock_for(identity.subject_id):
            return self._read_token(identity, service)

    async def has_token(self, identity: IdentityContext, service: str) -> bool:
        return await self.get(identity, service) is not None

    async def put(
        self,
        identity: IdentityContext,
        service: str,
        access_token: str,
    ) -> None:
        token = access_token.strip()
        if not token:
            raise ValueError("access_token must be a non-empty string")
        async with self._lock_for(identity.subject_id):
            payload = self._read(identity)
            payload.setdefault("tokens", {})[service] = {"access_token": token}
            self._write(identity, payload)

    def peek(self, identity: IdentityContext, service: str) -> StoredToken | None:
        return self._read_token(identity, service)

    def _lock_for(self, subject_id: str) -> asyncio.Lock:
        return self._locks.setdefault(subject_id, asyncio.Lock())

    def _path(self, subject_id: str) -> Path:
        digest = hashlib.sha256(subject_id.encode("utf-8")).hexdigest()
        return self._token_dir / f"{digest}.json"

    def _read(self, identity: IdentityContext) -> dict:
        path = self._path(identity.subject_id)
        if not path.exists():
            return {"subject_id": identity.subject_id, "tokens": {}}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("subject_id") != identity.subject_id:
            raise PermissionError("Token vault owner does not match caller identity")
        tokens = payload.get("tokens", {})
        if not isinstance(tokens, dict):
            raise TypeError(f"Invalid token vault data in {path}")
        payload["tokens"] = tokens
        return payload

    def _write(self, identity: IdentityContext, payload: dict) -> None:
        path = self._path(identity.subject_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _read_token(
        self,
        identity: IdentityContext,
        service: str,
    ) -> StoredToken | None:
        raw = self._read(identity).get("tokens", {}).get(service)
        if not isinstance(raw, dict):
            return None
        access_token = raw.get("access_token")
        if not isinstance(access_token, str) or not access_token.strip():
            return None
        return StoredToken(access_token=access_token.strip())
