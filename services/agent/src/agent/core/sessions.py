"""Helpers for constructing Strands FileSessionManager instances."""

import hashlib
import os
from pathlib import Path

from strands.session.file_session_manager import FileSessionManager


def session_storage_dir() -> str:
    root = Path(os.getenv("AGENT_DATA_DIR", ".agent-data")) / "sessions"
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


def scoped_session_id(session_id: str, subject_id: str) -> str:
    """Isolate the same client session_id across subjects."""

    return hashlib.sha256(f"{subject_id}:{session_id}".encode()).hexdigest()


def make_file_session_manager(session_id: str, subject_id: str) -> FileSessionManager:
    """Return Strands' FileSessionManager for this caller conversation."""

    return FileSessionManager(
        session_id=scoped_session_id(session_id, subject_id),
        storage_dir=session_storage_dir(),
    )
