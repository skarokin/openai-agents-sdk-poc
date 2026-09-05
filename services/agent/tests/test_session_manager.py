"""FileSessionManager helper tests."""

import tempfile
from pathlib import Path

from strands.session.file_session_manager import FileSessionManager

from agent.core.sessions import make_file_session_manager, scoped_session_id


def test_file_session_is_scoped_by_owner(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        monkeypatch.setenv("AGENT_DATA_DIR", directory)
        owner = make_file_session_manager("shared", "owner")
        other = make_file_session_manager("shared", "intruder")
        assert isinstance(owner, FileSessionManager)
        assert owner.session_id != other.session_id
        assert owner.session_id == scoped_session_id("shared", "owner")
        assert Path(directory, "sessions").is_dir()
