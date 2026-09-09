"""FileSessionManager helper tests."""

import tempfile
from pathlib import Path

from strands.session.file_session_manager import FileSessionManager

from agent.core.sessions import (
    ROOT_AGENT_ID,
    make_file_session_manager,
    scoped_session_id,
    session_storage_dir,
)


def test_file_session_is_scoped_by_owner(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        monkeypatch.setenv("AGENT_DATA_DIR", directory)
        owner = make_file_session_manager("shared", "owner")
        other = make_file_session_manager("shared", "intruder")
        assert isinstance(owner, FileSessionManager)
        assert owner.session_id != other.session_id
        assert owner.session_id == scoped_session_id("shared", "owner")
        assert Path(directory, "sessions").is_dir()


def test_scoped_session_id_is_stable_and_isolates_subjects():
    first = scoped_session_id("abc", "user-1")
    second = scoped_session_id("abc", "user-1")
    other_user = scoped_session_id("abc", "user-2")
    other_session = scoped_session_id("xyz", "user-1")

    assert first == second
    assert first != other_user
    assert first != other_session
    assert len(first) == 64  # sha256 hex digest


def test_session_storage_dir_uses_agent_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_DATA_DIR", str(tmp_path))
    path = Path(session_storage_dir())
    assert path == tmp_path / "sessions"
    assert path.is_dir()


def test_make_file_session_manager_uses_storage_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_DATA_DIR", str(tmp_path))
    manager = make_file_session_manager("s1", "u1")
    assert manager.session_id == scoped_session_id("s1", "u1")
    assert Path(manager.storage_dir) == tmp_path / "sessions"


def test_root_agent_id_is_stable():
    assert ROOT_AGENT_ID == "root"
