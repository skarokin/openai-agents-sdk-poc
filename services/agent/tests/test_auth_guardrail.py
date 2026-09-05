"""Auth interrupt helpers and vault behavior."""

import tempfile
from pathlib import Path

import pytest

from agent.core.models import IdentityContext
from agent.core.token_vault import TokenVault


@pytest.mark.asyncio
async def test_vault_put_get_round_trip():
    with tempfile.TemporaryDirectory() as directory:
        vault = TokenVault(Path(directory))
        identity = IdentityContext(subject_id="user-1", roles=frozenset({"auth_user"}))
        assert await vault.has_token(identity, "authentication_demo") is False
        await vault.put(identity, "authentication_demo", "tok-123")
        assert await vault.has_token(identity, "authentication_demo") is True
        stored = await vault.get(identity, "authentication_demo")
        assert stored is not None
        assert stored.access_token == "tok-123"


def test_vault_challenge_url():
    vault = TokenVault(Path(tempfile.mkdtemp()))
    challenge = vault.challenge("authentication_demo")
    assert challenge.service == "authentication_demo"
    assert challenge.authorization_url == "agent://vault/authentication_demo"
