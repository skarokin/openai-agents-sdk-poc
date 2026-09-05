"""Auth-related execution policy."""

from .vault_token import require_vault_token

__all__ = ["require_vault_token"]
