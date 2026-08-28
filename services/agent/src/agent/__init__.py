"""Multi-protocol OpenAI Agents SDK service."""


def main() -> None:
    from .main import main as run

    run()


__all__ = ["main"]
