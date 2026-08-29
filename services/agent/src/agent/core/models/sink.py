"""Event sink protocol for nested run output."""

from typing import Any, Protocol

from .events import EventSource


class EventSink(Protocol):
    """Destination for normalized events produced by nested work."""

    async def emit(self, event: Any, *, source: EventSource) -> None:
        """Publish an event for inclusion in the formatter's output stream."""
        ...
