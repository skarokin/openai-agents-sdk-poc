"""Process-wide logging and OpenTelemetry setup."""

import contextvars
import logging
import os
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from agents import TracingProcessor, add_trace_processor, flush_traces
from fastapi import FastAPI
from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.trace import Status, StatusCode

from agent.core.models import IdentityContext

_request_id = contextvars.ContextVar("request_id", default="-")
_subject_id = contextvars.ContextVar("subject_id", default="-")
_actor_id = contextvars.ContextVar("actor_id", default="-")
_configured = False
_processor: "AgentsOtelProcessor | None" = None


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = getattr(record, "request_id", _request_id.get())
        record.subject_id = getattr(record, "subject_id", _subject_id.get())
        record.actor_id = getattr(record, "actor_id", _actor_id.get())
        span_context = otel_trace.get_current_span().get_span_context()
        record.otel_trace_id = (
            f"{span_context.trace_id:032x}" if span_context.is_valid else "-"
        )
        return True


def _safe_attributes(values: Mapping[str, Any] | None) -> dict[str, Any]:
    if not values:
        return {}
    return {
        str(key): value
        for key, value in values.items()
        if isinstance(value, str | bool | int | float)
    }


class AgentsOtelProcessor(TracingProcessor):
    """Mirror Agents SDK traces into the configured OpenTelemetry provider."""

    def __init__(self):
        self._tracer = otel_trace.get_tracer("openai-agents")
        self._traces: dict[str, Any] = {}
        self._spans: dict[str, Any] = {}
        self._lock = threading.Lock()

    def on_trace_start(self, trace) -> None:
        try:
            exported = trace.export() or {}
            attributes = {
                "openai_agents.trace_id": trace.trace_id,
                "enduser.id": _subject_id.get(),
                "agent.request_id": _request_id.get(),
                "agent.actor_id": _actor_id.get(),
                **_safe_attributes(exported.get("metadata")),
            }
            span = self._tracer.start_span(trace.name, attributes=attributes)
            with self._lock:
                self._traces[trace.trace_id] = span
        except Exception:
            logging.getLogger(__name__).exception("failed to start mirrored trace")

    def on_trace_end(self, trace) -> None:
        with self._lock:
            span = self._traces.pop(trace.trace_id, None)
        if span is not None:
            span.end()

    def on_span_start(self, span) -> None:
        try:
            with self._lock:
                parent = (
                    self._spans.get(span.parent_id)
                    if span.parent_id is not None
                    else None
                ) or self._traces.get(span.trace_id)
            parent_context = (
                otel_trace.set_span_in_context(parent) if parent is not None else None
            )
            span_type = getattr(
                span.span_data,
                "type",
                type(span.span_data).__name__,
            )
            otel_span = self._tracer.start_span(
                f"openai_agents.{span_type}",
                context=parent_context,
                attributes={
                    "openai_agents.trace_id": span.trace_id,
                    "openai_agents.span_id": span.span_id,
                },
            )
            with self._lock:
                self._spans[span.span_id] = otel_span
        except Exception:
            logging.getLogger(__name__).exception("failed to start mirrored span")

    def on_span_end(self, span) -> None:
        with self._lock:
            otel_span = self._spans.pop(span.span_id, None)
        if otel_span is None:
            return
        if span.error:
            otel_span.set_status(Status(StatusCode.ERROR, str(span.error)))
        otel_span.end()

    def shutdown(self) -> None:
        self.force_flush()

    def force_flush(self) -> None:
        provider = otel_trace.get_tracer_provider()
        force_flush = getattr(provider, "force_flush", None)
        if force_flush:
            force_flush()


def setup_observability(app: FastAPI | None = None) -> None:
    """Configure telemetry once during application startup."""

    global _configured, _processor
    if not _configured:
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO").upper(),
            format=(
                "%(asctime)s %(levelname)s %(name)s "
                "request_id=%(request_id)s subject_id=%(subject_id)s "
                "actor_id=%(actor_id)s trace_id=%(otel_trace_id)s %(message)s"
            ),
        )
        context_filter = ContextFilter()
        root = logging.getLogger()
        for handler in root.handlers:
            handler.addFilter(context_filter)

        provider = TracerProvider(
            resource=Resource.create({"service.name": "agent-service"})
        )
        if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        elif os.getenv("OTEL_CONSOLE_EXPORTER", "").lower() in {"1", "true", "yes"}:
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        otel_trace.set_tracer_provider(provider)

        _processor = AgentsOtelProcessor()
        add_trace_processor(_processor)
        _configured = True

    if app is not None:
        FastAPIInstrumentor.instrument_app(app)


@contextmanager
def bind_observability_context(
    identity: IdentityContext,
    request_id: str,
) -> Iterator[None]:
    """Attach trusted identity to logs and the current request span."""

    request_token = _request_id.set(request_id)
    subject_token = _subject_id.set(identity.subject_id)
    actor_token = _actor_id.set(identity.actor_id or "-")
    span = otel_trace.get_current_span()
    span.set_attribute("enduser.id", identity.subject_id)
    span.set_attribute("agent.request_id", request_id)
    span.set_attribute("agent.roles", sorted(identity.roles))
    if identity.actor_id:
        span.set_attribute("agent.actor_id", identity.actor_id)
    try:
        yield
    finally:
        _actor_id.reset(actor_token)
        _subject_id.reset(subject_token)
        _request_id.reset(request_token)


def shutdown_observability() -> None:
    flush_traces()
    if _processor is not None:
        _processor.force_flush()
