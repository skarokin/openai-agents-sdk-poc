"""Process-wide logging and OpenTelemetry setup."""

import contextvars
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager

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

from agent.core.models import IdentityContext

_request_id = contextvars.ContextVar("request_id", default="-")
_subject_id = contextvars.ContextVar("subject_id", default="-")
_actor_id = contextvars.ContextVar("actor_id", default="-")
_configured = False


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


@contextmanager
def bind_observability_context(
    identity: IdentityContext,
    request_id: str,
) -> Iterator[None]:
    tokens = (
        _request_id.set(request_id),
        _subject_id.set(identity.subject_id),
        _actor_id.set(identity.actor_id or "-"),
    )
    try:
        yield
    finally:
        _request_id.reset(tokens[0])
        _subject_id.reset(tokens[1])
        _actor_id.reset(tokens[2])


def setup_observability(app: FastAPI) -> None:
    global _configured
    if _configured:
        FastAPIInstrumentor.instrument_app(app)
        return

    logging.basicConfig(
        level=os.getenv("AGENT_LOG_LEVEL", "INFO"),
        format=(
            "%(asctime)s %(levelname)s [%(name)s] "
            "request_id=%(request_id)s subject_id=%(subject_id)s "
            "actor_id=%(actor_id)s trace_id=%(otel_trace_id)s %(message)s"
        ),
    )
    for handler in logging.getLogger().handlers:
        handler.addFilter(ContextFilter())

    resource = Resource.create(
        {
            "service.name": os.getenv("OTEL_SERVICE_NAME", "agent"),
        }
    )
    provider = TracerProvider(resource=resource)
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        )
    elif os.getenv("AGENT_OTEL_CONSOLE", "").lower() in {"1", "true", "yes"}:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    otel_trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
    _configured = True


def shutdown_observability() -> None:
    provider = otel_trace.get_tracer_provider()
    shutdown = getattr(provider, "shutdown", None)
    if callable(shutdown):
        shutdown()
