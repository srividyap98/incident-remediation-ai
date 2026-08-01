"""
OpenTelemetry Distributed Tracing
-----------------------------------
Instruments each LangGraph agent node with named spans.
Exports to any OTLP-compatible backend: Honeycomb, Jaeger,
Splunk Observability Cloud, or a local OTel Collector.

Usage — wrap agent functions:
    from monitoring.otel_tracer import trace_agent

    @trace_agent("retriever")
    def retriever_agent(state: IncidentState) -> dict:
        ...

Or use the context manager directly:
    with agent_span("risk_evaluator", incident_id="INC-001") as span:
        span.set_attribute("severity", "P1")
        ...
"""
from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable

import structlog

logger = structlog.get_logger(__name__)

# Lazy import — OTel is optional; system works without it
_tracer = None


def _get_tracer():
    global _tracer
    if _tracer is not None:
        return _tracer
    try:
        from opentelemetry import trace
        _tracer = trace.get_tracer("incident-remediation-ai", "2.0.0")
    except ImportError:
        _tracer = _NoOpTracer()
    return _tracer


class _NoOpSpan:
    """Fallback when OTel is not installed — no-ops all calls."""
    def set_attribute(self, key: str, value: Any) -> None: pass
    def record_exception(self, exc: Exception) -> None: pass
    def set_status(self, *args, **kwargs) -> None: pass
    def __enter__(self): return self
    def __exit__(self, *args): pass


class _NoOpTracer:
    def start_as_current_span(self, name: str, **kwargs):
        return _NoOpSpan()


@contextmanager
def agent_span(agent_name: str, incident_id: str | None = None, **attributes):
    """
    Context manager that wraps a block in an OTel span.

    Example:
        with agent_span("retriever", incident_id="INC-001", severity="P1") as span:
            results = store.search(query)
            span.set_attribute("num_docs", len(results))
    """
    tracer = _get_tracer()
    with tracer.start_as_current_span(f"agent.{agent_name}") as span:
        try:
            if incident_id:
                span.set_attribute("incident.id", incident_id)
            span.set_attribute("agent.name", agent_name)
            for k, v in attributes.items():
                span.set_attribute(k, str(v))
            yield span
        except Exception as exc:
            try:
                from opentelemetry.trace import StatusCode
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc))
            except Exception:
                pass
            raise


def trace_agent(agent_name: str) -> Callable:
    """
    Decorator that wraps a LangGraph agent node function in an OTel span.
    Automatically records incident_id, severity, duration, and any error.

    Usage:
        @trace_agent("retriever")
        def retriever_agent(state: IncidentState) -> dict:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state: dict, *args, **kwargs) -> dict:
            incident_id = state.get("incident_id", "unknown")
            severity    = state.get("severity", "unknown")
            start       = time.perf_counter()

            with agent_span(agent_name, incident_id=incident_id, severity=severity) as span:
                result = fn(state, *args, **kwargs)
                duration = time.perf_counter() - start

                span.set_attribute("duration_ms", round(duration * 1000, 2))
                if isinstance(result, dict):
                    if result.get("risk_score") is not None:
                        span.set_attribute("risk_score", result["risk_score"])
                    if result.get("confidence_score") is not None:
                        span.set_attribute("confidence_score", result["confidence_score"])
                    if result.get("error"):
                        try:
                            from opentelemetry.trace import StatusCode
                            span.set_status(StatusCode.ERROR, result["error"])
                        except Exception:
                            pass

                return result
        return wrapper
    return decorator


def record_llm_call(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    duration_ms: float,
    incident_id: str | None = None,
) -> None:
    """Record LLM call metrics as OTel span attributes and Prometheus counters."""
    try:
        from monitoring.metrics import LLM_TOKEN_USAGE
        LLM_TOKEN_USAGE.labels(model=model_id, direction="input").inc(input_tokens)
        LLM_TOKEN_USAGE.labels(model=model_id, direction="output").inc(output_tokens)
    except Exception:
        pass

    logger.info(
        "llm_call",
        model=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_ms=duration_ms,
        incident_id=incident_id,
    )
