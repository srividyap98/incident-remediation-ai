"""FastAPI application — Incident Remediation AI service v2."""
from __future__ import annotations

import time
import uuid

import config.offline_mode  # noqa: E402,F401 — must patch boto3/embeddings before agent/rag imports
import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, make_asgi_app

from api.routers import incidents, health, hitl, voice, feedback, servicenow
from config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# ── OpenTelemetry setup ───────────────────────────────────────────────────────
def _setup_otel(app: FastAPI) -> None:
    """Instrument FastAPI with OpenTelemetry if endpoint is configured."""
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        provider = TracerProvider()
        exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app)
        logger.info("OTel tracing enabled", endpoint=settings.otel_exporter_otlp_endpoint)
    except Exception as exc:
        logger.warning("OTel setup failed — continuing without tracing", error=str(exc))


# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds", "Request latency", ["endpoint"]
)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Incident Remediation AI",
    description=(
        "Multi-Agent GenAI system for incident remediation & risk intelligence. "
        "Built on LangGraph + AWS Bedrock (Claude Sonnet 4.5)."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OTel
_setup_otel(app)

# ── Request logging middleware ────────────────────────────────────────────────
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    structlog.contextvars.bind_contextvars(request_id=request_id)

    response = await call_next(request)
    duration = time.perf_counter() - start

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)

    logger.info(
        "request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round(duration * 1000, 2),
        request_id=request_id,
    )
    structlog.contextvars.clear_contextvars()
    return response


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router, tags=["Health"])
app.include_router(incidents.router,  prefix="/api/v1/incidents", tags=["Incidents"])
app.include_router(hitl.router,       prefix="/api/v1/hitl",      tags=["Human-in-the-Loop"])
app.include_router(voice.router,      prefix="/api/v1/voice",     tags=["Voice / IVR"])
app.include_router(feedback.router,   prefix="/api/v1/feedback",  tags=["Feedback"])
app.include_router(
    servicenow.router,
    prefix="/api/v1/integrations/servicenow",
    tags=["ServiceNow Integration"],
)

# Prometheus metrics endpoint
app.mount("/metrics", make_asgi_app())
