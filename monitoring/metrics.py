"""
Prometheus Metrics Registry
----------------------------
Defines all custom application metrics. Import and use these throughout
the codebase — they're all registered at module load time.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ── Workflow metrics ──────────────────────────────────────────────────────────
INCIDENT_SUBMITTED = Counter(
    "incident_submitted_total",
    "Total incidents submitted to the workflow",
    ["severity"],
)

INCIDENT_COMPLETED = Counter(
    "incident_completed_total",
    "Total incidents completed (including HITL path)",
    ["severity", "status"],  # status: completed | pending_review | error
)

WORKFLOW_DURATION = Histogram(
    "workflow_duration_seconds",
    "End-to-end workflow duration",
    ["severity"],
    buckets=[1, 2, 5, 10, 20, 30, 60, 120],
)

# ── Agent-level metrics ───────────────────────────────────────────────────────
AGENT_LATENCY = Histogram(
    "agent_latency_seconds",
    "Per-agent execution latency",
    ["agent_name"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
)

AGENT_ERRORS = Counter(
    "agent_errors_total",
    "Agent-level errors",
    ["agent_name", "error_type"],
)

# ── RAG metrics ───────────────────────────────────────────────────────────────
RETRIEVAL_SCORE = Histogram(
    "retrieval_top_score",
    "Top similarity score from vector store retrieval",
    buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
)

RETRIEVAL_DOCS_RETURNED = Histogram(
    "retrieval_docs_returned",
    "Number of docs returned per retrieval",
    buckets=[0, 1, 2, 3, 5, 10],
)

# ── Risk metrics ──────────────────────────────────────────────────────────────
RISK_SCORE = Histogram(
    "risk_score_distribution",
    "Distribution of incident risk scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

HITL_TRIGGERED = Counter(
    "hitl_triggered_total",
    "Times human-in-the-loop review was triggered",
    ["severity"],
)

HITL_APPROVED = Counter(
    "hitl_decisions_total",
    "HITL review decisions",
    ["decision"],  # approved | rejected
)

# ── LLM metrics ───────────────────────────────────────────────────────────────
LLM_TOKEN_USAGE = Counter(
    "llm_tokens_total",
    "Estimated token usage by model",
    ["model", "direction"],  # direction: input | output
)

LLM_ERRORS = Counter(
    "llm_errors_total",
    "Bedrock API errors",
    ["model", "error_code"],
)

# ── Drift monitoring ──────────────────────────────────────────────────────────
RETRIEVAL_DRIFT_SCORE = Gauge(
    "retrieval_drift_score",
    "Embedding drift score (0 = no drift, 1 = full drift) from last baseline",
)

CONFIDENCE_ROLLING_MEAN = Gauge(
    "confidence_score_rolling_mean",
    "Rolling mean of LLM confidence scores (last 100 incidents)",
)
