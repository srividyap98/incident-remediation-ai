"""Shared LangGraph state schema — v2 adds grounding_warnings."""
from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict
import operator


class IncidentState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────────────
    incident_id: str
    incident_title: str
    incident_description: str
    severity: str
    affected_service: str
    timeline_events: list[dict[str, Any]]
    raw_logs: list[str]

    # ── Retriever ─────────────────────────────────────────────────────────────
    retrieved_docs: list[dict[str, Any]]
    retrieval_query: str

    # ── Context Builder ───────────────────────────────────────────────────────
    enriched_context: str
    structured_metadata: dict[str, Any]

    # ── Risk Evaluator ────────────────────────────────────────────────────────
    risk_score: float
    risk_factors: list[str]
    requires_human_review: bool

    # ── Response Generator ────────────────────────────────────────────────────
    remediation_steps: list[str]
    root_cause_analysis: str
    confidence_score: float
    final_response: str

    # ── Evaluator (v2) ────────────────────────────────────────────────────────
    grounding_warnings: list[str]   # Claims flagged as potentially ungrounded

    # ── Orchestration ─────────────────────────────────────────────────────────
    agent_messages: Annotated[list[str], operator.add]
    current_agent: str
    iteration_count: int
    hitl_approved: bool
    error: str | None
