"""API request and response schemas."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class TimelineEvent(BaseModel):
    timestamp: str
    event: str
    source: str = "unknown"


class IncidentRequest(BaseModel):
    incident_id: str = Field(..., description="Unique incident identifier, e.g. INC-2024-001")
    incident_title: str = Field(..., min_length=5, max_length=300)
    incident_description: str = Field(..., min_length=10)
    severity: Literal["P1", "P2", "P3", "P4"] = "P3"
    affected_service: str = Field(..., description="Name of the impacted service")
    org: str | None = Field(default=None, description="Org identifier, used to route the incident internally")
    timeline_events: list[TimelineEvent] = Field(default_factory=list)
    raw_logs: list[str] = Field(default_factory=list, max_length=50)


class IncidentResponse(BaseModel):
    incident_id: str
    thread_id: str
    org: str | None = None
    status: Literal["completed", "pending_review", "error"]
    risk_score: float | None = None
    requires_human_review: bool = False
    root_cause_analysis: str | None = None
    remediation_steps: list[str] = Field(default_factory=list)
    confidence_score: float | None = None
    final_response: str | None = None
    agent_messages: list[str] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)
    error: str | None = None


class HITLResumeRequest(BaseModel):
    thread_id: str = Field(..., description="LangGraph thread ID returned in original response")
    approved: bool = Field(..., description="True to approve and continue, False to reject")
    reviewer_id: str | None = None
    review_notes: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    vector_store_backend: str
    bedrock_model: str
