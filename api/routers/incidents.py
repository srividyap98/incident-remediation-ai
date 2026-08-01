"""Incidents router — submit incidents for AI-driven remediation."""
from __future__ import annotations

import uuid
from typing import TypedDict

import structlog
from fastapi import APIRouter, Depends, HTTPException
from langchain_core.runnables import RunnableConfig

from agents.orchestrator.graph import get_graph
from api.auth import get_current_principal
from api.schemas.models import IncidentRequest, IncidentResponse
from config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()
router = APIRouter(dependencies=[Depends(get_current_principal)])


class ThreadInfo(TypedDict):
    config: RunnableConfig
    incident_id: str


# In-memory thread store (replace with Redis/DynamoDB in production)
_thread_store: dict[str, ThreadInfo] = {}


@router.post("/", response_model=IncidentResponse, status_code=202)
async def submit_incident(request: IncidentRequest) -> IncidentResponse:
    """
    Submit an incident for multi-agent analysis and remediation.

    Returns immediately with status=pending_review if HITL is triggered,
    or status=completed with full remediation guidance otherwise.
    """
    thread_id = str(uuid.uuid4())
    log = logger.bind(incident_id=request.incident_id, thread_id=thread_id)
    log.info("Incident received")

    incident_input = request.model_dump()
    # Flatten timeline events to dicts
    incident_input["timeline_events"] = [e.model_dump() for e in request.timeline_events]

    graph = get_graph()
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    final_state = None
    interrupted = False

    try:
        for event in graph.stream(incident_input, config=config, stream_mode="values"):
            final_state = event
    except Exception as exc:
        # LangGraph raises its internal Interrupt signal — catch it here
        if "Interrupt" in type(exc).__name__:
            interrupted = True
            final_state = graph.get_state(config).values
        else:
            log.error("Workflow error", error=str(exc))
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    if final_state is None:
        raise HTTPException(status_code=500, detail="Workflow produced no output.")

    # Persist thread for potential HITL resume
    _thread_store[thread_id] = {"config": config, "incident_id": request.incident_id}

    status = (
        "pending_review" if (interrupted or final_state.get("requires_human_review") and not final_state.get("final_response"))
        else ("error" if final_state.get("error") else "completed")
    )

    log.info("Workflow complete", status=status, risk_score=final_state.get("risk_score"))

    return IncidentResponse(
        incident_id=request.incident_id,
        thread_id=thread_id,
        status=status,
        risk_score=final_state.get("risk_score"),
        requires_human_review=final_state.get("requires_human_review", False),
        root_cause_analysis=final_state.get("root_cause_analysis"),
        remediation_steps=final_state.get("remediation_steps", []),
        confidence_score=final_state.get("confidence_score"),
        final_response=final_state.get("final_response"),
        agent_messages=final_state.get("agent_messages", []),
        risk_factors=final_state.get("risk_factors", []),
        error=final_state.get("error"),
    )


@router.get("/{incident_id}", response_model=IncidentResponse)
async def get_incident_status(incident_id: str) -> IncidentResponse:
    """Retrieve current status of a submitted incident by ID (most recent thread)."""
    thread_id: str | None = None
    thread_info: ThreadInfo | None = None
    for tid, info in reversed(_thread_store.items()):
        if info["incident_id"] == incident_id:
            thread_id, thread_info = tid, info
            break

    if thread_id is None or thread_info is None:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found in memory store.")

    graph = get_graph()
    final_state = graph.get_state(thread_info["config"]).values

    status = (
        "pending_review" if (final_state.get("requires_human_review") and not final_state.get("final_response"))
        else ("error" if final_state.get("error") else "completed")
    )

    return IncidentResponse(
        incident_id=incident_id,
        thread_id=thread_id,
        status=status,
        risk_score=final_state.get("risk_score"),
        requires_human_review=final_state.get("requires_human_review", False),
        root_cause_analysis=final_state.get("root_cause_analysis"),
        remediation_steps=final_state.get("remediation_steps", []),
        confidence_score=final_state.get("confidence_score"),
        final_response=final_state.get("final_response"),
        agent_messages=final_state.get("agent_messages", []),
        risk_factors=final_state.get("risk_factors", []),
        error=final_state.get("error"),
    )
