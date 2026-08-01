"""Human-in-the-loop resume endpoint."""
from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException
from langgraph.types import Command

from agents.orchestrator.graph import get_graph
from api.auth import ANONYMOUS, get_current_principal
from api.schemas.models import HITLResumeRequest, IncidentResponse

logger = structlog.get_logger(__name__)
router = APIRouter()

# Import the same thread store from incidents router
from api.routers.incidents import _thread_store


@router.post("/resume", response_model=IncidentResponse)
async def resume_after_review(
    request: HITLResumeRequest,
    principal: str = Depends(get_current_principal),
) -> IncidentResponse:
    """
    Resume a paused incident workflow after human review.

    POST /api/v1/hitl/resume
    {
        "thread_id": "<uuid from original response>",
        "approved": true,
        "review_notes": "Verified root cause, proceed."
    }

    reviewer_id is derived from the authenticated caller (X-API-Key), not
    client-supplied — this endpoint approves risky remediation actions, so
    the audit trail must reflect who actually authenticated, not a
    self-reported name.
    """
    thread_info = _thread_store.get(request.thread_id)
    if not thread_info:
        raise HTTPException(status_code=404, detail=f"Thread {request.thread_id} not found.")

    reviewer_id = principal if principal != ANONYMOUS else (request.reviewer_id or ANONYMOUS)
    log = logger.bind(thread_id=request.thread_id, approved=request.approved)
    log.info("HITL resume requested", reviewer=reviewer_id)

    graph = get_graph()
    config = thread_info["config"]

    final_state = None
    try:
        for event in graph.stream(
            Command(resume=request.approved),
            config=config,
            stream_mode="values",
        ):
            final_state = event
    except Exception as exc:
        log.error("HITL resume error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if final_state is None:
        raise HTTPException(status_code=500, detail="Resume produced no output.")

    status = "completed" if final_state.get("final_response") else ("error" if final_state.get("error") else "pending_review")

    return IncidentResponse(
        incident_id=thread_info["incident_id"],
        thread_id=request.thread_id,
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
