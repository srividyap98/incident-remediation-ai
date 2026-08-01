"""
Voice / IVR Router
------------------
Accepts speech-to-text transcript input from voice channels
(Amazon Connect, Twilio, Genesys) and routes through the same
LangGraph workflow as the REST API.

Endpoint: POST /api/v1/voice/incident

Integration pattern:
  1. IVR system captures caller speech and runs STT (e.g. Amazon Transcribe)
  2. Transcript POSTed here
  3. This router normalises into IncidentRequest + runs workflow
  4. Response text returned for TTS playback to caller
"""
from __future__ import annotations

import re
import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from agents.orchestrator.graph import get_graph
from api.auth import get_current_principal
from api.schemas.models import IncidentResponse
from config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()
router = APIRouter(dependencies=[Depends(get_current_principal)])


class VoiceIncidentRequest(BaseModel):
    transcript: str
    caller_id: str | None = None
    service_hint: str | None = None    # e.g. "payments" hinted by IVR menu selection


class VoiceIncidentResponse(BaseModel):
    incident_id: str
    status: str
    tts_response: str                  # Text formatted for text-to-speech playback
    full_result: IncidentResponse


def _extract_severity(transcript: str) -> str:
    """Infer severity from language in the transcript."""
    t = transcript.lower()
    if any(w in t for w in ["critical", "down", "outage", "all", "complete", "revenue", "payment"]):
        return "P1"
    if any(w in t for w in ["degraded", "slow", "high latency", "errors", "failing"]):
        return "P2"
    if any(w in t for w in ["intermittent", "occasional", "some"]):
        return "P3"
    return "P4"


def _normalise_transcript(transcript: str, service_hint: str | None) -> dict:
    """Convert a raw transcript into IncidentRequest-compatible fields."""
    # Clean transcript
    clean = re.sub(r"\s+", " ", transcript.strip())
    # Use first sentence as title (up to 120 chars)
    title = re.split(r"[.!?]", clean)[0][:120].strip() or "Voice-reported incident"

    return {
        "incident_id": f"VOICE-{uuid.uuid4().hex[:8].upper()}",
        "incident_title": title,
        "incident_description": clean,
        "severity": _extract_severity(clean),
        "affected_service": service_hint or "unknown-voice",
        "timeline_events": [],
        "raw_logs": [],
    }


def _format_tts_response(result: IncidentResponse) -> str:
    """Format the AI response for text-to-speech playback."""
    if result.status == "error":
        return "I was unable to process this incident. Please contact your on-call engineer directly."

    if result.status == "pending_review":
        score = result.risk_score or 0
        return (
            f"This incident has a high risk score of {score:.0%}. "
            "It has been escalated for human review. "
            "Your on-call engineer has been notified and will follow up shortly."
        )

    steps = result.remediation_steps
    rca = result.root_cause_analysis or ""
    confidence = result.confidence_score or 0

    tts = f"I've analysed the incident. {rca} "
    if steps:
        tts += f"I recommend {len(steps)} remediation steps. "
        tts += f"The first step is: {steps[0]}. "
        if len(steps) > 1:
            tts += f"The second step is: {steps[1]}. "
    tts += f"My confidence in this assessment is {confidence:.0%}. "
    tts += "Full details have been logged to the incident management system."
    return tts


@router.post("/incident", response_model=VoiceIncidentResponse)
async def voice_incident(request: VoiceIncidentRequest) -> VoiceIncidentResponse:
    """
    Accept a speech-to-text transcript and run the full incident workflow.
    Returns a tts_response string ready for text-to-speech playback.
    """
    log = logger.bind(caller_id=request.caller_id, service_hint=request.service_hint)
    log.info("Voice incident received", transcript_len=len(request.transcript))

    if len(request.transcript.strip()) < 10:
        raise HTTPException(status_code=422, detail="Transcript too short to process.")

    incident_input = _normalise_transcript(request.transcript, request.service_hint)
    log = log.bind(incident_id=incident_input["incident_id"], severity=incident_input["severity"])
    log.info("Normalised voice transcript", title=incident_input["incident_title"])

    graph = get_graph()
    config: RunnableConfig = {"configurable": {"thread_id": incident_input["incident_id"]}}

    final_state = None
    interrupted = False
    try:
        for event in graph.stream(incident_input, config=config, stream_mode="values"):
            final_state = event
    except Exception as exc:
        if "Interrupt" in type(exc).__name__:
            interrupted = True
            final_state = graph.get_state(config).values
        else:
            log.error("Voice workflow error", error=str(exc))
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not final_state:
        raise HTTPException(status_code=500, detail="Workflow produced no output.")

    status = (
        "pending_review" if interrupted or (final_state.get("requires_human_review") and not final_state.get("final_response"))
        else ("error" if final_state.get("error") else "completed")
    )

    full_result = IncidentResponse(
        incident_id=incident_input["incident_id"],
        thread_id=incident_input["incident_id"],
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

    tts = _format_tts_response(full_result)
    log.info("Voice response generated", status=status, tts_len=len(tts))

    return VoiceIncidentResponse(
        incident_id=incident_input["incident_id"],
        status=status,
        tts_response=tts,
        full_result=full_result,
    )
