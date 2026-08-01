"""
Feedback Router
---------------
Collects engineer ratings on AI-generated remediation steps.
Data feeds into the evaluation/feedback_loop.py aggregation pipeline
and MLflow for experiment tracking.

POST /api/v1/feedback
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.auth import ANONYMOUS, get_current_principal
from config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()
router = APIRouter()

_FEEDBACK_STORE = Path("data/feedback.jsonl")   # Append-only JSONL; swap for DynamoDB in prod


class FeedbackRequest(BaseModel):
    incident_id: str
    rca_accuracy: int           # 1–5
    remediation_usefulness: int  # 1–5
    correction_notes: str | None = None
    reviewer_id: str | None = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    incident_id: str
    status: str
    average_rating: float


@router.post("/", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    principal: str = Depends(get_current_principal),
) -> FeedbackResponse:
    """Record engineer feedback on a completed AI remediation."""
    feedback_id = str(uuid.uuid4())
    avg = (request.rca_accuracy + request.remediation_usefulness) / 2
    reviewer_id = principal if principal != ANONYMOUS else (request.reviewer_id or ANONYMOUS)

    record = {
        "feedback_id": feedback_id,
        "incident_id": request.incident_id,
        "rca_accuracy": request.rca_accuracy,
        "remediation_usefulness": request.remediation_usefulness,
        "average_rating": avg,
        "correction_notes": request.correction_notes,
        "reviewer_id": reviewer_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Persist to append-only JSONL
    _FEEDBACK_STORE.parent.mkdir(parents=True, exist_ok=True)
    with _FEEDBACK_STORE.open("a") as f:
        f.write(json.dumps(record) + "\n")

    logger.info(
        "Feedback recorded",
        incident_id=request.incident_id,
        avg_rating=avg,
        reviewer=reviewer_id,
    )

    # Log to MLflow if available
    try:
        import mlflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        with mlflow.start_run(run_name=f"feedback-{request.incident_id}"):
            mlflow.log_metrics({
                "feedback_rca_accuracy": request.rca_accuracy,
                "feedback_remediation_usefulness": request.remediation_usefulness,
                "feedback_average_rating": avg,
            })
            mlflow.log_param("incident_id", request.incident_id)
    except Exception as exc:
        logger.warning("MLflow feedback logging failed", error=str(exc))

    return FeedbackResponse(
        feedback_id=feedback_id,
        incident_id=request.incident_id,
        status="recorded",
        average_rating=avg,
    )
