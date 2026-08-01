"""
Arize Phoenix LLM Observability
---------------------------------
Logs LLM prompts, responses, grounding scores, and feedback ratings
to Arize Phoenix for quality monitoring and drift detection.

Requires ARIZE_API_KEY and ARIZE_SPACE_ID in .env.
Falls back gracefully if not configured.

Usage:
    from monitoring.arize_logger import log_llm_interaction
    log_llm_interaction(
        incident_id="INC-001",
        prompt=enriched_context,
        response=final_response,
        risk_score=0.82,
        confidence_score=0.91,
        grounding_warnings=["Low grounding (0.31): '...'"],
    )
"""
from __future__ import annotations

import structlog

from config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


def _get_phoenix_client():
    """Lazy-init Arize Phoenix client. Returns None if not configured."""
    try:
        import arize.utils.types as arize_types
        from arize.api import Client as ArizeClient

        api_key  = getattr(settings, "arize_api_key", None)
        space_id = getattr(settings, "arize_space_id", None)

        if not api_key or not space_id:
            return None, None

        client = ArizeClient(space_key=space_id, api_key=api_key)
        return client, arize_types
    except ImportError:
        return None, None
    except Exception as exc:
        logger.warning("Arize client init failed", error=str(exc))
        return None, None


def log_llm_interaction(
    incident_id: str,
    prompt: str,
    response: str,
    risk_score: float | None = None,
    confidence_score: float | None = None,
    grounding_warnings: list[str] | None = None,
    feedback_rating: float | None = None,
    retrieved_context: str | None = None,
    model_id: str | None = None,
) -> None:
    """
    Log a complete LLM interaction to Arize Phoenix.

    All parameters except incident_id and prompt are optional.
    The function is always safe to call — it never raises.
    """
    client, arize_types = _get_phoenix_client()

    record = {
        "incident_id": incident_id,
        "prompt_length": len(prompt),
        "response_length": len(response),
        "risk_score": risk_score,
        "confidence_score": confidence_score,
        "grounding_warning_count": len(grounding_warnings or []),
        "feedback_rating": feedback_rating,
        "model_id": model_id or settings.bedrock_llm_model_id,
    }

    # Always log locally regardless of Arize config
    logger.info("llm_interaction", **record)

    if client is None:
        return  # Arize not configured — local log only

    try:
        tags = {
            "incident_id": incident_id,
            "model_id": model_id or settings.bedrock_llm_model_id,
        }
        if risk_score is not None:
            tags["risk_score"] = str(round(risk_score, 3))
        if grounding_warnings:
            tags["grounding_warnings"] = str(len(grounding_warnings))

        # Build Arize record
        features = {
            "prompt": prompt[:2000],          # Arize has field size limits
            "retrieved_context": (retrieved_context or "")[:2000],
        }
        labels = {
            "response": response[:2000],
        }
        if confidence_score is not None:
            labels["confidence_score"] = str(confidence_score)
        if feedback_rating is not None:
            labels["feedback_rating"] = str(feedback_rating)

        response_obj = client.log(
            prediction_id=incident_id,
            model_id="incident-remediation-ai",
            model_version="2.0.0",
            prediction_label=labels.get("response", ""),
            features=features,
            tags=tags,
        )

        if response_obj and response_obj.status_code != 200:
            logger.warning("Arize log returned non-200", status=response_obj.status_code)

    except Exception as exc:
        # Observability failures must never break the main workflow
        logger.warning("Arize logging failed", error=str(exc))


def log_grounding_scores(incident_id: str, scores: dict[str, float]) -> None:
    """Log per-claim grounding scores to Arize as a batch."""
    if not scores:
        return
    avg = sum(scores.values()) / len(scores)
    logger.info("grounding_scores", incident_id=incident_id, avg_score=round(avg, 3), num_claims=len(scores))
    # In production: send to Arize as a custom metric or embedding record
