"""
Risk Evaluator Agent
--------------------
Evaluates the incident risk score using a combination of:
  - Heuristic rules on structured metadata (severity, service tier)
  - LLM-based risk factor extraction from enriched context

Writes risk_score, risk_factors, and requires_human_review to state.
"""
from __future__ import annotations

import re

import boto3
import structlog

from agents.state import IncidentState
from config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# Severity base scores (P1 = highest risk)
_SEVERITY_SCORES: dict[str, float] = {"P1": 0.9, "P2": 0.65, "P3": 0.35, "P4": 0.1}

_RISK_PROMPT = """\
You are a risk analyst for enterprise incident management. Analyse the following incident context and provide:
1. A risk score between 0.0 (no risk) and 1.0 (critical/catastrophic risk).
2. A list of the top 5 specific risk factors driving the score.

Respond ONLY in this exact JSON format with no additional text:
{{
  "risk_score": <float 0.0-1.0>,
  "risk_factors": ["factor 1", "factor 2", "factor 3", "factor 4", "factor 5"]
}}

INCIDENT CONTEXT:
{context}
""".strip()


def _heuristic_score(state: IncidentState) -> float:
    """Fast, deterministic baseline score from structured metadata."""
    severity = state.get("severity", "P4").upper()
    base = _SEVERITY_SCORES.get(severity, 0.1)

    # Boost if multiple timeline events (complex incident)
    events = state.get("timeline_events", [])
    if len(events) > 5:
        base = min(1.0, base + 0.1)

    return base


def _llm_risk_assessment(context: str) -> tuple[float, list[str]]:
    """Call Bedrock Claude for nuanced risk scoring."""
    client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
    prompt = _RISK_PROMPT.format(context=context[:4000])  # Token guard

    response = client.converse(
        modelId=settings.bedrock_llm_model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 300, "temperature": 0.1},
    )

    raw = response["output"]["message"]["content"][0]["text"].strip()

    # Extract JSON safely
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"LLM returned non-JSON risk response: {raw[:200]}")

    import json
    data = json.loads(json_match.group())
    score = float(data.get("risk_score", 0.5))
    score = max(0.0, min(1.0, score))
    factors = data.get("risk_factors", [])[:5]
    return score, factors


def risk_evaluator_agent(state: IncidentState) -> dict:
    """LangGraph node: evaluate incident risk and flag for HITL if needed."""
    log = logger.bind(agent="risk_evaluator", incident_id=state.get("incident_id"))
    log.info("Starting risk evaluation")

    heuristic = _heuristic_score(state)
    llm_score: float | None = None
    risk_factors: list[str] = []

    try:
        llm_score, risk_factors = _llm_risk_assessment(state.get("enriched_context", ""))
        # Blend heuristic (30%) with LLM (70%)
        final_score = round(0.3 * heuristic + 0.7 * llm_score, 3)
        log.info("Risk scored", heuristic=heuristic, llm=llm_score, final=final_score)
    except Exception as exc:
        log.warning("LLM risk assessment failed, using heuristic only", error=str(exc))
        final_score = heuristic
        risk_factors = [f"Risk assessment degraded: {exc}"]

    requires_review = final_score >= settings.risk_score_threshold

    return {
        "risk_score": final_score,
        "risk_factors": risk_factors,
        "requires_human_review": requires_review,
        "current_agent": "risk_evaluator",
        "agent_messages": [
            f"[RiskEvaluator] Score={final_score:.3f} | "
            f"Threshold={settings.risk_score_threshold} | "
            f"HITL={'YES' if requires_review else 'NO'}"
        ],
    }
