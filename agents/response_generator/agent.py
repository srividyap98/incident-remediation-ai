"""
Response Generator Agent
------------------------
Calls AWS Bedrock (Claude Sonnet 4.5) with the fully enriched context to
generate:
  - Root cause analysis
  - Step-by-step remediation plan
  - Confidence score

Uses structured output (JSON mode) for reliable downstream parsing.
"""
from __future__ import annotations

import json
import re

import boto3
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from agents.state import IncidentState
from config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

_SYSTEM_PROMPT = """\
You are an expert Site Reliability Engineer and incident responder with deep knowledge of enterprise
distributed systems. You provide precise, actionable, evidence-based remediation guidance.

Always respond in valid JSON matching the schema provided. Never add commentary outside the JSON block.
""".strip()

_USER_PROMPT_TEMPLATE = """\
Analyse the following incident context and produce a structured remediation response.

{context}

Respond ONLY with valid JSON matching this exact schema:
{{
  "root_cause_analysis": "<2-4 sentence analysis of the likely root cause>",
  "remediation_steps": [
    "<Step 1: specific action>",
    "<Step 2: specific action>",
    "..."
  ],
  "confidence_score": <float 0.0-1.0>,
  "additional_notes": "<optional caveats or follow-up recommendations>"
}}
""".strip()


def _parse_response(raw: str) -> dict:
    """Extract and parse the JSON block from the LLM response."""
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON found in response: {raw[:300]}")
    return json.loads(json_match.group())


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    reraise=True,
)
def _call_bedrock(context: str) -> dict:
    client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
    prompt = _USER_PROMPT_TEMPLATE.format(context=context)

    response = client.converse(
        modelId=settings.bedrock_llm_model_id,
        system=[{"text": _SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={
            "maxTokens": settings.llm_max_tokens,
            "temperature": settings.llm_temperature,
        },
    )
    raw = response["output"]["message"]["content"][0]["text"].strip()
    return _parse_response(raw)


def response_generator_agent(state: IncidentState) -> dict:
    """LangGraph node: generate remediation response via Bedrock Claude."""
    log = logger.bind(agent="response_generator", incident_id=state.get("incident_id"))
    log.info("Generating remediation response")

    context = state.get("enriched_context", "")
    if not context:
        return {
            "error": "No enriched context available for response generation.",
            "agent_messages": ["[ResponseGenerator] ERROR: Missing enriched context."],
        }

    try:
        parsed = _call_bedrock(context)

        remediation_steps = parsed.get("remediation_steps", [])
        root_cause = parsed.get("root_cause_analysis", "")
        confidence = float(parsed.get("confidence_score", 0.5))
        notes = parsed.get("additional_notes", "")

        # Compose a human-readable final response
        final_response = (
            f"**Root Cause Analysis**\n{root_cause}\n\n"
            f"**Remediation Steps**\n"
            + "\n".join(f"{i}. {s}" for i, s in enumerate(remediation_steps, 1))
            + (f"\n\n**Notes**\n{notes}" if notes else "")
        )

        log.info(
            "Response generated",
            steps=len(remediation_steps),
            confidence=confidence,
        )

        return {
            "root_cause_analysis": root_cause,
            "remediation_steps": remediation_steps,
            "confidence_score": confidence,
            "final_response": final_response,
            "current_agent": "response_generator",
            "agent_messages": [
                f"[ResponseGenerator] Generated {len(remediation_steps)} remediation steps "
                f"(confidence={confidence:.2f})"
            ],
        }

    except Exception as exc:
        log.error("Response generation failed", error=str(exc))
        return {
            "error": f"Response generation error: {exc}",
            "agent_messages": [f"[ResponseGenerator] ERROR: {exc}"],
        }
