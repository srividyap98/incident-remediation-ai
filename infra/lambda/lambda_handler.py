"""
AWS Lambda Handler — Async Incident Processor
---------------------------------------------
Triggered by SQS or EventBridge for asynchronous incident processing.
Suitable for fire-and-forget submission where callers don't wait for
the full multi-agent workflow to complete.

Deploy via:
    aws lambda create-function \
        --function-name incident-ai-processor \
        --runtime python3.11 \
        --handler lambda_handler.handler \
        --role arn:aws:iam::<ACCOUNT>:role/incident-ai-lambda-role \
        --zip-file fileb://lambda.zip
"""
from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Callable

import boto3
import structlog

if TYPE_CHECKING:
    from agents.state import IncidentState

# Lazy imports — speed up cold start
logger = structlog.get_logger(__name__)


def _load_workflow() -> Callable[..., "IncidentState"]:
    """Import heavy deps only inside handler (reduces Lambda cold start time)."""
    from agents.orchestrator.graph import run_incident_workflow
    return run_incident_workflow


def _store_result(incident_id: str, result: dict[str, Any]) -> None:
    """Persist workflow result to DynamoDB."""
    dynamodb = boto3.resource("dynamodb", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    table = dynamodb.Table(os.environ.get("RESULTS_TABLE", "incident-ai-results"))
    table.put_item(
        Item={
            "incident_id": incident_id,
            "status": result.get("status", "unknown"),
            "risk_score": str(result.get("risk_score", "")),
            "final_response": result.get("final_response", ""),
            "remediation_steps": json.dumps(result.get("remediation_steps", [])),
            "requires_human_review": result.get("requires_human_review", False),
            "ttl": int(__import__("time").time()) + 60 * 60 * 24 * 30,  # 30-day TTL
        }
    )


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    Lambda entrypoint.

    Expects event body to be a JSON-encoded IncidentRequest, either:
    - Direct invocation: event = {incident_id, incident_title, ...}
    - SQS trigger:       event = {Records: [{body: "<json>"}]}
    - EventBridge:       event = {detail: {incident_id, ...}}
    """
    log = logger.bind(aws_request_id=getattr(context, "aws_request_id", "local"))
    log.info("Lambda invoked", event_keys=list(event.keys()))

    # Normalise event format
    if "Records" in event:
        # SQS trigger
        body = json.loads(event["Records"][0]["body"])
    elif "detail" in event:
        # EventBridge trigger
        body = event["detail"]
    else:
        # Direct invocation
        body = event

    incident_id = body.get("incident_id", "UNKNOWN")
    log = log.bind(incident_id=incident_id)

    try:
        run_workflow = _load_workflow()
        final_state = run_workflow(incident_input=body)

        result = {
            "status": "completed" if final_state.get("final_response") else "pending_review",
            "incident_id": incident_id,
            "risk_score": final_state.get("risk_score"),
            "requires_human_review": final_state.get("requires_human_review", False),
            "final_response": final_state.get("final_response"),
            "remediation_steps": final_state.get("remediation_steps", []),
        }

        _store_result(incident_id, result)
        log.info("Workflow completed", status=result["status"], risk_score=result["risk_score"])

        return {"statusCode": 200, "body": json.dumps(result)}

    except Exception as exc:
        log.error("Lambda handler error", error=str(exc))
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(exc), "incident_id": incident_id}),
        }
