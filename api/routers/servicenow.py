"""
ServiceNow Integration Router
------------------------------
Adapter endpoint for ServiceNow. Validates an inbound ServiceNow-shaped
incident payload, resolves org-based routing, translates it into our
internal IncidentRequest, and runs it through the *existing* AI workflow via
api.routers.incidents._run_workflow_and_build_response — this file only adds
translation, auth, and routing. It does not change the AI workflow itself.

Assumes ServiceNow is the only caller of this endpoint (see verify_servicenow_client
below for the placeholder auth model).

POST /api/v1/integrations/servicenow/incidents
"""
from __future__ import annotations

import secrets
from typing import Any, Literal

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from api.routers.incidents import _run_workflow_and_build_response
from api.schemas.models import IncidentRequest, IncidentResponse, TimelineEvent
from config.settings import get_settings

logger = structlog.get_logger(__name__)
router = APIRouter()


# ── Inbound payload ──────────────────────────────────────────────────────────

class ServiceNowIncidentPayload(BaseModel):
    """
    Placeholder shape for an inbound ServiceNow Incident record.

    Field names loosely mirror ServiceNow's `incident` table (sys_id, number,
    short_description, urgency/impact, cmdb_ci, ...). This is NOT a full or
    exact mirror of the real Table API schema — just enough to demonstrate
    the mapping. Replace with the actual payload shape ServiceNow sends once
    this is wired to a real instance.
    """
    sys_id: str = Field(..., description="ServiceNow record sys_id (unique GUID)")
    number: str = Field(..., description="Human-readable incident number, e.g. INC0010042")
    short_description: str = Field(..., min_length=5, max_length=300)
    description: str = Field(..., min_length=10)
    urgency: int = Field(..., ge=1, le=3, description="ServiceNow urgency: 1=High, 2=Medium, 3=Low")
    impact: int = Field(..., ge=1, le=3, description="ServiceNow impact: 1=High, 2=Medium, 3=Low")
    cmdb_ci: str = Field(..., description="Configuration item / affected service name")
    org: str = Field(..., description="Org identifier used to route this incident internally")
    caller_id: str | None = Field(default=None, description="ServiceNow sys_id of the reporting user")
    opened_at: str | None = Field(default=None, description="ServiceNow-formatted open timestamp")
    work_notes: list[str] = Field(default_factory=list, description="Existing work notes / log lines, if any")


# Example of what a real ServiceNow Flow/IntegrationHub REST step would POST
# here. Useful for local testing:
#   curl -X POST http://localhost:8000/api/v1/integrations/servicenow/incidents \
#     -H "Authorization: Bearer $SERVICENOW_SHARED_SECRET" \
#     -H "Content-Type: application/json" \
#     -d '<this dict, as JSON>'
SAMPLE_SERVICENOW_PAYLOAD: dict[str, Any] = {
    "sys_id": "8a4d9f3ee1b1a4109d3cf5b1846d43ab",
    "number": "INC0010042",
    "short_description": "Payments API returning 503 errors",
    "description": "Customers report checkout failures. Payments API health check failing since 09:14 UTC.",
    "urgency": 1,
    "impact": 1,
    "cmdb_ci": "payments-api",
    "org": "acme-corp",
    "caller_id": "6816f79cc0a8016401c5a33be04be441",
    "opened_at": "2026-08-01 09:15:00",
    "work_notes": ["Auto-detected by monitoring at 09:14 UTC"],
}


# ── Outbound response ────────────────────────────────────────────────────────

class ServiceNowIncidentResponse(BaseModel):
    """
    Response returned to ServiceNow. Wraps the standard IncidentResponse with
    the org-routing decision, so a ServiceNow Flow/IntegrationHub action can
    use routed_team / routed_queue to set assignment_group (or similar) on
    the originating record without needing to understand our internal AI
    workflow's response shape.
    """
    servicenow_number: str
    org: str
    routed_team: str
    routed_queue: str
    result: IncidentResponse


# ── Org routing ───────────────────────────────────────────────────────────────

class OrgRoute(BaseModel):
    """Internal routing target resolved for a given org."""
    org: str
    team: str
    queue: str
    notify_channel: str


# Placeholder routing table. Real integration point: replace this static dict
# with a lookup against whatever actually owns org -> team/queue mappings
# (a database table, an internal config service, or a ServiceNow
# cmn_department / company table read via IntegrationHub).
_ORG_ROUTES: dict[str, OrgRoute] = {
    "acme-corp": OrgRoute(
        org="acme-corp", team="acme-platform-oncall",
        queue="acme-incidents", notify_channel="#acme-incidents",
    ),
    "globex": OrgRoute(
        org="globex", team="globex-sre",
        queue="globex-incidents", notify_channel="#globex-incidents",
    ),
}
_DEFAULT_ORG_ROUTE = OrgRoute(
    org="default", team="platform-oncall",
    queue="unrouted-incidents", notify_channel="#incident-triage",
)


def resolve_org_route(org: str) -> OrgRoute:
    """Resolve an org identifier to its internal routing target.

    Unrecognised orgs fall back to a default triage queue rather than being
    rejected — ServiceNow is the trusted caller here, so an unknown org is
    treated as "not yet configured for routing", not an error.
    """
    return _ORG_ROUTES.get(org, _DEFAULT_ORG_ROUTE)


# ── Placeholder auth ──────────────────────────────────────────────────────────

def verify_servicenow_client(authorization: str | None = Header(default=None)) -> str:
    """
    Placeholder credential check for the trusted ServiceNow integration user.

    This endpoint assumes ServiceNow is the only caller and authenticates it
    with a single shared secret sent as a bearer token:

        Authorization: Bearer <SERVICENOW_SHARED_SECRET>

    --- Okta swap-in point --------------------------------------------------
    Replace the body of this function with real OIDC validation when this
    integration moves off the shared secret:
      1. Keep extracting the bearer token the same way.
      2. Verify its signature against Okta's JWKS endpoint (e.g. via
         `authlib` or `python-jose`), checking `iss`, `aud`, and `exp`.
      3. Check a `scope` or `groups` claim confirms the caller is the
         ServiceNow integration user/app (e.g. groups contains
         "servicenow-integration").
      4. Return the token's subject claim instead of the literal string
         below. Callers of this dependency already treat the return value as
         an opaque "who authenticated" identifier, so nothing else needs to
         change.
    --------------------------------------------------------------------------
    """
    settings = get_settings()
    if not settings.servicenow_shared_secret:
        # No secret configured -> local/offline dev; mirrors api/auth.py's
        # "empty config = open" convention.
        return "servicenow"

    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token.")

    token = authorization.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(token, settings.servicenow_shared_secret):
        raise HTTPException(status_code=401, detail="Invalid ServiceNow credentials.")

    return "servicenow"


# ── Payload translation ──────────────────────────────────────────────────────

# ServiceNow priority matrix (urgency x impact -> our P1-P4). Real ServiceNow
# computes an equivalent `priority` field server-side from the same two
# inputs; recomputing it here keeps this adapter self-contained.
_Severity = Literal["P1", "P2", "P3", "P4"]

_PRIORITY_MATRIX: dict[tuple[int, int], _Severity] = {
    (1, 1): "P1", (1, 2): "P1", (1, 3): "P2",
    (2, 1): "P1", (2, 2): "P2", (2, 3): "P3",
    (3, 1): "P2", (3, 2): "P3", (3, 3): "P4",
}


def _map_priority(urgency: int, impact: int) -> _Severity:
    return _PRIORITY_MATRIX.get((urgency, impact), "P3")


def _to_internal_request(payload: ServiceNowIncidentPayload) -> IncidentRequest:
    """Map a ServiceNow-shaped payload onto our internal IncidentRequest.

    This is the seam where a real integration would also pull in any extra
    ServiceNow-side context (related CIs, prior incidents, CMDB relationships)
    before handing off to the AI workflow. Kept to a direct field mapping here.
    """
    timeline_events = []
    if payload.opened_at:
        timeline_events.append(
            TimelineEvent(timestamp=payload.opened_at, event="Incident opened in ServiceNow", source="servicenow")
        )

    return IncidentRequest(
        incident_id=payload.number or payload.sys_id,
        incident_title=payload.short_description,
        incident_description=payload.description,
        severity=_map_priority(payload.urgency, payload.impact),
        affected_service=payload.cmdb_ci,
        org=payload.org,
        timeline_events=timeline_events,
        raw_logs=payload.work_notes,
    )


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    "/incidents",
    response_model=ServiceNowIncidentResponse,
    status_code=202,
    summary="Submit a ServiceNow incident for AI-driven remediation",
)
async def submit_servicenow_incident(
    payload: ServiceNowIncidentPayload,
    _client: str = Depends(verify_servicenow_client),
) -> ServiceNowIncidentResponse:
    """
    Validate an inbound ServiceNow incident, resolve org-based routing, and
    run it through the same AI workflow used by /api/v1/incidents/.

    # --- ServiceNow Flow / IntegrationHub connects here --------------------
    # In a real integration, a ServiceNow Flow (triggered on incident
    # create/update) or an IntegrationHub REST step would POST the payload
    # below to this URL, then:
    #   - use routed_team / routed_queue from the response to set
    #     assignment_group on the originating incident record, and
    #   - use result.final_response / result.remediation_steps to populate
    #     work_notes via a follow-up ServiceNow Table API call.
    # -------------------------------------------------------------------------
    """
    log = logger.bind(servicenow_number=payload.number, org=payload.org)
    route = resolve_org_route(payload.org)
    log.info("ServiceNow incident received", routed_team=route.team, routed_queue=route.queue)

    internal_request = _to_internal_request(payload)
    result = await _run_workflow_and_build_response(internal_request)

    return ServiceNowIncidentResponse(
        servicenow_number=payload.number,
        org=payload.org,
        routed_team=route.team,
        routed_queue=route.queue,
        result=result,
    )
