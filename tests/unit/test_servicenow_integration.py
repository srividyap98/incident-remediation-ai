"""
Unit tests for the ServiceNow integration adapter.

Covers request validation, the placeholder auth stub, org-routing
resolution, and payload translation — not the AI workflow itself (that's
covered by the agent-level unit tests and tests/integration/test_workflow.py).
The one end-to-end test here mocks the workflow graph the same way
test_retriever_agent.py mocks the vector store.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.routers.servicenow import (
    SAMPLE_SERVICENOW_PAYLOAD,
    ServiceNowIncidentPayload,
    _map_priority,
    _to_internal_request,
    resolve_org_route,
    verify_servicenow_client,
)
from config.settings import get_settings


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """get_settings() is @lru_cache'd; clear it so env changes in a test take effect."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ── Request validation ───────────────────────────────────────────────────────

class TestPayloadValidation:
    def test_sample_payload_is_valid(self):
        payload = ServiceNowIncidentPayload(**SAMPLE_SERVICENOW_PAYLOAD)
        assert payload.org == "acme-corp"
        assert payload.number == "INC0010042"

    def test_missing_required_field_rejected(self):
        bad = dict(SAMPLE_SERVICENOW_PAYLOAD)
        del bad["short_description"]
        with pytest.raises(ValidationError):
            ServiceNowIncidentPayload(**bad)

    def test_missing_org_rejected(self):
        bad = dict(SAMPLE_SERVICENOW_PAYLOAD)
        del bad["org"]
        with pytest.raises(ValidationError):
            ServiceNowIncidentPayload(**bad)

    def test_urgency_out_of_range_rejected(self):
        bad = dict(SAMPLE_SERVICENOW_PAYLOAD)
        bad["urgency"] = 9
        with pytest.raises(ValidationError):
            ServiceNowIncidentPayload(**bad)

    def test_short_description_too_short_rejected(self):
        bad = dict(SAMPLE_SERVICENOW_PAYLOAD)
        bad["short_description"] = "x"
        with pytest.raises(ValidationError):
            ServiceNowIncidentPayload(**bad)

    def test_work_notes_defaults_to_empty(self):
        minimal = dict(SAMPLE_SERVICENOW_PAYLOAD)
        del minimal["work_notes"]
        payload = ServiceNowIncidentPayload(**minimal)
        assert payload.work_notes == []


# ── Priority mapping ──────────────────────────────────────────────────────────

class TestPriorityMapping:
    @pytest.mark.parametrize("urgency,impact,expected", [
        (1, 1, "P1"),
        (1, 2, "P1"),
        (2, 1, "P1"),
        (1, 3, "P2"),
        (2, 2, "P2"),
        (3, 1, "P2"),
        (2, 3, "P3"),
        (3, 2, "P3"),
        (3, 3, "P4"),
    ])
    def test_matrix(self, urgency, impact, expected):
        assert _map_priority(urgency, impact) == expected


# ── Org routing ───────────────────────────────────────────────────────────────

class TestOrgRouting:
    def test_known_org_routes_to_its_team(self):
        route = resolve_org_route("acme-corp")
        assert route.team == "acme-platform-oncall"
        assert route.queue == "acme-incidents"

    def test_different_orgs_route_differently(self):
        assert resolve_org_route("acme-corp").team != resolve_org_route("globex").team

    def test_unknown_org_falls_back_to_default_route(self):
        route = resolve_org_route("some-org-that-does-not-exist")
        assert route.queue == "unrouted-incidents"
        assert route.team == "platform-oncall"


# ── Payload -> internal IncidentRequest translation ──────────────────────────

class TestPayloadConversion:
    def test_maps_core_fields(self):
        payload = ServiceNowIncidentPayload(**SAMPLE_SERVICENOW_PAYLOAD)
        internal = _to_internal_request(payload)

        assert internal.incident_id == payload.number
        assert internal.incident_title == payload.short_description
        assert internal.incident_description == payload.description
        assert internal.affected_service == payload.cmdb_ci
        assert internal.org == "acme-corp"

    def test_urgency_impact_maps_to_severity(self):
        payload = ServiceNowIncidentPayload(**SAMPLE_SERVICENOW_PAYLOAD)  # urgency=1, impact=1
        internal = _to_internal_request(payload)
        assert internal.severity == "P1"

    def test_falls_back_to_sys_id_when_number_missing(self):
        data = dict(SAMPLE_SERVICENOW_PAYLOAD)
        data["number"] = ""
        payload = ServiceNowIncidentPayload(**data)
        internal = _to_internal_request(payload)
        assert internal.incident_id == payload.sys_id

    def test_opened_at_becomes_timeline_event(self):
        payload = ServiceNowIncidentPayload(**SAMPLE_SERVICENOW_PAYLOAD)
        internal = _to_internal_request(payload)
        assert len(internal.timeline_events) == 1
        assert internal.timeline_events[0].source == "servicenow"

    def test_work_notes_become_raw_logs(self):
        payload = ServiceNowIncidentPayload(**SAMPLE_SERVICENOW_PAYLOAD)
        internal = _to_internal_request(payload)
        assert internal.raw_logs == payload.work_notes


# ── Placeholder auth stub ─────────────────────────────────────────────────────

class TestAuthStub:
    def test_no_secret_configured_allows_request(self, monkeypatch):
        monkeypatch.delenv("SERVICENOW_SHARED_SECRET", raising=False)
        get_settings.cache_clear()
        assert verify_servicenow_client(authorization=None) == "servicenow"

    def test_missing_header_rejected_when_secret_configured(self, monkeypatch):
        monkeypatch.setenv("SERVICENOW_SHARED_SECRET", "sn-test-secret")
        get_settings.cache_clear()
        with pytest.raises(HTTPException) as exc_info:
            verify_servicenow_client(authorization=None)
        assert exc_info.value.status_code == 401

    def test_non_bearer_header_rejected(self, monkeypatch):
        monkeypatch.setenv("SERVICENOW_SHARED_SECRET", "sn-test-secret")
        get_settings.cache_clear()
        with pytest.raises(HTTPException) as exc_info:
            verify_servicenow_client(authorization="Basic dXNlcjpwYXNz")
        assert exc_info.value.status_code == 401

    def test_wrong_token_rejected(self, monkeypatch):
        monkeypatch.setenv("SERVICENOW_SHARED_SECRET", "sn-test-secret")
        get_settings.cache_clear()
        with pytest.raises(HTTPException) as exc_info:
            verify_servicenow_client(authorization="Bearer wrong-token")
        assert exc_info.value.status_code == 401

    def test_correct_token_accepted(self, monkeypatch):
        monkeypatch.setenv("SERVICENOW_SHARED_SECRET", "sn-test-secret")
        get_settings.cache_clear()
        assert verify_servicenow_client(authorization="Bearer sn-test-secret") == "servicenow"


# ── End-to-end through FastAPI (AI workflow mocked out) ──────────────────────

class TestServiceNowEndpoint:
    def _mock_graph_stream(self):
        mock_graph = MagicMock()
        mock_graph.stream.return_value = iter([{
            "final_response": "Mocked remediation.",
            "remediation_steps": ["Step 1"],
            "risk_score": 0.4,
            "requires_human_review": False,
            "confidence_score": 0.9,
            "root_cause_analysis": "Mocked root cause.",
            "agent_messages": [],
            "risk_factors": [],
        }])
        return mock_graph

    @patch("api.routers.incidents.get_graph")
    def test_valid_request_routes_and_runs_workflow(self, mock_get_graph, monkeypatch):
        monkeypatch.delenv("SERVICENOW_SHARED_SECRET", raising=False)
        get_settings.cache_clear()
        mock_get_graph.return_value = self._mock_graph_stream()

        from api.main import app
        client = TestClient(app)

        response = client.post(
            "/api/v1/integrations/servicenow/incidents",
            json=SAMPLE_SERVICENOW_PAYLOAD,
        )

        assert response.status_code == 202
        body = response.json()
        assert body["servicenow_number"] == "INC0010042"
        assert body["org"] == "acme-corp"
        assert body["routed_team"] == "acme-platform-oncall"
        assert body["routed_queue"] == "acme-incidents"
        assert body["result"]["status"] == "completed"
        assert body["result"]["org"] == "acme-corp"

    @patch("api.routers.incidents.get_graph")
    def test_unrecognised_org_still_processed_via_default_route(self, mock_get_graph, monkeypatch):
        monkeypatch.delenv("SERVICENOW_SHARED_SECRET", raising=False)
        get_settings.cache_clear()
        mock_get_graph.return_value = self._mock_graph_stream()

        payload = dict(SAMPLE_SERVICENOW_PAYLOAD)
        payload["org"] = "unknown-org"

        from api.main import app
        client = TestClient(app)
        response = client.post("/api/v1/integrations/servicenow/incidents", json=payload)

        assert response.status_code == 202
        body = response.json()
        assert body["routed_queue"] == "unrouted-incidents"

    def test_invalid_payload_rejected_before_auth_or_workflow(self, monkeypatch):
        monkeypatch.delenv("SERVICENOW_SHARED_SECRET", raising=False)
        get_settings.cache_clear()

        bad_payload = dict(SAMPLE_SERVICENOW_PAYLOAD)
        del bad_payload["description"]

        from api.main import app
        client = TestClient(app)
        response = client.post("/api/v1/integrations/servicenow/incidents", json=bad_payload)

        assert response.status_code == 422

    def test_rejected_when_secret_configured_and_no_token_sent(self, monkeypatch):
        monkeypatch.setenv("SERVICENOW_SHARED_SECRET", "sn-test-secret")
        get_settings.cache_clear()

        from api.main import app
        client = TestClient(app)
        response = client.post(
            "/api/v1/integrations/servicenow/incidents",
            json=SAMPLE_SERVICENOW_PAYLOAD,
        )

        assert response.status_code == 401

    @patch("api.routers.incidents.get_graph")
    def test_accepted_when_secret_configured_and_correct_token_sent(self, mock_get_graph, monkeypatch):
        monkeypatch.setenv("SERVICENOW_SHARED_SECRET", "sn-test-secret")
        get_settings.cache_clear()
        mock_get_graph.return_value = self._mock_graph_stream()

        from api.main import app
        client = TestClient(app)
        response = client.post(
            "/api/v1/integrations/servicenow/incidents",
            json=SAMPLE_SERVICENOW_PAYLOAD,
            headers={"Authorization": "Bearer sn-test-secret"},
        )

        assert response.status_code == 202
