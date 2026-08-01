"""Unit tests for Context Builder and Risk Evaluator agents."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.context_builder.agent import context_builder_agent
from agents.risk_evaluator.agent import _heuristic_score, risk_evaluator_agent
from agents.state import IncidentState


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def base_state() -> IncidentState:
    return IncidentState(
        incident_id="INC-TEST-002",
        incident_title="Memory leak in order-service",
        incident_description="Order service pods are OOMKilled every 2 hours.",
        severity="P2",
        affected_service="order-service",
        timeline_events=[
            {"timestamp": "2024-11-01T10:00:00Z", "event": "Alerts fired", "source": "pagerduty"},
            {"timestamp": "2024-11-01T10:05:00Z", "event": "Engineers engaged", "source": "slack"},
        ],
        raw_logs=["OOMKilled: container exceeded memory limit 512Mi", "GC overhead limit exceeded"],
        retrieved_docs=[
            {
                "content": "To resolve OOM issues: increase pod memory limit or investigate heap dumps.",
                "source": "runbooks/memory.txt",
                "score": 0.88,
                "metadata": {},
            }
        ],
        retrieval_query="Memory leak order-service OOMKilled",
    )


# ── Context Builder tests ─────────────────────────────────────────────────────

class TestContextBuilderAgent:
    def test_returns_enriched_context(self, base_state):
        result = context_builder_agent(base_state)
        assert "enriched_context" in result
        assert len(result["enriched_context"]) > 100

    def test_context_contains_metadata(self, base_state):
        result = context_builder_agent(base_state)
        assert "INC-TEST-002" in result["enriched_context"]
        assert "order-service" in result["enriched_context"]

    def test_context_contains_retrieved_docs(self, base_state):
        result = context_builder_agent(base_state)
        assert "heap dumps" in result["enriched_context"]

    def test_context_contains_logs(self, base_state):
        result = context_builder_agent(base_state)
        assert "OOMKilled" in result["enriched_context"]

    def test_structured_metadata_dict(self, base_state):
        result = context_builder_agent(base_state)
        assert isinstance(result["structured_metadata"], dict)
        assert result["structured_metadata"]["severity"] == "P2"

    def test_empty_docs_handled(self, base_state):
        base_state["retrieved_docs"] = []
        result = context_builder_agent(base_state)
        assert "No relevant historical documents" in result["enriched_context"]

    def test_agent_message_logged(self, base_state):
        result = context_builder_agent(base_state)
        assert any("ContextBuilder" in m for m in result["agent_messages"])


# ── Risk Evaluator tests ──────────────────────────────────────────────────────

class TestHeuristicScore:
    @pytest.mark.parametrize("severity,expected_min", [
        ("P1", 0.85), ("P2", 0.6), ("P3", 0.3), ("P4", 0.05),
    ])
    def test_severity_scaling(self, severity, expected_min):
        state = IncidentState(severity=severity, timeline_events=[])
        score = _heuristic_score(state)
        assert score >= expected_min

    def test_many_events_boost_score(self):
        state = IncidentState(
            severity="P3",
            timeline_events=[{"timestamp": str(i), "event": f"e{i}", "source": "x"} for i in range(10)],
        )
        base_state = IncidentState(severity="P3", timeline_events=[])
        assert _heuristic_score(state) > _heuristic_score(base_state)


class TestRiskEvaluatorAgent:
    def _mock_bedrock_response(self, score=0.75, factors=None):
        factors = factors or ["High blast radius", "No circuit breaker", "Peak traffic"]
        response = {
            "output": {
                "message": {
                    "content": [{"text": json.dumps({"risk_score": score, "risk_factors": factors})}]
                }
            }
        }
        return response

    @patch("agents.risk_evaluator.agent.boto3.client")
    def test_blended_score_computed(self, mock_boto, base_state):
        mock_client = MagicMock()
        mock_client.converse.return_value = self._mock_bedrock_response(score=0.80)
        mock_boto.return_value = mock_client
        base_state["enriched_context"] = "some context"

        result = risk_evaluator_agent(base_state)

        # Blended = 0.3 * heuristic + 0.7 * llm_score
        assert 0.0 <= result["risk_score"] <= 1.0
        assert result["risk_factors"] != []

    @patch("agents.risk_evaluator.agent.boto3.client")
    def test_hitl_triggered_above_threshold(self, mock_boto, base_state):
        mock_client = MagicMock()
        mock_client.converse.return_value = self._mock_bedrock_response(score=0.95)
        mock_boto.return_value = mock_client
        base_state["enriched_context"] = "context"

        result = risk_evaluator_agent(base_state)
        assert result["requires_human_review"] is True

    @patch("agents.risk_evaluator.agent.boto3.client")
    def test_fallback_to_heuristic_on_llm_error(self, mock_boto, base_state):
        mock_client = MagicMock()
        mock_client.converse.side_effect = Exception("Bedrock throttled")
        mock_boto.return_value = mock_client
        base_state["enriched_context"] = "context"

        result = risk_evaluator_agent(base_state)

        # Should still return a score (heuristic fallback)
        assert "risk_score" in result
        assert result["risk_score"] > 0
