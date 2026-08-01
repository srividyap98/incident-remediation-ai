"""
Integration test — full LangGraph workflow with mocked AWS services.
Uses moto to mock Bedrock calls so no real AWS credentials are needed.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.orchestrator.graph import build_graph
from agents.state import IncidentState


LOW_RISK_INCIDENT = {
    "incident_id": "INC-INT-001",
    "incident_title": "Slow API responses on /search endpoint",
    "incident_description": "P4 degradation on search, no data loss, auto-scaling triggered.",
    "severity": "P4",
    "affected_service": "search-api",
    "timeline_events": [],
    "raw_logs": ["WARN: response time 2100ms > threshold 2000ms"],
}

HIGH_RISK_INCIDENT = {
    "incident_id": "INC-INT-002",
    "incident_title": "Payment processing database down",
    "incident_description": "Primary DB replica unreachable, transactions failing, revenue impact.",
    "severity": "P1",
    "affected_service": "payments-db",
    "timeline_events": [
        {"timestamp": "2024-11-01T09:00:00Z", "event": "DB unreachable", "source": "monitor"},
        {"timestamp": "2024-11-01T09:01:00Z", "event": "Failover failed", "source": "rds"},
        {"timestamp": "2024-11-01T09:02:00Z", "event": "Revenue alerts", "source": "datadog"},
    ],
    "raw_logs": ["FATAL: could not connect to server: Connection refused"] * 5,
}


def _mock_bedrock_client(risk_score: float = 0.4, confidence: float = 0.85):
    """Build a mock boto3 Bedrock client that returns sensible responses."""
    client = MagicMock()

    def converse_side_effect(**kwargs):
        prompt = kwargs.get("messages", [{}])[0].get("content", [{}])[0].get("text", "")
        # Risk evaluation call
        if "risk_score" in prompt.lower():
            body = json.dumps({
                "risk_score": risk_score,
                "risk_factors": ["DB unreachable", "Revenue impact", "Failover failed"],
            })
        else:
            # Response generation call
            body = json.dumps({
                "root_cause_analysis": "Primary DB replica lost quorum due to network partition.",
                "remediation_steps": [
                    "Promote read replica to primary via RDS console.",
                    "Update connection strings in application secrets.",
                    "Verify replication lag on new primary.",
                    "Run smoke tests on payment endpoint.",
                ],
                "confidence_score": confidence,
                "additional_notes": "Monitor for replication lag for 24h post-recovery.",
            })
        return {"output": {"message": {"content": [{"text": body}]}}}

    client.converse.side_effect = converse_side_effect
    return client


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.similarity_search_with_score.return_value = [
        (
            MagicMock(
                page_content="For DB connectivity issues: check RDS parameter groups and VPC security groups.",
                metadata={"source": "runbooks/db.txt"},
            ),
            0.88,
        )
    ]
    return store


class TestFullWorkflow:
    @patch("agents.risk_evaluator.agent.boto3.client")
    @patch("agents.response_generator.agent.boto3.client")
    @patch("agents.retriever.agent.get_vector_store")
    def test_low_risk_completes_without_hitl(
        self, mock_vs_factory, mock_resp_boto, mock_risk_boto, mock_vector_store
    ):
        mock_vs_factory.return_value = mock_vector_store
        bedrock_mock = _mock_bedrock_client(risk_score=0.2, confidence=0.9)
        mock_risk_boto.return_value = bedrock_mock
        mock_resp_boto.return_value = bedrock_mock

        graph = build_graph()
        config = {"configurable": {"thread_id": "test-low-risk"}}

        final = None
        for event in graph.stream(LOW_RISK_INCIDENT, config=config, stream_mode="values"):
            final = event

        assert final is not None
        assert final.get("final_response") is not None
        assert final.get("requires_human_review") is False
        assert len(final.get("remediation_steps", [])) > 0
        assert "agent_messages" in final

    @patch("agents.risk_evaluator.agent.boto3.client")
    @patch("agents.retriever.agent.get_vector_store")
    def test_high_risk_triggers_hitl_interrupt(
        self, mock_vs_factory, mock_risk_boto, mock_vector_store
    ):
        mock_vs_factory.return_value = mock_vector_store
        bedrock_mock = _mock_bedrock_client(risk_score=0.95)
        mock_risk_boto.return_value = bedrock_mock

        graph = build_graph()
        config = {"configurable": {"thread_id": "test-high-risk"}}

        interrupted = False
        final = None
        try:
            for event in graph.stream(HIGH_RISK_INCIDENT, config=config, stream_mode="values"):
                final = event
        except Exception as exc:
            if "Interrupt" in type(exc).__name__:
                interrupted = True

        # Either interrupted OR paused with requires_human_review
        if final:
            assert final.get("requires_human_review") is True or interrupted
        else:
            assert interrupted

    @patch("agents.risk_evaluator.agent.boto3.client")
    @patch("agents.response_generator.agent.boto3.client")
    @patch("agents.retriever.agent.get_vector_store")
    def test_all_agents_contribute_messages(
        self, mock_vs_factory, mock_resp_boto, mock_risk_boto, mock_vector_store
    ):
        mock_vs_factory.return_value = mock_vector_store
        bedrock_mock = _mock_bedrock_client(risk_score=0.2)
        mock_risk_boto.return_value = bedrock_mock
        mock_resp_boto.return_value = bedrock_mock

        graph = build_graph()
        config = {"configurable": {"thread_id": "test-messages"}}

        final = None
        for event in graph.stream(LOW_RISK_INCIDENT, config=config, stream_mode="values"):
            final = event

        messages = final.get("agent_messages", [])
        assert any("Retriever" in m for m in messages)
        assert any("ContextBuilder" in m for m in messages)
        assert any("RiskEvaluator" in m for m in messages)
        assert any("ResponseGenerator" in m for m in messages)
