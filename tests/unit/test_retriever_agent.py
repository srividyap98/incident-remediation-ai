"""Unit tests for the Retriever Agent."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.retriever.agent import _build_query, retriever_agent
from agents.state import IncidentState


@pytest.fixture
def sample_state() -> IncidentState:
    return IncidentState(
        incident_id="INC-001",
        incident_title="Database connection pool exhausted",
        incident_description="All connections to PostgreSQL are being refused. Service is degraded.",
        severity="P1",
        affected_service="payments-api",
        timeline_events=[],
        raw_logs=["ERROR: connection refused", "WARN: pool timeout after 30s"],
    )


class TestBuildQuery:
    def test_includes_title(self, sample_state):
        q = _build_query(sample_state)
        assert "Database connection pool exhausted" in q

    def test_includes_severity(self, sample_state):
        q = _build_query(sample_state)
        assert "P1" in q

    def test_includes_service(self, sample_state):
        q = _build_query(sample_state)
        assert "payments-api" in q

    def test_truncates_long_description(self, sample_state):
        sample_state["incident_description"] = "x" * 1000
        q = _build_query(sample_state)
        # Description should be capped at 500 chars
        assert len(q) < 900


class TestRetrieverAgent:
    def _make_mock_doc(self, content="Fix: restart the service", score=0.85):
        doc = MagicMock()
        doc.page_content = content
        doc.metadata = {"source": "runbooks/db.txt"}
        return doc, score

    @patch("agents.retriever.agent.get_vector_store")
    def test_successful_retrieval(self, mock_factory, sample_state):
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = [
            self._make_mock_doc("Restart connection pool", 0.92),
            self._make_mock_doc("Check max_connections in postgres.conf", 0.78),
        ]
        mock_factory.return_value = mock_store

        result = retriever_agent(sample_state)

        assert len(result["retrieved_docs"]) == 2
        assert result["retrieved_docs"][0]["score"] == 0.92
        assert result["retrieved_docs"][0]["source"] == "runbooks/db.txt"
        assert "retrieval_query" in result
        assert "Retriever" in result["agent_messages"][0]

    @patch("agents.retriever.agent.get_vector_store")
    def test_empty_results(self, mock_factory, sample_state):
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = []
        mock_factory.return_value = mock_store

        result = retriever_agent(sample_state)

        assert result["retrieved_docs"] == []
        assert "0" in result["agent_messages"][0]

    @patch("agents.retriever.agent.get_vector_store")
    def test_error_handled_gracefully(self, mock_factory, sample_state):
        mock_factory.side_effect = ConnectionError("Vector store unavailable")

        result = retriever_agent(sample_state)

        assert "error" in result
        assert result["retrieved_docs"] == []
        assert "ERROR" in result["agent_messages"][0]
