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


class TestParentSectionDeduplication:
    """Multiple child-chunk hits from the same parent section should collapse
    into a single retrieved doc whose content is the full parent section —
    not the small fragment that was actually matched."""

    def _make_child_doc(self, fragment: str, parent_content: str, parent_id: str, score: float):
        doc = MagicMock()
        doc.page_content = fragment
        doc.metadata = {
            "source": "runbooks/db.txt",
            "section_title": "REMEDIATION STEPS",
            "parent_id": parent_id,
            "parent_content": parent_content,
        }
        return doc, score

    @patch("agents.retriever.agent.get_vector_store")
    def test_multiple_children_of_same_parent_collapse_to_one_result(self, mock_factory, sample_state):
        parent_text = "REMEDIATION STEPS\n1. Restart pods\n2. Kill idle connections\n3. Scale out"
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = [
            self._make_child_doc("1. Restart pods", parent_text, "runbooks/db.txt::section-2", 0.95),
            self._make_child_doc("2. Kill idle connections", parent_text, "runbooks/db.txt::section-2", 0.90),
        ]
        mock_factory.return_value = mock_store

        result = retriever_agent(sample_state)

        assert len(result["retrieved_docs"]) == 1
        assert result["retrieved_docs"][0]["content"] == parent_text

    @patch("agents.retriever.agent.get_vector_store")
    def test_returned_content_is_the_parent_not_the_matched_fragment(self, mock_factory, sample_state):
        parent_text = "REMEDIATION STEPS\nFull section with much more detail than the fragment below."
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = [
            self._make_child_doc("Full section with", parent_text, "runbooks/db.txt::section-2", 0.9),
        ]
        mock_factory.return_value = mock_store

        result = retriever_agent(sample_state)

        assert result["retrieved_docs"][0]["content"] == parent_text
        assert result["retrieved_docs"][0]["content"] != "Full section with"

    @patch("agents.retriever.agent.get_vector_store")
    def test_different_parents_both_returned(self, mock_factory, sample_state):
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = [
            self._make_child_doc("frag A", "Parent A full text", "doc.txt::section-0", 0.9),
            self._make_child_doc("frag B", "Parent B full text", "doc.txt::section-1", 0.8),
        ]
        mock_factory.return_value = mock_store

        result = retriever_agent(sample_state)

        contents = {d["content"] for d in result["retrieved_docs"]}
        assert contents == {"Parent A full text", "Parent B full text"}

    @patch("agents.retriever.agent.get_vector_store")
    def test_docs_without_parent_metadata_fall_back_to_page_content(self, mock_factory, sample_state):
        """Backward compatibility with data ingested before hierarchical chunking existed."""
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = [
            self._make_mock_doc("Legacy flat chunk", 0.85),
        ]
        mock_factory.return_value = mock_store

        result = retriever_agent(sample_state)

        assert result["retrieved_docs"][0]["content"] == "Legacy flat chunk"
        assert result["retrieved_docs"][0]["section_title"] is None

    def _make_mock_doc(self, content="Fix: restart the service", score=0.85):
        doc = MagicMock()
        doc.page_content = content
        doc.metadata = {"source": "runbooks/db.txt"}
        return doc, score

    @patch("agents.retriever.agent.get_vector_store")
    def test_result_count_capped_at_retriever_top_k_after_dedup(self, mock_factory, sample_state):
        mock_store = MagicMock()
        # 3x retriever_top_k distinct-parent hits — should still cap at top_k.
        from config.settings import get_settings
        top_k = get_settings().retriever_top_k
        mock_store.similarity_search_with_score.return_value = [
            self._make_child_doc(f"frag {i}", f"Parent {i} text", f"doc.txt::section-{i}", 0.9 - i * 0.01)
            for i in range(top_k * 3)
        ]
        mock_factory.return_value = mock_store

        result = retriever_agent(sample_state)

        assert len(result["retrieved_docs"]) == top_k
