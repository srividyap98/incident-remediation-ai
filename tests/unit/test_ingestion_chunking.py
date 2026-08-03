"""Unit tests for the hierarchical, document-aware chunker in rag/ingestion/pipeline.py."""
from __future__ import annotations

from langchain_core.documents import Document

from rag.ingestion.pipeline import (
    _CHUNK_SIZE,
    _chunk_document_hierarchically,
    _looks_like_header,
    _split_into_sections,
)

# Mirrors the real runbook format used in scripts/seed_dev_db.py, including
# the tricky cases: a parenthetical-qualified header, a hyphenated header,
# and a label-like sentence that must NOT be mistaken for a header.
_SAMPLE_RUNBOOK = """
RUNBOOK: Database Connection Pool Exhaustion
Service: Any service using PostgreSQL or MySQL
Severity: P1 / P2

SYMPTOMS
- Application logs show "too many connections"
- Health check endpoints return 503

ROOT CAUSES (most common)
1. Connection leak — connections opened but not properly closed
2. Sudden traffic spike overwhelming pool size

REMEDIATION STEPS
1. Immediate: Restart the affected service pods
2. Kill idle connections older than 10 minutes

POST-INCIDENT
- Review slow query log for queries >5s

ESCALATION: DBA team if DB is unresponsive after step 3.
""".strip()


class TestLooksLikeHeader:
    def test_all_caps_single_word_is_header(self):
        assert _looks_like_header("SYMPTOMS") == "SYMPTOMS"

    def test_all_caps_multi_word_is_header(self):
        assert _looks_like_header("REMEDIATION STEPS") == "REMEDIATION STEPS"

    def test_hyphenated_caps_is_header(self):
        assert _looks_like_header("POST-INCIDENT") == "POST-INCIDENT"

    def test_caps_with_parenthetical_is_header(self):
        assert _looks_like_header("ROOT CAUSES (most common)") == "ROOT CAUSES (most common)"

    def test_markdown_header_is_header(self):
        assert _looks_like_header("## Symptoms") == "Symptoms"

    def test_label_like_sentence_is_not_a_header(self):
        # Has lowercase content after the colon — a real sentence, not a header.
        assert _looks_like_header("ESCALATION: DBA team if DB is unresponsive after step 3.") is None

    def test_title_line_is_not_a_header(self):
        assert _looks_like_header("RUNBOOK: Database Connection Pool Exhaustion") is None

    def test_numbered_list_item_is_not_a_header(self):
        assert _looks_like_header("1. Connection leak — connections opened but not properly closed") is None

    def test_blank_line_is_not_a_header(self):
        assert _looks_like_header("   ") is None

    def test_long_line_is_not_a_header(self):
        assert _looks_like_header("A" * 61) is None


class TestSplitIntoSections:
    def test_detects_all_expected_sections(self):
        sections = _split_into_sections(_SAMPLE_RUNBOOK)
        titles = [title for title, _ in sections]
        assert "Preamble" in titles
        assert "SYMPTOMS" in titles
        assert "ROOT CAUSES (most common)" in titles
        assert "REMEDIATION STEPS" in titles
        assert "POST-INCIDENT" in titles

    def test_preamble_captures_text_before_first_header(self):
        sections = dict(_split_into_sections(_SAMPLE_RUNBOOK))
        assert "RUNBOOK: Database Connection Pool Exhaustion" in sections["Preamble"]
        assert "Severity: P1 / P2" in sections["Preamble"]

    def test_escalation_sentence_stays_inside_post_incident_section(self):
        sections = dict(_split_into_sections(_SAMPLE_RUNBOOK))
        assert "ESCALATION: DBA team" in sections["POST-INCIDENT"]

    def test_section_content_excludes_its_own_header_line(self):
        sections = dict(_split_into_sections(_SAMPLE_RUNBOOK))
        assert "SYMPTOMS" not in sections["SYMPTOMS"].splitlines()[0]

    def test_document_with_no_headers_becomes_single_section(self):
        plain = "Just a plain paragraph with no structure at all, spanning one section."
        sections = _split_into_sections(plain)
        assert len(sections) == 1
        assert sections[0][0] == "Preamble"
        assert sections[0][1] == plain

    def test_empty_text_yields_no_sections(self):
        assert _split_into_sections("") == []


class TestChunkDocumentHierarchically:
    def test_short_section_yields_one_child_equal_to_the_parent(self):
        doc = Document(page_content="SYMPTOMS\nShort symptom text.", metadata={"source": "test.txt"})
        children = _chunk_document_hierarchically(doc)
        symptom_children = [c for c in children if c.metadata["section_title"] == "SYMPTOMS"]
        assert len(symptom_children) == 1
        assert symptom_children[0].page_content == symptom_children[0].metadata["parent_content"]

    def test_long_section_splits_into_multiple_children_sharing_one_parent(self):
        long_body = "This sentence repeats to force a split. " * 60  # well over _CHUNK_SIZE
        assert len(long_body) > _CHUNK_SIZE
        doc = Document(
            page_content=f"SYMPTOMS\n{long_body}",
            metadata={"source": "test.txt"},
        )
        children = _chunk_document_hierarchically(doc)
        assert len(children) > 1
        parent_ids = {c.metadata["parent_id"] for c in children}
        assert len(parent_ids) == 1  # all children belong to the same parent section
        parent_contents = {c.metadata["parent_content"] for c in children}
        assert len(parent_contents) == 1
        # Every child is smaller than the full parent section it came from.
        assert all(len(c.page_content) < len(next(iter(parent_contents))) for c in children)

    def test_metadata_is_preserved_and_extended(self):
        doc = Document(
            page_content="SYMPTOMS\nSomething broke.",
            metadata={"source": "test.txt", "type": "local"},
        )
        children = _chunk_document_hierarchically(doc)
        assert children[0].metadata["source"] == "test.txt"
        assert children[0].metadata["type"] == "local"
        assert "parent_id" in children[0].metadata
        assert "parent_content" in children[0].metadata

    def test_distinct_sections_get_distinct_parent_ids(self):
        doc = Document(page_content=_SAMPLE_RUNBOOK, metadata={"source": "runbook.txt"})
        children = _chunk_document_hierarchically(doc)
        parent_ids_by_section = {c.metadata["section_title"]: c.metadata["parent_id"] for c in children}
        assert parent_ids_by_section["SYMPTOMS"] != parent_ids_by_section["REMEDIATION STEPS"]

    def test_full_sample_runbook_round_trips_without_losing_content(self):
        doc = Document(page_content=_SAMPLE_RUNBOOK, metadata={"source": "runbook.txt"})
        children = _chunk_document_hierarchically(doc)
        # Every section's full text should appear intact as some child's parent_content.
        parent_contents = {c.metadata["parent_content"] for c in children}
        assert any("Kill idle connections" in pc for pc in parent_contents)
        assert any("Review slow query log" in pc for pc in parent_contents)
