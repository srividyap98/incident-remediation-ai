"""
Context Builder Agent
---------------------
Implements Model Context Protocol (MCP)-style context assembly:
  - Combines structured incident metadata with unstructured retrieved docs + logs.
  - Produces a single enriched_context string ready for the LLM prompt.
  - Enforces a token budget so the context never exceeds LLM limits.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import structlog

from agents.state import IncidentState
from config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# Rough char-to-token ratio for Claude models (conservative)
_CHARS_PER_TOKEN = 3.5
_CONTEXT_TOKEN_BUDGET = 6_000   # Reserve headroom for the full prompt + output


def _token_estimate(text: str) -> int:
    return int(len(text) / _CHARS_PER_TOKEN)


def _format_structured_metadata(state: IncidentState) -> dict:
    """Extract structured fields into a clean metadata dict."""
    return {
        "incident_id": state.get("incident_id", "N/A"),
        "title": state.get("incident_title", "N/A"),
        "severity": state.get("severity", "N/A"),
        "affected_service": state.get("affected_service", "N/A"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "timeline_events": state.get("timeline_events", [])[:10],  # Cap for context
    }


def _build_docs_section(docs: list[dict], budget_chars: int) -> str:
    """Fit as many retrieved docs as possible within the char budget."""
    sections: list[str] = []
    used = 0
    for i, doc in enumerate(docs, 1):
        section_title = doc.get("section_title")
        header = f"[Doc {i} | Source: {doc['source']}"
        if section_title:
            header += f" | Section: {section_title}"
        header += f" | Relevance: {doc['score']:.2f}]"
        snippet = f"{header}\n{doc['content']}"
        if used + len(snippet) > budget_chars:
            break
        sections.append(snippet)
        used += len(snippet)
    return "\n\n".join(sections) if sections else "No relevant historical documents found."


def _build_logs_section(logs: list[str], budget_chars: int) -> str:
    """Include as many log lines as fit in the budget."""
    combined = "\n".join(logs)
    return combined[:budget_chars] if len(combined) > budget_chars else combined


_CONTEXT_TEMPLATE = """\
=== INCIDENT METADATA ===
{metadata_json}

=== RETRIEVED HISTORICAL CONTEXT ===
{docs_section}

=== RAW LOG EXCERPTS ===
{logs_section}

=== RETRIEVAL QUERY USED ===
{retrieval_query}
""".strip()


def context_builder_agent(state: IncidentState) -> dict:
    """LangGraph node: assemble enriched prompt context from all sources."""
    log = logger.bind(agent="context_builder", incident_id=state.get("incident_id"))
    log.info("Building context")

    metadata = _format_structured_metadata(state)
    metadata_json = json.dumps(metadata, indent=2)

    # Split remaining token budget between docs and logs
    total_budget_chars = int(_CONTEXT_TOKEN_BUDGET * _CHARS_PER_TOKEN)
    meta_chars = len(metadata_json)
    remaining = max(0, total_budget_chars - meta_chars)
    docs_budget = int(remaining * 0.7)
    logs_budget = remaining - docs_budget

    docs_section = _build_docs_section(state.get("retrieved_docs", []), docs_budget)
    logs_section = _build_logs_section(state.get("raw_logs", []), logs_budget)

    enriched_context = _CONTEXT_TEMPLATE.format(
        metadata_json=metadata_json,
        docs_section=docs_section,
        logs_section=logs_section or "No logs provided.",
        retrieval_query=state.get("retrieval_query", "N/A"),
    )

    token_est = _token_estimate(enriched_context)
    log.info("Context assembled", estimated_tokens=token_est, docs=len(state.get("retrieved_docs", [])))

    return {
        "enriched_context": enriched_context,
        "structured_metadata": metadata,
        "current_agent": "context_builder",
        "agent_messages": [
            f"[ContextBuilder] Assembled context (~{token_est} tokens, "
            f"{len(state.get('retrieved_docs', []))} docs, "
            f"{len(state.get('raw_logs', []))} log lines)"
        ],
    }
