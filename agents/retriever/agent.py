"""
Retriever Agent
---------------
Responsible for:
  1. Building an optimised semantic search query from the incident state.
  2. Querying the vector store for relevant historical incidents / runbooks.
  3. Writing retrieved_docs + retrieval_query back to state.
"""
from __future__ import annotations

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from agents.state import IncidentState
from config.settings import get_settings
from rag.vector_store.factory import get_vector_store

logger = structlog.get_logger(__name__)
settings = get_settings()

_QUERY_TEMPLATE = """
Incident: {title}
Severity: {severity}
Service: {service}
Description: {description}
""".strip()


def _build_query(state: IncidentState) -> str:
    """Construct a focused retrieval query from incident fields."""
    return _QUERY_TEMPLATE.format(
        title=state.get("incident_title", ""),
        severity=state.get("severity", "unknown"),
        service=state.get("affected_service", "unknown"),
        description=(state.get("incident_description", "")[:500]),
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _search(query: str) -> list[dict]:
    store = get_vector_store()
    results = store.similarity_search_with_score(query, k=settings.retriever_top_k)
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score": float(score),
            "metadata": doc.metadata,
        }
        for doc, score in results
    ]


def retriever_agent(state: IncidentState) -> dict:
    """LangGraph node: retrieve relevant documents for this incident."""
    log = logger.bind(
        agent="retriever",
        incident_id=state.get("incident_id"),
        severity=state.get("severity"),
    )
    log.info("Starting retrieval")

    query = _build_query(state)

    try:
        docs = _search(query)
        log.info("Retrieval complete", num_docs=len(docs))
        return {
            "retrieval_query": query,
            "retrieved_docs": docs,
            "current_agent": "retriever",
            "agent_messages": [
                f"[Retriever] Found {len(docs)} relevant documents "
                f"(top score: {docs[0]['score']:.3f})"
                if docs
                else f"[Retriever] Found {len(docs)} relevant documents."
            ],
        }
    except Exception as exc:
        log.error("Retrieval failed", error=str(exc))
        return {
            "retrieval_query": query,
            "retrieved_docs": [],
            "error": f"Retrieval error: {exc}",
            "agent_messages": [f"[Retriever] ERROR: {exc}"],
        }
