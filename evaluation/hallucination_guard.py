"""
Hallucination Guard
-------------------
Post-generation grounding check. Verifies that claims in the LLM
response are supported by text in the retrieved documents.

Uses embedding cosine similarity to score each remediation step
and the root cause analysis against retrieved document chunks.
Ungrounded assertions are flagged in IncidentState as grounding_warnings.

Called from the evaluator node in the LangGraph orchestrator.
"""
from __future__ import annotations

import numpy as np
import structlog

from config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# Minimum cosine similarity for a claim to be considered "grounded"
_GROUNDING_THRESHOLD = 0.40


def _cosine_sim(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def _embed(text: str) -> list[float]:
    """Embed a single string using Bedrock Titan (or offline mock)."""
    from rag.embeddings.bedrock import get_embeddings
    embeddings = get_embeddings()
    return embeddings.embed_query(text)


def _max_similarity(claim_vec: list[float], doc_vecs: list[list[float]]) -> float:
    """Return the highest cosine similarity between a claim and any doc chunk."""
    if not doc_vecs:
        return 0.0
    return max(_cosine_sim(claim_vec, dv) for dv in doc_vecs)


def check_grounding(
    claims: list[str],
    retrieved_docs: list[dict],
) -> tuple[list[str], dict[str, float]]:
    """
    Check each claim against the retrieved documents.

    Args:
        claims:        List of text claims to verify (remediation steps + RCA).
        retrieved_docs: Retrieved documents from the vector store.

    Returns:
        warnings:      List of ungrounded claim strings.
        scores:        Dict mapping claim (truncated) to its grounding score.
    """
    if not claims:
        return [], {}

    if not retrieved_docs:
        logger.warning("No retrieved docs available for grounding check")
        return [], {}

    # Pre-embed all document content chunks
    doc_vecs: list[list[float]] = []
    for doc in retrieved_docs:
        content = doc.get("content", "")
        if content:
            try:
                doc_vecs.append(_embed(content[:600]))
            except Exception as exc:
                logger.warning("Doc embedding failed", error=str(exc))

    if not doc_vecs:
        return [], {}

    warnings: list[str] = []
    scores: dict[str, float] = {}

    for claim in claims:
        if len(claim.strip()) < 15:
            continue  # Skip very short claims
        try:
            claim_vec = _embed(claim[:400])
            score = _max_similarity(claim_vec, doc_vecs)
            key = claim[:60]
            scores[key] = round(score, 3)

            if score < _GROUNDING_THRESHOLD:
                warnings.append(
                    f"Low grounding ({score:.2f}): '{claim[:80]}...'"
                    if len(claim) > 80
                    else f"Low grounding ({score:.2f}): '{claim}'"
                )
        except Exception as exc:
            logger.warning("Grounding check failed for claim", error=str(exc))

    logger.info(
        "Grounding check complete",
        total_claims=len(claims),
        warnings=len(warnings),
        avg_score=round(np.mean(list(scores.values())), 3) if scores else 0,
    )
    return warnings, scores
