"""
Drift Detector
--------------
Monitors two types of drift:
  1. Retrieval drift  — cosine distance between recent and baseline embeddings.
  2. Confidence drift — rolling mean of LLM confidence scores.

Called from the post-workflow hook and nightly Airflow DAG.
"""
from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path

import numpy as np
import structlog

from monitoring.metrics import CONFIDENCE_ROLLING_MEAN, RETRIEVAL_DRIFT_SCORE

logger = structlog.get_logger(__name__)

_CONFIDENCE_WINDOW = 100          # Rolling window size
_DRIFT_ALERT_THRESHOLD = 0.25     # Cosine distance above this = drift warning
_STATE_PATH = Path("./data/drift_state.json")

# In-process rolling buffer (replaced by Redis/DynamoDB in production)
_confidence_buffer: deque[float] = deque(maxlen=_CONFIDENCE_WINDOW)


# ── Cosine drift ──────────────────────────────────────────────────────────────

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute 1 - cosine_similarity between two mean embedding vectors."""
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))


def compute_retrieval_drift(
    recent_embeddings: list[list[float]],
    baseline_path: str | None = None,
) -> float:
    """
    Compare recent embeddings to a saved baseline.

    Args:
        recent_embeddings: List of embedding vectors from the current batch.
        baseline_path: Path to baseline .npy file. Defaults to data/baseline_embedding.npy.

    Returns:
        Cosine drift score in [0, 1].
    """
    baseline_path = baseline_path or "./data/baseline_embedding.npy"

    if not os.path.exists(baseline_path):
        logger.warning("No baseline found — saving current embeddings as baseline", path=baseline_path)
        baseline = np.mean(recent_embeddings, axis=0)
        os.makedirs(os.path.dirname(baseline_path) or ".", exist_ok=True)
        np.save(baseline_path, baseline)
        return 0.0

    baseline = np.load(baseline_path)
    recent_mean = np.mean(recent_embeddings, axis=0)
    drift = _cosine_distance(baseline, recent_mean)

    RETRIEVAL_DRIFT_SCORE.set(drift)

    if drift > _DRIFT_ALERT_THRESHOLD:
        logger.warning(
            "Retrieval drift detected — consider retraining embeddings",
            drift_score=round(drift, 4),
            threshold=_DRIFT_ALERT_THRESHOLD,
        )
    else:
        logger.info("Retrieval drift within bounds", drift_score=round(drift, 4))

    return drift


# ── Confidence drift ──────────────────────────────────────────────────────────

def record_confidence(score: float) -> float:
    """
    Append a confidence score to the rolling buffer and update Prometheus gauge.

    Returns:
        Current rolling mean.
    """
    _confidence_buffer.append(score)
    mean = float(np.mean(_confidence_buffer))
    CONFIDENCE_ROLLING_MEAN.set(mean)

    if len(_confidence_buffer) >= 10 and mean < 0.5:
        logger.warning(
            "Low confidence rolling mean — model may need retraining",
            rolling_mean=round(mean, 3),
            window=len(_confidence_buffer),
        )

    return mean


# ── State persistence ─────────────────────────────────────────────────────────

def save_drift_state() -> None:
    """Persist current drift state to disk (for DAG restarts)."""
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "confidence_buffer": list(_confidence_buffer),
        "confidence_mean": float(np.mean(_confidence_buffer)) if _confidence_buffer else None,
    }
    _STATE_PATH.write_text(json.dumps(state))


def load_drift_state() -> None:
    """Restore drift state from disk on startup."""
    if not _STATE_PATH.exists():
        return
    state = json.loads(_STATE_PATH.read_text())
    for v in state.get("confidence_buffer", []):
        _confidence_buffer.append(v)
    logger.info("Drift state restored", buffer_size=len(_confidence_buffer))
