"""
MLflow Experiment Tracking
--------------------------
Utilities for logging agent runs, retrieval metrics, and model
performance to MLflow. Called from agents and the API layer.
"""
from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable

import mlflow
import structlog

from config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


def init_mlflow() -> None:
    """Configure MLflow tracking URI and experiment."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    logger.info("MLflow initialised", uri=settings.mlflow_tracking_uri)


@contextmanager
def incident_run(incident_id: str, tags: dict[str, str] | None = None):
    """
    Context manager that wraps an incident workflow in an MLflow run.

    Usage:
        with incident_run("INC-001", tags={"severity": "P1"}) as run:
            mlflow.log_metric("risk_score", 0.82)
    """
    init_mlflow()
    with mlflow.start_run(run_name=f"incident-{incident_id}") as run:
        mlflow.set_tags({"incident_id": incident_id, **(tags or {})})
        yield run


def log_retrieval_metrics(docs: list[dict], query: str) -> None:
    """Log RAG retrieval quality metrics."""
    if not docs:
        mlflow.log_metric("retrieval_num_docs", 0)
        return
    scores = [d.get("score", 0.0) for d in docs]
    mlflow.log_metrics(
        {
            "retrieval_num_docs": len(docs),
            "retrieval_top_score": max(scores),
            "retrieval_mean_score": sum(scores) / len(scores),
        }
    )
    mlflow.log_param("retrieval_query_length", len(query))


def log_agent_outcome(state: dict[str, Any]) -> None:
    """Log final workflow outcome metrics from IncidentState."""
    mlflow.log_metrics(
        {
            "risk_score": state.get("risk_score", 0.0),
            "confidence_score": state.get("confidence_score", 0.0),
            "remediation_steps_count": len(state.get("remediation_steps", [])),
            "requires_hitl": float(state.get("requires_human_review", False)),
        }
    )
    mlflow.log_params(
        {
            "severity": state.get("severity", "unknown"),
            "affected_service": state.get("affected_service", "unknown"),
            "vector_store_backend": settings.vector_store_backend,
            "llm_model": settings.bedrock_llm_model_id,
        }
    )


def track_latency(agent_name: str) -> Callable:
    """Decorator: log agent execution latency as an MLflow metric."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            duration = time.perf_counter() - start
            try:
                mlflow.log_metric(f"{agent_name}_latency_seconds", round(duration, 4))
            except Exception:
                pass  # Never let tracking break the workflow
            return result
        return wrapper
    return decorator
