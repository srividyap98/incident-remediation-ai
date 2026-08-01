"""
Multi-Agent Orchestrator v2
----------------------------
LangGraph StateGraph connecting all 5 nodes:

    [retriever] → [context_builder] → [risk_evaluator]
                                              │
                               ┌──────────────┴─────────────┐
                        score < threshold             score ≥ threshold
                               │                            │
                    [response_generator]         [human_review_checkpoint]
                               │                            │
                               └──────────────┬─────────────┘
                                         [evaluator]   ← NEW v2
                                              │
                                           [END]

New in v2:
  - evaluator node runs hallucination grounding check
  - OTel spans wrap each agent node
  - Arize logging after response generation
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, cast

import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from agents.context_builder.agent import context_builder_agent
from agents.response_generator.agent import response_generator_agent
from agents.retriever.agent import retriever_agent
from agents.risk_evaluator.agent import risk_evaluator_agent
from agents.state import IncidentState
from config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


# ── Evaluator node (v2) ───────────────────────────────────────────────────────

def evaluator_node(state: IncidentState) -> dict:
    """
    Post-generation quality gate.
    Runs hallucination grounding check and logs to Arize.
    """
    log = logger.bind(agent="evaluator", incident_id=state.get("incident_id"))

    # Skip evaluation if no response was generated (e.g. HITL rejected)
    if not state.get("final_response"):
        return {
            "current_agent": "evaluator",
            "agent_messages": ["[Evaluator] Skipped — no response to evaluate."],
        }

    claims = state.get("remediation_steps", [])
    rca    = state.get("root_cause_analysis", "")
    if rca:
        claims = [rca] + claims

    grounding_warnings: list[str] = []
    grounding_scores: dict = {}

    try:
        from evaluation.hallucination_guard import check_grounding
        grounding_warnings, grounding_scores = check_grounding(
            claims=claims,
            retrieved_docs=state.get("retrieved_docs", []),
        )
    except Exception as exc:
        log.warning("Hallucination guard failed", error=str(exc))

    # Log to Arize
    try:
        from monitoring.arize_logger import log_llm_interaction, log_grounding_scores
        log_llm_interaction(
            incident_id=state.get("incident_id", "unknown"),
            prompt=state.get("enriched_context", ""),
            response=state.get("final_response", ""),
            risk_score=state.get("risk_score"),
            confidence_score=state.get("confidence_score"),
            grounding_warnings=grounding_warnings,
        )
        if grounding_scores:
            log_grounding_scores(state.get("incident_id", "unknown"), grounding_scores)
    except Exception as exc:
        log.warning("Arize logging failed", error=str(exc))

    log.info(
        "Evaluation complete",
        grounding_warnings=len(grounding_warnings),
        grounding_checks=len(grounding_scores),
    )

    return {
        "grounding_warnings": grounding_warnings,
        "current_agent": "evaluator",
        "agent_messages": [
            f"[Evaluator] Grounding check: {len(grounding_scores)} claims checked, "
            f"{len(grounding_warnings)} warnings."
        ],
    }


# ── HITL checkpoint ───────────────────────────────────────────────────────────

def human_review_node(state: IncidentState) -> dict:
    logger.info(
        "HITL checkpoint",
        incident_id=state.get("incident_id"),
        risk_score=state.get("risk_score"),
    )
    approved = interrupt({
        "message": "High-risk incident requires human approval before generating response.",
        "incident_id": state.get("incident_id"),
        "risk_score": state.get("risk_score"),
        "risk_factors": state.get("risk_factors", []),
    })
    return {
        "hitl_approved": bool(approved),
        "agent_messages": [f"[HITL] Human review {'approved' if approved else 'rejected'}."],
    }


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_risk(state: IncidentState) -> str:
    if state.get("error"):
        return END
    return "human_review" if state.get("requires_human_review") else "response_generator"


def route_after_hitl(state: IncidentState) -> str:
    return "response_generator" if state.get("hitl_approved") else END


# ── Graph factory ─────────────────────────────────────────────────────────────

def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    if checkpointer is None:
        checkpointer = MemorySaver()

    builder = StateGraph(IncidentState)

    builder.add_node("retriever",           retriever_agent)
    builder.add_node("context_builder",     context_builder_agent)
    builder.add_node("risk_evaluator",      risk_evaluator_agent)
    builder.add_node("human_review",        human_review_node)
    builder.add_node("response_generator",  response_generator_agent)
    builder.add_node("evaluator",           evaluator_node)       # v2

    builder.add_edge(START,              "retriever")
    builder.add_edge("retriever",        "context_builder")
    builder.add_edge("context_builder",  "risk_evaluator")

    builder.add_conditional_edges(
        "risk_evaluator", route_after_risk,
        {"human_review": "human_review", "response_generator": "response_generator", END: END},
    )
    builder.add_conditional_edges(
        "human_review", route_after_hitl,
        {"response_generator": "response_generator", END: END},
    )

    builder.add_edge("response_generator", "evaluator")   # v2: always evaluate
    builder.add_edge("evaluator",          END)

    graph = builder.compile(checkpointer=checkpointer)
    logger.info("LangGraph v2 compiled", nodes=list(builder.nodes))
    return graph


@lru_cache(maxsize=1)
def get_graph() -> CompiledStateGraph:
    """
    Process-wide singleton graph with a persistent in-memory checkpointer.

    build_graph() alone creates a fresh MemorySaver per call — fine for tests
    that stream a workflow to completion in one shot, but HITL resume and
    status lookups need to see checkpoints written by an earlier request, so
    callers that span multiple requests (the API routers) must share one
    graph instance instead of calling build_graph() each time.
    """
    return build_graph()


def run_incident_workflow(incident_input: dict[str, Any], thread_id: str | None = None) -> IncidentState:
    graph: CompiledStateGraph = build_graph()
    config: RunnableConfig = {"configurable": {"thread_id": thread_id or incident_input.get("incident_id", "default")}}
    final: dict[str, Any] | None = None
    for event in graph.stream(incident_input, config=config, stream_mode="values"):
        final = event
        logger.info("agent_step", agent=event.get("current_agent"))
    if final is None:
        raise RuntimeError("Workflow produced no output.")
    return cast(IncidentState, final)
