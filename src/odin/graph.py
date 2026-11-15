"""LangGraph workflow definition for ODIN.

Two entry paths, one graph
--------------------------

Path A — new request  (is_continuation=False)
    START → generate → execute → …

Path B — user feedback on active run  (is_continuation=True)
    START → feedback → execute → …

The conditional at START chooses between paths based on state["is_continuation"].

Path A (normal loop):
    generate → execute
      ├─ success → END
      └─ failure → analyze_error → retrieve → repair → verify
                                                         ├─ should_end → END
                                                         └─ continue   → execute

Path B (feedback loop):
    feedback → execute
      ├─ success → END
      └─ failure → analyze_error → retrieve → repair → verify
                                                         ├─ should_end → END
                                                         └─ continue   → execute

After feedback_node resets is_continuation=False, any subsequent failure
falls through the normal analyze→retrieve→repair loop, NOT back to feedback.
"""
from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from odin.nodes import (
    analyze_error_node,
    execute_node,
    feedback_node,
    generate_node,
    repair_node,
    retrieve_node,
    verify_node,
)
from odin.state import GraphState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def _route_start(state: GraphState) -> str:
    """Choose between a fresh generation and a user-feedback revision."""
    if state.get("is_continuation"):
        logger.info("Routing START → feedback (continuation)")
        return "feedback"
    logger.info("Routing START → generate (new request)")
    return "generate"


def _route_after_execute(state: GraphState) -> str:
    if state["success"]:
        logger.info("Routing execute → END (success)")
        return "end"
    logger.info("Routing execute → analyze_error")
    return "analyze_error"


def _route_after_verify(state: GraphState) -> str:
    if state["should_end"]:
        logger.info("Routing verify → END")
        return "end"
    logger.info("Routing verify → execute (next iteration)")
    return "execute"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Build and compile the ODIN LangGraph workflow."""
    graph = StateGraph(GraphState)

    # ── Nodes ────────────────────────────────────────────────────────────
    graph.add_node("generate", generate_node)
    graph.add_node("feedback", feedback_node)
    graph.add_node("execute", execute_node)
    graph.add_node("analyze_error", analyze_error_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("repair", repair_node)
    graph.add_node("verify", verify_node)

    # ── Entry: conditional on is_continuation ────────────────────────────
    graph.set_conditional_entry_point(
        _route_start,
        {"generate": "generate", "feedback": "feedback"},
    )

    # ── Path A: generate → execute ───────────────────────────────────────
    graph.add_edge("generate", "execute")

    # ── Path B: feedback → execute ───────────────────────────────────────
    graph.add_edge("feedback", "execute")

    # ── Shared: execute → [end | analyze] ────────────────────────────────
    graph.add_conditional_edges(
        "execute",
        _route_after_execute,
        {"end": END, "analyze_error": "analyze_error"},
    )

    # ── Shared: debug loop ───────────────────────────────────────────────
    graph.add_edge("analyze_error", "retrieve")
    graph.add_edge("retrieve", "repair")
    graph.add_edge("repair", "verify")

    graph.add_conditional_edges(
        "verify",
        _route_after_verify,
        {"end": END, "execute": "execute"},
    )

    return graph.compile()
