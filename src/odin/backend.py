"""Backend API for ODIN — UI-agnostic, fully decoupled from Textual.

Public surface
--------------
run_pipeline(requirement, ...)
    New request.  Queries run_memory for similar past runs, runs the graph,
    writes the iteration folder structure + run_summary.json, embeds the run
    into run_memory.

continue_run(run_history, user_feedback, ...)
    Continue an existing unresolved run with user feedback.  Does NOT query
    run_memory — the caller already has the active run in memory.

The two paths are kept as distinct methods so the TUI (and future harnesses)
can make the distinction explicit and unambiguous.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from odin.config import config
from odin.graph import build_graph
from odin.persistence import RunHistoryPersistence
from odin.rag.run_memory import RunMemory
from odin.state import GraphState, PastRunContext, RunHistory

logger = logging.getLogger(__name__)


class OdinBackend:
    """Backend interface for running ODIN pipelines."""

    def __init__(self) -> None:
        config.validate_config()
        config.ensure_directories()

        self._graph = build_graph()
        self._persistence = RunHistoryPersistence()

        # Shared instances — embedding model loaded once
        self._run_memory = RunMemory()

        logger.info("ODIN backend initialised")

    # ── New request ──────────────────────────────────────────────────────

    async def run_pipeline(
        self,
        requirement: str,
        test_input: str = "",
        max_iterations: Optional[int] = None,
        execution_timeout: Optional[int] = None,
    ) -> RunHistory:
        """Run a new ODIN pipeline for a fresh requirement.

        Steps:
        1. Query run_memory for a similar past run.
        2. Build initial graph state (inject past_run_context if found).
        3. Invoke the LangGraph graph (entry → generate → …).
        4. Write per-iteration folders + run_summary.json to outputs/.
        5. Embed this run into run_memory.
        6. Return the RunHistory object.

        Args:
            requirement:       NL task description.
            test_input:        Text to pipe to stdin during execution.
            max_iterations:    Override config.max_iterations.
            execution_timeout: Override config.execution_timeout.

        Returns:
            Completed RunHistory.
        """
        logger.info("Starting new pipeline: %.100s", requirement)

        # Step 1 — query run_memory (new requests only)
        past_ctx: PastRunContext | None = self._run_memory.query(requirement)

        run_history = RunHistory(requirement=requirement)

        initial_state: GraphState = {
            "requirement": requirement,
            "test_input": test_input,
            "is_continuation": False,
            "user_feedback": "",
            "past_run_context": past_ctx,
            "iteration": 0,
            "current_files": [],
            "last_execution": None,
            "last_error_analysis": None,
            "last_retrieval": None,
            "max_iterations": max_iterations or config.max_iterations,
            "execution_timeout": execution_timeout or config.execution_timeout,
            "run_history": run_history,
            "success": False,
            "should_end": False,
        }

        return await self._execute_graph_with_persistence(initial_state, run_history)

    # ── User-feedback continuation ───────────────────────────────────────

    async def continue_run(
        self,
        run_history: RunHistory,
        user_feedback: str,
        execution_timeout: Optional[int] = None,
    ) -> RunHistory:
        """Continue an active, unresolved run with explicit user feedback.

        This does NOT query run_memory.  The run_history object from the
        active session is passed directly so the graph picks up from where
        it left off.

        After the graph completes, iteration folders for the new attempts are
        appended to the existing run folder, and run_summary.json is updated.
        The run is then re-embedded into run_memory (upsert) to reflect the
        latest outcome.

        Args:
            run_history:       The existing RunHistory from the active run.
            user_feedback:     The user's correction / dissatisfaction note.
            execution_timeout: Override config.execution_timeout.

        Returns:
            Updated RunHistory.
        """
        logger.info(
            "Continuing run %s with user feedback: %.80s",
            run_history.run_id,
            user_feedback,
        )

        # Current files = last generated version in this run
        current_files = (
            run_history.code_versions[-1].files
            if run_history.code_versions
            else []
        )
        current_iteration = (
            run_history.code_versions[-1].iteration
            if run_history.code_versions
            else 0
        )

        state: GraphState = {
            "requirement": run_history.requirement,
            "test_input": "",
            # ↓ This is the key flag — feedback path, NOT run_memory path
            "is_continuation": True,
            "user_feedback": user_feedback,
            "past_run_context": None,   # explicitly None for continuations
            "iteration": current_iteration,
            "current_files": current_files,
            "last_execution": run_history.executions[-1] if run_history.executions else None,
            "last_error_analysis": run_history.error_analyses[-1] if run_history.error_analyses else None,
            "last_retrieval": run_history.retrievals[-1] if run_history.retrievals else None,
            "max_iterations": config.max_iterations,
            "execution_timeout": execution_timeout or config.execution_timeout,
            "run_history": run_history,
            "success": False,
            "should_end": False,
        }

        return await self._execute_graph_with_persistence(state, run_history)

    # ── Shared Persistence Logic ─────────────────────────────────────────

    async def _execute_graph_with_persistence(self, state: GraphState, run_history: RunHistory) -> RunHistory:
        """Execute the LangGraph workflow, finalize state, save to disk, and embed."""
        try:
            await self._graph.ainvoke(state)
        except Exception:
            logger.exception("Pipeline raised an exception")
            if not run_history.final_status:
                run_history.finalize("failure")
        finally:
            if not run_history.final_status:
                run_history.finalize("failure")

            # Write folder structure
            run_dir = self._persistence.save_run(run_history)

            # Embed into run_memory (success and failure both stored)
            try:
                self._run_memory.embed_run(run_history, run_dir)
            except Exception:
                logger.exception("Failed to embed run into run_memory — non-fatal")

        logger.info("Pipeline run complete: %s", run_history.final_status)
        return run_history

    # ── Accessors ────────────────────────────────────────────────────────

    def get_run_history(self, run_id: str) -> RunHistory:
        return self._persistence.load_run(run_id)

    def list_runs(self) -> list[dict]:
        return self._persistence.list_runs()

    def compute_metrics(self, run_id: str) -> dict:
        return self._persistence.compute_metrics(run_id)

    def read_final_files(self, run_folder: str) -> list[dict[str, str]]:
        """Read the actual source files from a past run's final iteration."""
        return self._persistence.read_final_files(run_folder)
