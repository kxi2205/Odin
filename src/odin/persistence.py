"""Folder-based persistence for ODIN run history.

Every run is a directory under outputs/:

    outputs/
      run_<short_id>_<slug>/
        iteration_1/
          <generated files>
          metadata.json
        iteration_2/
          <generated files>
          metadata.json
        ...
        run_summary.json

Design decisions
----------------
- The folder tree IS the ground truth.  Nothing about a run's content exists
  only inside a vector store or only in memory.
- run_summary.json stores the top-level run record (requirement, outcome,
  timestamps, total iterations, path to the final working iteration folder).
- Each iteration_N/metadata.json stores exactly what that iteration produced:
  code, execution result, error classification, retrieval decision, token usage,
  and timestamps.  This matches the existing RunHistory sub-model schema so the
  data is identical to what was previously stuffed into a single JSON blob.
- The old flat  data/runs/<uuid>.json  format is gone.  RunHistoryPersistence
  now writes/reads the folder structure instead.
- The operator.add annotation on GraphState.current_files is removed in state.py
  (task 4) to stop LangGraph from accumulating stale file lists across iterations.
  Here we always read code_versions[-1].files for the current iteration's files.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from odin.config import config
from odin.state import CodeVersion, RunHistory

logger = logging.getLogger(__name__)

# Maximum characters kept from the requirement when building the folder slug.
_SLUG_MAX_CHARS = 40
_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")


def _make_run_dir_name(run_id: UUID, requirement: str) -> str:
    """Return a human-readable folder name for a run.

    Format:  run_<first-8-chars-of-uuid>_<slug>
    Example: run_3f7a1b2c_reverse_linked_list
    """
    slug = _SLUG_RE.sub("_", requirement[:_SLUG_MAX_CHARS]).strip("_").lower()
    return f"run_{str(run_id)[:8]}_{slug}"


def _run_dir(run_id: UUID, requirement: str) -> Path:
    return config.outputs_path / _make_run_dir_name(run_id, requirement)


def _iteration_dir(run_dir: Path, iteration: int) -> Path:
    return run_dir / f"iteration_{iteration}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RunHistoryPersistence:
    """Saves and loads run history using the folder-per-run structure."""

    def __init__(self, storage_path: Path | None = None) -> None:
        self.storage_path = storage_path or config.outputs_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

    # ── Writing ─────────────────────────────────────────────────────────────

    def save_iteration(self, run_history: RunHistory, iteration: int) -> Path:
        """Write a single iteration folder immediately after it completes.

        Called by nodes (or the backend) right after each generate/repair cycle
        so that files are on disk incrementally — not only at the end.

        Returns the path to the iteration_N/ folder.
        """
        run_dir = _run_dir(run_history.run_id, run_history.requirement)
        iter_dir = _iteration_dir(run_dir, iteration)
        iter_dir.mkdir(parents=True, exist_ok=True)

        # Write the generated source files
        code_version = next(
            (cv for cv in run_history.code_versions if cv.iteration == iteration),
            None,
        )
        if code_version:
            for file in code_version.files:
                dest = iter_dir / file.path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(file.content, encoding="utf-8")

        # Build metadata dict for this iteration
        metadata = _build_iteration_metadata(run_history, iteration)
        (iter_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )

        logger.info("Saved iteration %d to %s", iteration, iter_dir)
        return iter_dir

    def save_run(self, run_history: RunHistory) -> Path:
        """Write run_summary.json once the run is complete.

        Also calls save_iteration() for any iterations that haven't been flushed
        yet (handles the case where callers don't call save_iteration mid-run).

        Returns the run folder path.
        """
        run_dir = _run_dir(run_history.run_id, run_history.requirement)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Ensure every iteration folder exists
        for cv in run_history.code_versions:
            iter_dir = _iteration_dir(run_dir, cv.iteration)
            if not iter_dir.exists():
                self.save_iteration(run_history, cv.iteration)

        # Determine final working code location
        final_iter = run_history.code_versions[-1].iteration if run_history.code_versions else None
        final_iter_path = (
            str(_iteration_dir(run_dir, final_iter)) if final_iter else None
        )

        summary = {
            "run_id": str(run_history.run_id),
            "requirement": run_history.requirement,
            "final_status": run_history.final_status,
            "total_iterations": run_history.total_iterations,
            "start_time": run_history.start_time.isoformat() if run_history.start_time else None,
            "end_time": run_history.end_time.isoformat() if run_history.end_time else None,
            "final_working_code_path": final_iter_path,
            "run_folder": str(run_dir),
        }

        summary_path = run_dir / "run_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        logger.info("Saved run summary to %s", summary_path)
        return run_dir

    # ── Reading ─────────────────────────────────────────────────────────────

    def load_run(self, run_id: UUID | str) -> RunHistory:
        """Reconstruct a RunHistory from a run folder.

        Scans all iteration_N/ sub-directories and reassembles the history
        from their metadata.json files, then reads run_summary.json for the
        top-level fields.
        """
        run_id_str = str(run_id)[:8]
        # Find the matching run folder (prefix match on short id)
        matching = [
            p for p in self.storage_path.iterdir()
            if p.is_dir() and p.name.startswith(f"run_{run_id_str}")
        ]
        if not matching:
            raise FileNotFoundError(f"No run folder found for id prefix: {run_id_str}")
        run_dir = matching[0]

        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"run_summary.json missing in {run_dir}")

        with summary_path.open(encoding="utf-8") as fh:
            summary = json.load(fh)

        # Reassemble sub-records from iteration metadata files
        code_versions, executions, error_analyses, retrievals, token_usage = [], [], [], [], []
        stage_timestamps: dict[str, list[str]] = {}

        for iter_dir in sorted(run_dir.glob("iteration_*"), key=lambda p: _iter_num(p)):
            meta_path = iter_dir / "metadata.json"
            if not meta_path.exists():
                continue
            with meta_path.open(encoding="utf-8") as fh:
                meta = json.load(fh)

            if cv := meta.get("code_version"):
                code_versions.append(cv)
            if ex := meta.get("execution"):
                executions.append(ex)
            if ea := meta.get("error_analysis"):
                error_analyses.append(ea)
            if rt := meta.get("retrieval"):
                retrievals.append(rt)
            if tu := meta.get("token_usage"):
                token_usage.append(tu)
            for stage, ts_list in meta.get("stage_timestamps", {}).items():
                stage_timestamps.setdefault(stage, []).extend(ts_list)

        run_history = RunHistory(
            run_id=summary["run_id"],
            requirement=summary["requirement"],
            start_time=summary.get("start_time"),
            end_time=summary.get("end_time"),
            final_status=summary.get("final_status"),
            total_iterations=summary.get("total_iterations", 0),
            code_versions=code_versions,
            executions=executions,
            error_analyses=error_analyses,
            retrievals=retrievals,
            token_usage=token_usage,
            stage_timestamps=stage_timestamps,
        )
        return run_history

    def list_runs(self) -> list[dict[str, Any]]:
        """Return lightweight summaries of all runs, newest first."""
        runs = []
        for run_dir in self.storage_path.iterdir():
            if not run_dir.is_dir():
                continue
            summary_path = run_dir / "run_summary.json"
            if not summary_path.exists():
                continue
            try:
                with summary_path.open(encoding="utf-8") as fh:
                    s = json.load(fh)
                runs.append({
                    "run_id": s["run_id"],
                    "requirement": s["requirement"][:100],
                    "start_time": s.get("start_time"),
                    "final_status": s.get("final_status"),
                    "total_iterations": s.get("total_iterations", 0),
                    "run_folder": s.get("run_folder"),
                })
            except Exception as exc:
                logger.warning("Failed to read %s: %s", summary_path, exc)
        runs.sort(key=lambda r: r.get("start_time") or "", reverse=True)
        return runs

    def read_final_files(self, run_folder: str | Path) -> list[dict[str, str]]:
        """Read the actual source files from the last iteration of a past run.

        Returns a list of {"path": relative_path, "content": text} dicts.
        Used by the two-step run_memory lookup: semantic search gives us the
        folder path; this method reads the real files from disk.
        """
        run_dir = Path(run_folder)
        iter_dirs = sorted(run_dir.glob("iteration_*"), key=_iter_num)
        if not iter_dirs:
            return []
        last_iter = iter_dirs[-1]

        files = []
        for fpath in last_iter.rglob("*"):
            if fpath.is_file() and fpath.name != "metadata.json":
                try:
                    rel = fpath.relative_to(last_iter)
                    files.append({"path": str(rel), "content": fpath.read_text(encoding="utf-8")})
                except Exception as exc:
                    logger.warning("Could not read %s: %s", fpath, exc)
        return files

    # ── Metrics (unchanged from original) ───────────────────────────────────

    def compute_metrics(self, run_id: UUID | str) -> dict[str, Any]:
        run_history = self.load_run(run_id)
        total_tokens = sum(t["total_tokens"] for t in run_history.token_usage if isinstance(t, dict))
        total_retrieval_count = sum(
            1 for r in run_history.retrievals
            if (r.get("triggered") if isinstance(r, dict) else r.triggered)
        )
        execution_times = [
            (e.get("execution_time_ms") if isinstance(e, dict) else e.execution_time_ms)
            for e in run_history.executions
        ]
        avg_exec = sum(execution_times) / len(execution_times) if execution_times else 0
        if run_history.end_time and run_history.start_time:
            total_wall = (
                datetime.fromisoformat(str(run_history.end_time))
                - datetime.fromisoformat(str(run_history.start_time))
            ).total_seconds()
        else:
            total_wall = 0
        category_counts: dict[str, int] = {}
        for ea in run_history.error_analyses:
            cat = ea.get("category") if isinstance(ea, dict) else ea.category
            category_counts[cat] = category_counts.get(cat, 0) + 1
        return {
            "run_id": str(run_history.run_id),
            "requirement": run_history.requirement,
            "final_status": run_history.final_status,
            "total_iterations": run_history.total_iterations,
            "success": run_history.final_status == "success",
            "total_wall_time_seconds": total_wall,
            "total_llm_tokens": total_tokens,
            "total_retrieval_count": total_retrieval_count,
            "avg_execution_time_ms": avg_exec,
            "error_category_distribution": category_counts,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iter_num(p: Path) -> int:
    """Extract the integer N from an iteration_N folder name."""
    try:
        return int(p.name.split("_")[1])
    except (IndexError, ValueError):
        return 0


def _build_iteration_metadata(run_history: RunHistory, iteration: int) -> dict[str, Any]:
    """Compose the metadata.json content for a single iteration."""
    # code version
    cv = next((c for c in run_history.code_versions if c.iteration == iteration), None)
    # execution
    ex = next((e for e in run_history.executions if e.iteration == iteration), None)
    # error analysis
    ea = next((a for a in run_history.error_analyses if a.iteration == iteration), None)
    # retrieval
    rt = next((r for r in run_history.retrievals if r.iteration == iteration), None)
    # token usage
    tu = next((t for t in run_history.token_usage if t.iteration == iteration), None)

    # Stage timestamps relevant to this iteration
    relevant_stages: dict[str, list[str]] = {}
    for stage, ts_list in run_history.stage_timestamps.items():
        if ts_list:
            # each stage is visited once per iteration; pick the entry for this one
            all_iters = sorted(run_history.stage_timestamps.keys())
            idx = iteration - 1
            if idx < len(ts_list):
                relevant_stages[stage] = [str(ts_list[idx])]

    return {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "code_version": cv.model_dump(mode="json") if cv else None,
        "execution": ex.model_dump(mode="json") if ex else None,
        "error_analysis": ea.model_dump(mode="json") if ea else None,
        "retrieval": rt.model_dump(mode="json") if rt else None,
        "token_usage": tu.model_dump(mode="json") if tu else None,
        "stage_timestamps": relevant_stages,
    }
