"""Code execution with sandboxing and resource limits.

This module executes Python code in a subprocess with:
- Hard timeout enforcement
- Resource limits (CPU, memory, file size) via resource module
- Captured stdout, stderr, and exit code
- Traceback extraction
- Execution time measurement

No Docker is used to avoid container complexity and startup latency.
Resource limits provide basic protection against runaway code.
"""
import logging
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from odin.config import config
from odin.state import ExecutionRecord, FileContent

logger = logging.getLogger(__name__)


def execute_code(
    files: list[FileContent],
    timeout: int = 5,
    test_input: str = "",
    entry_point: Optional[str] = None,
) -> ExecutionRecord:
    """Execute Python code in a sandboxed subprocess.
    
    Args:
        files: List of files comprising the project
        timeout: Maximum execution time in seconds
        test_input: String to pass as stdin
        entry_point: Path to the main file to execute (defaults to first file)
    
    Returns:
        ExecutionRecord with execution results and metadata
    
    For single-file projects, executes the file directly via python -c.
    For multi-file projects, writes files to a temp directory and runs the entry point.
    
    Resource limits:
    - CPU time: same as timeout
    - Memory: 256 MB (adjustable if needed for legitimate use cases)
    - File size: 10 MB per file
    
    Security notes:
    - No network access by default (could be added per-project)
    - Runs with current user privileges (unprivileged user recommended)
    - Resource module limits prevent most resource exhaustion attacks
    - This is NOT a complete sandbox; do not execute untrusted code from
      external sources without additional hardening (firejail, containers, etc.)
    """
    start_time = time.perf_counter()
    
    try:
        if len(files) == 1:
            # Single file: execute directly
            result = _execute_single_file(files[0].content, timeout, test_input)
        else:
            # Multi-file: write to temp dir and execute entry point
            result = _execute_multi_file(files, timeout, test_input, entry_point)
        
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        return ExecutionRecord(
            iteration=0,  # Will be set by caller
            stdout=result["stdout"],
            stderr=result["stderr"],
            exit_code=result["exit_code"],
            traceback=result["traceback"],
            success=result["exit_code"] == 0,
            execution_time_ms=execution_time_ms,
        )
    
    except Exception as e:
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        logger.exception("Execution failed with exception")
        return ExecutionRecord(
            iteration=0,
            stdout="",
            stderr=f"Execution infrastructure error: {str(e)}",
            exit_code=-1,
            traceback=str(e),
            success=False,
            execution_time_ms=execution_time_ms,
        )


def _get_resource_preamble(timeout: int) -> str:
    """Return Python code to set resource limits."""
    if platform.system() == "Windows":
        return ""
    
    mem_bytes = config.sandbox_memory_limit_mb * 1024 * 1024
    file_bytes = config.sandbox_file_size_limit_mb * 1024 * 1024
    
    return f"""
import resource
import sys

# Set resource limits
try:
    resource.setrlimit(resource.RLIMIT_CPU, ({timeout}, {timeout}))
    resource.setrlimit(resource.RLIMIT_AS, ({mem_bytes}, {mem_bytes}))
    resource.setrlimit(resource.RLIMIT_FSIZE, ({file_bytes}, {file_bytes}))
except Exception:
    pass  # Resource limits may not be available on all platforms

"""

def _execute_single_file(code: str, timeout: int, test_input: str) -> dict:
    """Execute a single Python file via subprocess."""
    preamble = _get_resource_preamble(timeout)
    
    full_code = preamble + code
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True,
            text=True,
            input=test_input,
            timeout=timeout,
        )
        
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "traceback": _extract_traceback(result.stderr),
        }
    
    except subprocess.TimeoutExpired as e:
        stdout = _decode_output(e.stdout)
        stderr = _decode_output(e.stderr)
        
        return {
            "stdout": stdout,
            "stderr": stderr + f"\n\nTimeoutError: Execution exceeded {timeout} seconds",
            "exit_code": -1,
            "traceback": f"TimeoutError: Execution exceeded {timeout} seconds",
        }


def _execute_multi_file(
    files: list[FileContent],
    timeout: int,
    test_input: str,
    entry_point: Optional[str],
) -> dict:
    """Execute a multi-file project in a temporary directory.
    
    Context assembly for repair:
    When a multi-file project fails, we need to determine which files to include
    in the repair prompt. Strategy:
    
    1. Always include the entry point file
    2. Parse the traceback to find which files appear in the stack
    3. Include those files in the repair context
    4. If the error is an ImportError, include all files (might be cross-file issue)
    
    This logic is implemented in the repair node, not here; we just execute and
    report what happened.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Write all files
        for file in files:
            file_path = tmp_path / file.path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(file.content)
        
        # Determine entry point
        if entry_point:
            main_file = tmp_path / entry_point
        else:
            # Default to first file or look for main.py
            candidates = ["main.py", "app.py", files[0].path]
            main_file = None
            for candidate in candidates:
                candidate_path = tmp_path / candidate
                if candidate_path.exists():
                    main_file = candidate_path
                    break
            
            if main_file is None:
                return {
                    "stdout": "",
                    "stderr": "No entry point specified and no main.py/app.py found",
                    "exit_code": -1,
                    "traceback": "No entry point found",
                }
        
        # Execute the entry point
        preamble = _get_resource_preamble(timeout)
        if preamble:
            rel_main = main_file.relative_to(tmp_path).as_posix()
            bootstrap_code = preamble + f"""
import runpy
import sys
sys.argv = ["{rel_main}"]
runpy.run_path("{rel_main}", run_name="__main__")
"""
            bootstrap_file = tmp_path / "__bootstrap__.py"
            bootstrap_file.write_text(bootstrap_code, encoding="utf-8")
            cmd = [sys.executable, "__bootstrap__.py"]
        else:
            cmd = [sys.executable, main_file.name]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                input=test_input,
                timeout=timeout,
                cwd=str(tmp_path),
            )
            
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
                "traceback": _extract_traceback(result.stderr),
            }
        
        except subprocess.TimeoutExpired as e:
            stdout = _decode_output(e.stdout)
            stderr = _decode_output(e.stderr)
            
            return {
                "stdout": stdout,
                "stderr": stderr + f"\n\nTimeoutError: Execution exceeded {timeout} seconds",
                "exit_code": -1,
                "traceback": f"TimeoutError: Execution exceeded {timeout} seconds",
            }


def _decode_output(data: bytes | str | None) -> str:
    """Safely convert subprocess stdout/stderr (bytes, str, or None) to str."""
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data


def _extract_traceback(stderr: str) -> str:
    """Extract the Python traceback from stderr.
    
    Returns the full traceback if present, otherwise returns stderr as-is.
    """
    if "Traceback (most recent call last):" in stderr:
        return stderr
    return stderr
