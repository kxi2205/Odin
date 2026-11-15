# ODIN: Orchestrated Debugging and Intelligent Navigation

**An agentic self-debugging framework for automated code repair using Adaptive Retrieval-Augmented Generation (RAG)**

ODIN iteratively generates, executes, analyzes, and repairs Python code, using external knowledge retrieval only when the error type suggests it will actually help.

## Core Features

- **Six-category error classification**: syntax, runtime, logical, library_api, version_specific, uncommon_pattern
- **Adaptive RAG**: Deterministic routing decides when to retrieve based on error category (not LLM-based)
- **Multi-file project support**: Handles both single-file DSA problems and multi-file projects
- **Complete run history**: Every code version, execution, error analysis, retrieval decision, and token usage logged to disk
- **Sandboxed execution**: Resource-limited subprocess execution (timeout, CPU, memory limits where supported)
- **Textual TUI**: Clean terminal interface with live code display, execution output, and per-iteration transparency
- **Production-ready**: Type hints throughout, structured logging, proper exception handling, no hardcoded secrets

## Architecture

```
requirement → generate → execute → [success?] → END
                           ↓
                         [fail]
                           ↓
                   analyze_error → classify into 6 categories
                           ↓
                      retrieve? (adaptive routing)
                           ↓
                   [if yes: fetch from ChromaDB]
                           ↓
                   repair → generate corrected code
                           ↓
                   execute → [loop up to max iterations]
```

**The key innovation**: The `retrieve` node uses a deterministic lookup table to decide whether to trigger RAG:
- `syntax`, `runtime`, `logical` → skip retrieval (resolvable from traceback + reasoning)
- `library_api`, `version_specific`, `uncommon_pattern` → trigger retrieval (external knowledge helps)

This routing table is easily overridable for experiments.

## Installation

### Requirements

- Python 3.12+
- Google Gemini API key (free tier works)

### Setup

1. **Clone/navigate to the project**:
   ```powershell
   cd c:\Users\sneha\Desktop\Odin
   ```

2. **Create and activate virtual environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

4. **Configure environment**:
   ```powershell
   Copy-Item .env.example .env
   ```
   
   Edit `.env` and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_actual_key_here
   ODIN_MAX_ITERATIONS=5
   ODIN_EXECUTION_TIMEOUT=5
   ODIN_RAG_DB_PATH=./data/chromadb
   ODIN_RUN_HISTORY_PATH=./data/runs
   ```

5. **Ingest knowledge base** (required before first run):
   ```powershell
   python -m odin.rag.ingest
   ```
   
   This creates the ChromaDB collection with built-in Python stdlib documentation. The collection starts with ~8 curated snippets covering common library API and version-specific issues.
   
   **To extend the knowledge base**:
   - Create a `knowledge/` directory at the project root
   - Add markdown files with documentation (will be chunked automatically)
   - Create `knowledge/examples.md` with error-fix examples in this format:
     ```markdown
     # Example: Pandas DataFrame append deprecated
     Category: version_specific
     Error:
     ```
     AttributeError: 'DataFrame' object has no attribute 'append'
     ```
     Fix:
     ```
     Use pd.concat([df1, df2]) instead of df.append()
     ```
     ```
   - Run `python -m odin.rag.ingest --source-dir ./knowledge --examples-file ./knowledge/examples.md`

## Usage

### Run the TUI

```powershell
python -m odin.main
```

Or if installed as a package:
```powershell
odin
```

The TUI provides:
- Input field for natural language requirements
- Optional test input (for programs that read from stdin)
- Live code display with syntax highlighting
- Per-iteration transparency: error category, retrieval decision (triggered/skipped), reasoning
- Execution output (stdout/stderr)
- Run history panel
- Final status (success/failure/max_iterations_reached)

**Keyboard shortcuts**:
- `r` or click "Run" button: Execute pipeline
- `c` or click "Clear" button: Clear display
- `q`: Quit

### Use the Backend API Directly

For experiment harnesses or programmatic use:

```python
import asyncio
from odin.backend import OdinBackend

async def run_experiment():
    backend = OdinBackend()
    
    run_history = await backend.run_pipeline(
        requirement="Write a function to reverse a linked list",
        test_input="",
        max_iterations=5,
    )
    
    print(f"Status: {run_history.final_status}")
    print(f"Iterations: {run_history.total_iterations}")
    
    # Compute metrics
    metrics = backend.compute_metrics(str(run_history.run_id))
    print(f"Total tokens: {metrics['total_llm_tokens']}")
    print(f"Retrieval count: {metrics['total_retrieval_count']}")

asyncio.run(run_experiment())
```

### Run Unit Tests

```powershell
pytest tests/ -v
```

Tests cover:
- Error classification logic
- Routing decisions
- Code execution with timeouts
- Run history persistence and metrics computation

## Project Structure

```
src/odin/
├── __init__.py           # Package exports
├── main.py               # TUI entry point
├── backend.py            # UI-agnostic async API
├── config.py             # Configuration management
├── state.py              # Pydantic state models
├── graph.py              # LangGraph workflow definition
├── nodes.py              # Graph nodes (generate, execute, analyze, retrieve, repair, verify)
├── error_classifier.py   # 6-category error classification
├── routing.py            # Deterministic retrieval routing
├── executor.py           # Sandboxed code execution
├── llm.py                # Gemini API interface with token tracking
├── persistence.py        # Run history storage (JSON files)
├── tui.py                # Textual TUI
└── rag/
    ├── __init__.py
    ├── retriever.py      # ChromaDB query interface
    └── ingest.py         # Standalone ingestion script

tests/
├── test_error_classifier.py
├── test_executor.py
├── test_routing.py
└── test_persistence.py

data/
├── chromadb/             # ChromaDB persistent storage
└── runs/                 # Run history JSON files (one per run)
```

## Configuration

Environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | (required) | Google Gemini API key |
| `ODIN_MAX_ITERATIONS` | `5` | Maximum repair iterations per run |
| `ODIN_EXECUTION_TIMEOUT` | `5` | Execution timeout in seconds |
| `ODIN_RAG_DB_PATH` | `./data/chromadb` | ChromaDB storage path |
| `ODIN_RUN_HISTORY_PATH` | `./data/runs` | Run history JSON files path |

## Run History and Logging

Every run is logged to `./data/runs/<run_id>.json` with:

- Original requirement and unique run ID
- Every code version (all files, tagged with iteration)
- Every execution: stdout, stderr, exit code, traceback, wall-clock time
- Every error classification: category and reasoning
- Every retrieval decision: triggered or skipped, reason, query, chunks retrieved, latency, token count
- Token usage per LLM call (from Gemini API response metadata)
- Timestamp for every node/stage
- Final outcome: success, failure, or max_iterations_reached

This format is designed to directly support batch experiments. A future experiment harness can:
1. Read these JSON files back
2. Compute metrics: repair success rate, avg iterations, total tokens, retrieval frequency
3. Compare configurations (different routing tables, different retrieval strategies)
4. All without re-running anything or relying on external tracing services

## Design Decisions and Tradeoffs

### Error Classification

**Heuristic-based, not LLM-based**. We parse exception types and error messages using regex patterns and keyword matching. Why?
- **Determinism**: Same error → same classification every time
- **Speed**: No extra LLM call per error
- **Debuggability**: Logic is explicit and inspectable

Where signals are ambiguous (e.g., `AttributeError` can be runtime, library_api, or uncommon_pattern), we use documented heuristics:
- Check if a known library appears in the traceback → library_api
- Check for deprecation keywords → version_specific
- Check for uncommon modules (itertools, asyncio) → uncommon_pattern
- Default to runtime if no special context

This is necessarily imperfect, but the classification logic is in one place (`error_classifier.py`) and easy to refine.

### Retrieval Routing

**Deterministic lookup table, not LLM-decided**. Given an error category, a simple dict maps to `True` (retrieve) or `False` (skip). Why?
- **Transparency**: The decision is explicit and logged
- **Tunability**: Swap the routing table for experiments without touching the graph
- **No LLM tax**: Retrieval decisions don't burn tokens

The routing table is defined in `routing.py` and can be overridden via `override_routing()` for different experimental configurations.

### Sandboxing Strategy

**Subprocess + resource limits, no Docker**. Why?
- **Simplicity**: No container orchestration, no image builds
- **Startup latency**: subprocess is instant; Docker adds 100ms+ per run
- **Portability**: Works on any system with Python

**Security caveat**: This is NOT a complete sandbox. It prevents resource exhaustion (CPU, memory, file size) via `resource.setrlimit` on Unix-like systems (Windows does not support this, so limits are skipped there). It does NOT prevent:
- Malicious file writes outside temp directories
- Network access (though none is granted by default)
- Privilege escalation

**Recommendation**: If running untrusted code, add `firejail` or similar, or use Docker. For LLM-generated code (the intended use case), subprocess + timeouts are a pragmatic tradeoff.

### Multi-file Context Assembly

When a multi-file project fails, we need to decide which files to include in the repair prompt (to reduce token usage). Strategy:
1. If `ImportError` or `ModuleNotFoundError`: include all files (cross-file issue)
2. Otherwise: parse traceback to find which files appear in the stack
3. Always include the entry point

This is implemented in `llm.py:_select_relevant_files_for_repair()`.

### Persistence: JSON Files vs. SQLite

**Choice**: JSON files (one per run).

**Why?**
- **Human-readable**: Easy to inspect a run without tooling
- **Schema-free**: No migrations needed as we iterate
- **Simple**: Write = json.dump, read = json.load
- **Sufficient**: Experiment harnesses read O(100s) of files, which is fast enough

**Tradeoff**: No relational queries. If we later need complex aggregations (e.g., "show all runs where library_api retrieval triggered on pandas errors"), we'd migrate to SQLite. But for now, JSON is simpler.

### Embedding Model Choice

**Model**: `all-MiniLM-L6-v2` (Sentence Transformers)

**Why?**
- **Fast**: 384-dimensional embeddings, encodes in milliseconds
- **Small**: ~90MB model size
- **Effective**: Performs well on semantic similarity tasks (SBERT benchmark)
- **Free**: No API calls, runs locally

**Tradeoff**: Not as powerful as large models (e.g., OpenAI embeddings), but sufficient for matching error messages to documentation snippets.

### Chunking Strategy

**Approach**:
- Documentation: 500-token chunks with 50-token overlap
- API references: One chunk per function/class/method (not implemented in ingestion script, would require parsing)
- Error examples: One chunk per example (error + fix pair)

**Why 500/50?**
- Balances context size vs. retrieval precision
- 50-token overlap ensures we don't split mid-sentence/mid-concept

**Metadata attached**:
- `source`: File path or URL
- `doc_type`: "documentation", "api_reference", or "example"
- `library`: Relevant library name (pandas, numpy, etc.)
- `category`: Which error categories this knowledge helps (optional)

This metadata could be used for filtering (e.g., "only retrieve pandas docs for library_api errors in pandas code"), but the current implementation retrieves by semantic similarity alone.

## Extending ODIN

### Add More Knowledge

1. Create a `knowledge/` directory
2. Add markdown files with documentation:
   ```
   knowledge/
   ├── pandas_api.md
   ├── numpy_common_errors.md
   └── asyncio_patterns.md
   ```
3. Optionally create `knowledge/examples.md` with error-fix examples
4. Run `python -m odin.rag.ingest --source-dir ./knowledge --examples-file ./knowledge/examples.md`

### Override Retrieval Routing for Experiments

```python
from odin.routing import override_routing, reset_routing

# Force retrieval on every error
override_routing({
    "syntax": True,
    "runtime": True,
    "logical": True,
    "library_api": True,
    "version_specific": True,
    "uncommon_pattern": True,
})

# Run experiments...

# Restore defaults
reset_routing()
```

### Build an Experiment Harness

The backend API and run history format are designed for batch experiments:

```python
import asyncio
from odin.backend import OdinBackend
from odin.persistence import RunHistoryPersistence

async def run_batch_experiment(problems, config_name):
    backend = OdinBackend()
    run_ids = []
    
    for problem in problems:
        run_history = await backend.run_pipeline(
            requirement=problem["description"],
            test_input=problem.get("input", ""),
        )
        run_ids.append(str(run_history.run_id))
    
    # Compute aggregate metrics
    persistence = RunHistoryPersistence()
    metrics = persistence.compute_batch_metrics(run_ids)
    
    print(f"Config: {config_name}")
    print(f"Success rate: {metrics['success_rate']:.2%}")
    print(f"Avg iterations: {metrics['avg_iterations']:.2f}")
    print(f"Retrieval frequency: {metrics['retrieval_frequency']:.2%}")
    print(f"Total tokens: {metrics['total_tokens']}")
```

## Logging

All logs are written to `odin.log` in the project root. The TUI runs with file-only logging to avoid cluttering the terminal.

Log level: `INFO` by default (change in `main.py` or `tui.py`).

## Known Limitations

1. **Sandboxing is basic**: Resource limits on Unix, none on Windows. Not suitable for untrusted code without additional hardening.

2. **Error classification heuristics are imperfect**: Some ambiguous errors may be misclassified. The logic is explicit and documented, so you can refine it.

3. **Multi-file repair context assembly is heuristic**: We parse tracebacks to find relevant files, but complex cross-file issues might need more files than we include.

4. **No test case validation**: For "logical" errors, we rely on AssertionError or explicit test failure signals. We don't have a built-in test runner; you'd need to write code with asserts or pass test cases in the requirement.

5. **Gemini API rate limits**: Free tier has rate limits. For batch experiments, add retry logic or use a paid tier.

## Acknowledgments

- **Chunking strategy**: Inspired by LangChain's RecursiveCharacterTextSplitter
- **Embedding model**: `all-MiniLM-L6-v2` by Sentence Transformers (MIT license)
- **Error categories**: Informed by program repair literature (Tufano et al., "An Empirical Study on Learning Bug-Fixing Patches")

## License

MIT (see project license if applicable)

## Future Work (Not Implemented)

These are ideas for future phases, deliberately excluded from this build:

- **Test case generation**: Automatically generate test cases from requirements
- **Multi-language support**: Currently Python-only
- **Streaming TUI updates**: The TUI currently shows final results; could stream node-by-node
- **Advanced sandboxing**: firejail, gvisor, or full containers
- **Fine-tuned error classifier**: Replace heuristics with a small fine-tuned model
- **RAG ranking**: Re-rank retrieved chunks with a cross-encoder
- **Cost tracking**: Log Gemini API costs (not just tokens)
- **Web UI**: Supplement or replace the TUI with a web interface

---

**Questions or issues?** The codebase is fully documented with docstrings and inline comments. Start with `backend.py` for the API, `graph.py` for the workflow, and `error_classifier.py` for classification logic.
