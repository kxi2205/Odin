"""LLM interface for code generation and repair using Google Gemini.

Handles:
- Initial code generation (with optional past-run context from run_memory)
- Code repair given execution errors and optional RAG-retrieved docs
- User-feedback re-prompting (continuing an unresolved run)
- Token usage tracking from Gemini API response metadata
"""
from __future__ import annotations

import logging
from typing import Optional

import google.generativeai as genai

from odin.config import config
from odin.state import FileContent, PastRunContext, RetrievalRecord, TokenUsage

logger = logging.getLogger(__name__)

genai.configure(api_key=config.gemini_api_key)

_GEN_CFG = genai.types.GenerationConfig(temperature=0.2, max_output_tokens=4096)


# ---------------------------------------------------------------------------
# Public functions called by nodes
# ---------------------------------------------------------------------------

def generate_code(
    requirement: str,
    iteration: int,
    is_multi_file: bool = False,
    past_run_context: Optional[PastRunContext] = None,
) -> tuple[list[FileContent], TokenUsage]:
    """Generate initial code from a NL requirement.

    If a similar past run was found by run_memory, its final working code is
    injected into the prompt so the model can reference it directly rather
    than starting cold.

    Args:
        requirement:      The user's NL task description.
        iteration:        Current iteration number (for token tracking).
        is_multi_file:    Whether to generate multiple files.
        past_run_context: Optional similar past run found via run_memory.

    Returns:
        Tuple of (list[FileContent], TokenUsage).
    """
    logger.info("Generating code for: %.100s", requirement)

    if is_multi_file:
        prompt = _build_multi_file_generate_prompt(requirement, past_run_context)
    else:
        prompt = _build_single_file_generate_prompt(requirement, past_run_context)

    model = genai.GenerativeModel(config.gemini_model)
    response = model.generate_content(prompt, generation_config=_GEN_CFG)

    token_usage = TokenUsage(
        iteration=iteration,
        call_type="generate",
        input_tokens=response.usage_metadata.prompt_token_count,
        output_tokens=response.usage_metadata.candidates_token_count,
        total_tokens=response.usage_metadata.total_token_count,
    )
    files = _parse_code_response(response.text, is_multi_file)
    logger.info("Generated %d file(s), tokens: %d", len(files), token_usage.total_tokens)
    return files, token_usage


def repair_code(
    requirement: str,
    previous_files: list[FileContent],
    error_message: str,
    traceback: str,
    iteration: int,
    retrieval: Optional[RetrievalRecord] = None,
) -> tuple[list[FileContent], TokenUsage]:
    """Repair code given an execution error and optional retrieved context.

    For multi-file projects this implements context assembly:
    - Parse traceback to find which files are involved.
    - Always include the entry point.
    - If ImportError/ModuleNotFoundError include all files.
    """
    logger.info("Repairing code at iteration %d", iteration)

    relevant_files = _select_relevant_files_for_repair(
        previous_files, traceback, error_message
    )

    if len(previous_files) == 1:
        prompt = _build_single_file_repair_prompt(
            requirement, previous_files[0].content, error_message, traceback, retrieval
        )
    else:
        prompt = _build_multi_file_repair_prompt(
            requirement, relevant_files, error_message, traceback, retrieval
        )

    model = genai.GenerativeModel(config.gemini_model)
    response = model.generate_content(prompt, generation_config=_GEN_CFG)

    token_usage = TokenUsage(
        iteration=iteration,
        call_type="repair",
        input_tokens=response.usage_metadata.prompt_token_count,
        output_tokens=response.usage_metadata.candidates_token_count,
        total_tokens=response.usage_metadata.total_token_count,
    )
    is_multi = len(previous_files) > 1
    files = _parse_code_response(response.text, is_multi)
    logger.info("Repaired %d file(s), tokens: %d", len(files), token_usage.total_tokens)
    return files, token_usage


def apply_feedback(
    requirement: str,
    current_files: list[FileContent],
    user_feedback: str,
    iteration: int,
) -> tuple[list[FileContent], TokenUsage]:
    """Re-generate code incorporating explicit user feedback.

    This is used by feedback_node when the user says "the code works but is
    wrong / please do it differently" on an active, unresolved run.  It does
    NOT trigger a run_memory lookup — the caller (backend) guarantees that.

    Args:
        requirement:   Original NL requirement for the current run.
        current_files: The most-recently generated (possibly working) code.
        user_feedback: The user's free-text correction/dissatisfaction note.
        iteration:     Iteration counter (for token tracking).

    Returns:
        Tuple of (list[FileContent], TokenUsage).
    """
    logger.info("Applying user feedback at iteration %d", iteration)
    is_multi = len(current_files) > 1
    prompt = _build_feedback_prompt(requirement, current_files, user_feedback, is_multi)

    model = genai.GenerativeModel(config.gemini_model)
    response = model.generate_content(prompt, generation_config=_GEN_CFG)

    token_usage = TokenUsage(
        iteration=iteration,
        call_type="feedback",
        input_tokens=response.usage_metadata.prompt_token_count,
        output_tokens=response.usage_metadata.candidates_token_count,
        total_tokens=response.usage_metadata.total_token_count,
    )
    files = _parse_code_response(response.text, is_multi)
    logger.info("Feedback revision produced %d file(s), tokens: %d", len(files), token_usage.total_tokens)
    return files, token_usage


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _past_run_section(ctx: PastRunContext) -> str:
    """Format a past-run context block for injection into a generate prompt."""
    lines = [
        "\n\n--- Similar Past Run ---",
        f"Original requirement: {ctx.requirement}",
        f"Outcome: {ctx.outcome}",
        f"Similarity score: {ctx.similarity_score:.2f}",
    ]
    if ctx.error_categories_seen:
        lines.append(f"Errors encountered: {', '.join(ctx.error_categories_seen)}")
    if ctx.final_files:
        lines.append("\nFinal working code from that run (use as reference, not copy-paste):")
        for f in ctx.final_files:
            lines.append(f"\n# --- {f['path']} ---\n{f['content']}")
    lines.append("--- End Past Run ---\n")
    return "\n".join(lines)


def _build_single_file_generate_prompt(
    requirement: str,
    past_run_context: Optional[PastRunContext],
) -> str:
    memory_section = _past_run_section(past_run_context) if past_run_context else ""
    return (
        f"You are a Python code generator. Generate executable Python code for the following task.\n\n"
        f"Task: {requirement}"
        f"{memory_section}\n\n"
        "Requirements:\n"
        "- Write complete, runnable Python code\n"
        "- Do not use markdown code fences (```)\n"
        "- Include all necessary imports\n"
        "- Write clean, well-structured code\n\n"
        "Generate only the code, nothing else."
    )


def _build_multi_file_generate_prompt(
    requirement: str,
    past_run_context: Optional[PastRunContext],
) -> str:
    memory_section = _past_run_section(past_run_context) if past_run_context else ""
    return (
        f"You are a Python project generator. Generate a multi-file Python project.\n\n"
        f"Task: {requirement}"
        f"{memory_section}\n\n"
        "Requirements:\n"
        "- Format your response as:\n"
        "  FILE: path/to/file.py\n"
        "  <code content>\n"
        "  FILE: path/to/another.py\n"
        "  <code content>\n"
        "- Do not use markdown code fences\n"
        "- Include all necessary imports\n"
        "- Designate one file as the entry point (e.g., main.py)\n\n"
        "Generate the complete project."
    )


def _build_single_file_repair_prompt(
    requirement: str,
    failing_code: str,
    error_message: str,
    traceback: str,
    retrieval: Optional[RetrievalRecord],
) -> str:
    prompt = (
        f"You are debugging Python code. Fix the error.\n\n"
        f"Original Task: {requirement}\n\n"
        f"Failing Code:\n{failing_code}\n\n"
        f"Error Message:\n{error_message}\n\n"
        f"Full Traceback:\n{traceback}\n"
    )
    if retrieval and retrieval.chunks:
        prompt += "\n\nRelevant Documentation:\n"
        for i, chunk in enumerate(retrieval.chunks, 1):
            prompt += f"\n--- Reference {i} ---\n{chunk['content']}\n"
    prompt += (
        "\nRequirements:\n"
        "- Fix the error and return corrected code\n"
        "- Do not use markdown code fences (```)\n"
        "- Preserve the original intent\n\n"
        "Generate only the corrected code, nothing else."
    )
    return prompt


def _build_multi_file_repair_prompt(
    requirement: str,
    relevant_files: list[FileContent],
    error_message: str,
    traceback: str,
    retrieval: Optional[RetrievalRecord],
) -> str:
    prompt = f"You are debugging a Python project. Fix the error.\n\nOriginal Task: {requirement}\n\nProject Files:\n"
    for file in relevant_files:
        prompt += f"\nFILE: {file.path}\n{file.content}\n"
    prompt += f"\nError Message:\n{error_message}\n\nFull Traceback:\n{traceback}\n"
    if retrieval and retrieval.chunks:
        prompt += "\n\nRelevant Documentation:\n"
        for i, chunk in enumerate(retrieval.chunks, 1):
            prompt += f"\n--- Reference {i} ---\n{chunk['content']}\n"
    prompt += (
        "\nRequirements:\n"
        "- Fix the error and return all files that changed\n"
        "- Format as FILE: path\\n<code>\n"
        "- Do not use markdown code fences\n\n"
        "Generate the corrected files."
    )
    return prompt


def _build_feedback_prompt(
    requirement: str,
    current_files: list[FileContent],
    user_feedback: str,
    is_multi: bool,
) -> str:
    """Prompt that incorporates the user's explicit dissatisfaction/correction."""
    files_section = ""
    for f in current_files:
        files_section += f"\nFILE: {f.path}\n{f.content}\n"

    format_instruction = (
        "- Format multi-file output as FILE: path\\n<code> blocks\n"
        if is_multi
        else "- Return only the complete corrected code\n"
    )

    return (
        "You are revising Python code based on user feedback.\n\n"
        f"Original Task: {requirement}\n\n"
        f"Current Code:\n{files_section}\n"
        f"User Feedback:\n{user_feedback}\n\n"
        "Requirements:\n"
        "- Address the user's feedback completely\n"
        "- Do not use markdown code fences (```)\n"
        f"{format_instruction}\n"
        "Generate the revised code now."
    )


# ---------------------------------------------------------------------------
# Parsing & helpers (unchanged logic, kept here for locality)
# ---------------------------------------------------------------------------

def _parse_code_response(response_text: str, is_multi_file: bool) -> list[FileContent]:
    if not is_multi_file:
        code = response_text.strip()
        code = code.removeprefix("```python").removeprefix("```").removesuffix("```").strip()
        return [FileContent(path="main.py", content=code)]

    files: list[FileContent] = []
    current_path: Optional[str] = None
    current_content: list[str] = []

    for line in response_text.split("\n"):
        if line.startswith("FILE:"):
            if current_path:
                files.append(FileContent(path=current_path, content="\n".join(current_content).strip()))
            current_path = line[5:].strip()
            current_content = []
        elif current_path:
            current_content.append(line)

    if current_path:
        files.append(FileContent(path=current_path, content="\n".join(current_content).strip()))

    if not files:
        logger.warning("Failed to parse multi-file response, treating as single file")
        code = response_text.strip().removeprefix("```python").removeprefix("```").removesuffix("```").strip()
        return [FileContent(path="main.py", content=code)]

    return files


def _select_relevant_files_for_repair(
    all_files: list[FileContent],
    traceback: str,
    error_message: str,
) -> list[FileContent]:
    if len(all_files) == 1:
        return all_files
        
    is_import_error = "ImportError" in error_message or "ModuleNotFoundError" in error_message
    
    # 1. Entry point (highest priority)
    entry_point = all_files[0].path
    
    # 2. Files explicitly mentioned in the traceback
    mentioned = {f.path for f in all_files if f.path in traceback and f.path != entry_point}
    
    # Ordered candidate paths: entry point -> mentioned -> others (if ImportError)
    candidate_paths = [entry_point] + list(mentioned)
    
    if is_import_error:
        others = [f.path for f in all_files if f.path not in mentioned and f.path != entry_point]
        candidate_paths.extend(others)
        
    # Map paths back to FileContent
    path_to_file = {f.path: f for f in all_files}
    
    selected_files = []
    current_tokens = 0
    token_limit = config.repair_token_limit
    
    for path in candidate_paths:
        f = path_to_file.get(path)
        if not f:
            continue
            
        # Rough token estimate: 1 token ~= 4 chars
        tokens = max(1, len(f.content) // 4)
        
        # Always include the entry point even if it exceeds the limit
        if current_tokens + tokens > token_limit and len(selected_files) > 0:
            logger.warning(
                "Truncating repair context: token limit (%d) reached. Excluded %d file(s).",
                token_limit, len(candidate_paths) - len(selected_files)
            )
            break
            
        selected_files.append(f)
        current_tokens += tokens

    logger.info("Selected %d/%d files for repair context (~%d tokens)", len(selected_files), len(all_files), current_tokens)
    return selected_files
