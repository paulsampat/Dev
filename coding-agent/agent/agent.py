"""
Phase 6: CLI & UX polish — resume sessions, model/iteration overrides.

Added on top of Phase 5:
  - resume_session_id : load a saved session from disk and continue it,
                        preserving the full message history and cumulative cost.
  - model override    : pass a different Claude model string at call time.
  - max_iterations    : override the default iteration cap per run.
"""

import os
import time
import uuid

import anthropic
from dotenv import load_dotenv

load_dotenv()

from agent.logger import (
    log_error,
    log_iteration,
    log_session_end,
    log_session_start,
    log_tool_call,
    log_tool_result,
    log_usage,
    log_warning,
)
from agent.persistence import load_session, save_session
from tools.tool_executor import dispatch
from tools.tools import TOOLS

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL = "claude-opus-4-6"
MAX_TOKENS = 16_000
MAX_ITERATIONS = 20

# Pricing per million tokens (Opus 4.6)
PRICE_INPUT        = 5.00 / 1_000_000   # full-price input
PRICE_CACHE_WRITE  = 6.25 / 1_000_000   # cache write (1.25x)
PRICE_CACHE_READ   = 0.50 / 1_000_000   # cache read  (0.10x)
PRICE_OUTPUT       = 25.00 / 1_000_000

SYSTEM_PROMPT = """You are an expert coding agent. You have access to tools that let you
read files, write files, edit files, run shell commands, search for files, and grep
through code.

Guidelines:
- Before writing or editing code, read the relevant files first to understand context.
- Prefer str_replace_file for targeted edits over rewriting entire files.
- After making changes, run tests or check syntax to verify correctness.
- When exploring an unfamiliar codebase, use search_files and grep_files before reading
  individual files — it's faster than guessing paths.
- Be concise in your responses. Explain what you're doing and why, but don't repeat
  file contents back unless asked.
- If a task is ambiguous, make a reasonable attempt rather than asking clarifying
  questions — you can always adjust based on feedback.
- Whenever you generate a new project or set of code files, always create three
  standard files if they don't already exist:
    1. .env             — with placeholder entries for any required secrets/API keys
    2. requirements.txt — listing all third-party dependencies the code needs
    3. .gitignore       — including .env, __pycache__/, *.pyc, .venv/, and any other
                          generated or sensitive files relevant to the project
"""

# ── Cached prompt components ──────────────────────────────────────────────────
#
# Caching is a prefix match: tools → system → messages (render order).
# We cache the system prompt and the tool definitions together by placing
# cache_control on the last tool and on the system prompt block.
#
# Both are stable across every iteration — they never change during a run —
# so the cache is warm from iteration 2 onwards and across multiple runs
# within the 5-minute TTL window.

# Add cache_control to the last tool definition so the entire tool list
# (which renders before the system prompt) is cached as one prefix.
CACHED_TOOLS = [
    tool if i < len(TOOLS) - 1
    else {**tool, "cache_control": {"type": "ephemeral"}}
    for i, tool in enumerate(TOOLS)
]

# System prompt as a content block with cache_control.
CACHED_SYSTEM = [
    {
        "type": "text",
        "text": SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
]


# ── Logging helpers ───────────────────────────────────────────────────────────

def _print_tool_call(name: str, tool_input: dict) -> None:
    print(f"\n  ► tool call : {name}")
    for k, v in tool_input.items():
        display = repr(v) if not isinstance(v, str) else v
        if len(display) > 120:
            display = display[:120] + "…"
        print(f"    {k}: {display}")


def _print_tool_result(result: str) -> None:
    lines = result.splitlines()
    preview = lines[0] if lines else "(empty)"
    suffix = f"  … ({len(lines) - 1} more lines)" if len(lines) > 1 else ""
    print(f"  ◄ result    : {preview}{suffix}")


def _print_usage(usage, iteration_cost: float, total_cost: float) -> None:
    """Print token usage and cost breakdown for one iteration."""
    cache_read    = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write   = getattr(usage, "cache_creation_input_tokens", 0) or 0
    input_tokens  = usage.input_tokens or 0
    output_tokens = usage.output_tokens or 0

    cache_status = "HIT ✓" if cache_read > 0 else ("WRITE" if cache_write > 0 else "MISS")

    print(
        f"\n  [usage]  in: {input_tokens:,}  "
        f"cache_write: {cache_write:,}  "
        f"cache_read: {cache_read:,}  "
        f"out: {output_tokens:,}  "
        f"cache: {cache_status}  "
        f"cost: ${iteration_cost:.4f}  "
        f"total: ${total_cost:.4f}"
    )


# ── Core loop ─────────────────────────────────────────────────────────────────

def run(
    task: str,
    *,
    resume_session_id: str | None = None,
    model: str = MODEL,
    max_iterations: int = MAX_ITERATIONS,
) -> str:
    """
    Run the coding agent on a task.

    Args:
        task               : Natural language description of what to do.
        resume_session_id  : If set, load this session from disk and continue it.
        model              : Claude model string to use (default: MODEL constant).
        max_iterations     : Maximum agentic loop iterations (default: MAX_ITERATIONS).

    Returns:
        The agent's final text response.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # ── Resume or start fresh ─────────────────────────────────────────────────
    if resume_session_id:
        saved = load_session(resume_session_id)
        session_id = resume_session_id
        task       = saved["task"]
        messages   = saved["messages"]
        total_cost = saved.get("total_cost", 0.0)
        print(f"\n{'='*60}")
        print(f"Resuming session: {session_id}")
        print(f"Original task: {task}")
        print(f"Cost so far: ${total_cost:.4f}")
        print(f"{'='*60}")
    else:
        session_id = str(uuid.uuid4())
        messages   = [{"role": "user", "content": task}]
        total_cost = 0.0
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"Session: {session_id}")
        print(f"{'='*60}")

    log_session_start(session_id, task)

    try:
        for iteration in range(max_iterations):
            print(f"\n[iteration {iteration + 1}]")

            # ── API call with retry on rate limit / connection error ──────────
            max_retries = 4
            backoff = 5  # seconds

            for attempt in range(max_retries):
                try:
                    with client.messages.stream(
                        model=model,
                        max_tokens=MAX_TOKENS,
                        system=CACHED_SYSTEM,
                        tools=CACHED_TOOLS,
                        thinking={"type": "adaptive"},
                        messages=messages,
                    ) as stream:
                        for event in stream:
                            if (
                                event.type == "content_block_delta"
                                and event.delta.type == "text_delta"
                            ):
                                print(event.delta.text, end="", flush=True)

                        response = stream.get_final_message()
                    break  # success — exit retry loop

                except anthropic.RateLimitError as e:
                    log_error(session_id, e, context=f"iteration {iteration + 1} attempt {attempt + 1}")
                    if attempt == max_retries - 1:
                        raise
                    wait = backoff * (2 ** attempt)
                    print(f"\n  [rate limit] retrying in {wait}s…")
                    time.sleep(wait)

                except anthropic.APIConnectionError as e:
                    log_error(session_id, e, context=f"iteration {iteration + 1} attempt {attempt + 1}")
                    if attempt == max_retries - 1:
                        raise
                    wait = backoff * (2 ** attempt)
                    print(f"\n  [connection error] retrying in {wait}s…")
                    time.sleep(wait)

            print()

            # ── Calculate and display cost for this iteration ─────────────────
            u = response.usage
            cache_read    = getattr(u, "cache_read_input_tokens", 0) or 0
            cache_write   = getattr(u, "cache_creation_input_tokens", 0) or 0
            input_tokens  = u.input_tokens or 0
            output_tokens = u.output_tokens or 0

            iteration_cost = (
                input_tokens  * PRICE_INPUT
                + cache_write * PRICE_CACHE_WRITE
                + cache_read  * PRICE_CACHE_READ
                + output_tokens * PRICE_OUTPUT
            )
            total_cost += iteration_cost
            _print_usage(u, iteration_cost, total_cost)

            log_usage(
                session_id, iteration + 1,
                input_tokens, cache_write, cache_read, output_tokens,
                iteration_cost, total_cost,
            )

            # ── Append assistant response to history ──────────────────────────
            messages.append({"role": "assistant", "content": response.content})

            log_iteration(session_id, iteration + 1, response.stop_reason)

            # ── Check stop reason ─────────────────────────────────────────────
            if response.stop_reason == "end_turn":
                final_text = next(
                    (block.text for block in response.content if block.type == "text"),
                    "(no text response)",
                )
                print(f"\n{'='*60}")
                print(f"Agent finished.  Total cost: ${total_cost:.4f}")
                print(f"{'='*60}\n")

                save_session(session_id, task, messages, total_cost, iteration + 1, status="completed")
                log_session_end(session_id, total_cost, iteration + 1)
                return final_text

            if response.stop_reason == "pause_turn":
                print("  [pause_turn — resuming]")
                continue

            if response.stop_reason != "tool_use":
                msg = f"Unexpected stop reason: {response.stop_reason}"
                print(f"\n[warning] {msg}")
                log_warning(session_id, msg)
                break

            # ── Execute tool calls ────────────────────────────────────────────
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                _print_tool_call(block.name, block.input)
                log_tool_call(session_id, block.name, block.input)

                result = dispatch(block.name, block.input)

                _print_tool_result(result)
                log_tool_result(session_id, block.name, result)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

            messages.append({"role": "user", "content": tool_results})

        print(f"\n[warning] Reached iteration limit ({max_iterations}).")
        save_session(session_id, task, messages, total_cost, max_iterations, status="interrupted")
        log_warning(session_id, f"Reached iteration limit ({max_iterations})")
        return "Agent stopped: iteration limit reached."

    except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APIStatusError) as e:
        log_error(session_id, e, context="fatal API error")
        save_session(session_id, task, messages, total_cost, len(messages), status="error")
        raise

    except Exception as e:
        log_error(session_id, e, context="unexpected error")
        save_session(session_id, task, messages, total_cost, len(messages), status="error")
        raise


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python agent.py '<task description>'")
        print("Example: python agent.py 'add a hello_world() function to main.py'")
        sys.exit(1)

    task = " ".join(sys.argv[1:])
    result = run(task)
    print(result)
