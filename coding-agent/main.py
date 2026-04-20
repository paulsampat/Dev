"""
Phase 6: CLI entry point for the coding agent.

Usage:
    python main.py 'task description'
    python main.py 'task description' --model claude-sonnet-4-6
    python main.py 'task description' --max-iter 30
    python main.py --resume <session_id>
    python main.py --list
"""

import argparse
import sys

from agent.agent import MODEL, MAX_ITERATIONS
from agent.persistence import list_sessions


def cmd_list() -> None:
    """Print all saved sessions, most recent first."""
    sessions = list_sessions()
    if not sessions:
        print("No saved sessions found.")
        return

    print(f"\n{'─'*80}")
    print(f"  {'SESSION ID':<38}  {'STATUS':<12}  {'COST':>7}  TASK")
    print(f"{'─'*80}")
    for s in sessions:
        task_preview = (s["task"] or "")[:45]
        if len(s["task"] or "") > 45:
            task_preview += "…"
        print(
            f"  {s['session_id']:<38}  {s['status']:<12}  "
            f"${s['total_cost']:>6.4f}  {task_preview}"
        )
    print(f"{'─'*80}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Coding agent powered by Claude.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python main.py 'add a /health endpoint to app.py'
  python main.py 'refactor auth.py' --model claude-sonnet-4-6
  python main.py 'add tests' --max-iter 30
  python main.py --resume 3f2a1b4c-...
  python main.py --list
        """,
    )

    parser.add_argument(
        "task",
        nargs="?",
        help="Task for the agent to perform.",
    )
    parser.add_argument(
        "--resume",
        metavar="SESSION_ID",
        help="Resume a saved session by its ID.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all saved sessions.",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        metavar="MODEL",
        help=f"Claude model to use (default: {MODEL}).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=MAX_ITERATIONS,
        metavar="N",
        dest="max_iter",
        help=f"Maximum agentic loop iterations (default: {MAX_ITERATIONS}).",
    )
    parser.add_argument(
        "--skill",
        nargs="+",
        metavar="SKILL",
        dest="skills",
        help="One or more skills to load from the skills/ directory (e.g. --skill fastapi testing).",
    )

    args = parser.parse_args()

    # ── --list ────────────────────────────────────────────────────────────────
    if args.list:
        cmd_list()
        return

    # ── --resume ──────────────────────────────────────────────────────────────
    if args.resume:
        from agent.agent import run
        from agent.persistence import load_session
        try:
            load_session(args.resume)  # validate it exists before importing heavy deps
        except FileNotFoundError:
            print(f"Error: session '{args.resume}' not found. Run --list to see saved sessions.")
            sys.exit(1)
        result = run(
            task="",
            resume_session_id=args.resume,
            model=args.model,
            max_iterations=args.max_iter,
            skills=args.skills,
        )
        print(result)
        return

    # ── new task ──────────────────────────────────────────────────────────────
    if not args.task:
        parser.print_help()
        sys.exit(1)

    from agent.agent import run
    result = run(
        args.task,
        model=args.model,
        max_iterations=args.max_iter,
        skills=args.skills,
    )
    print(result)


if __name__ == "__main__":
    main()
