"""
Best Practices Agent.

Scans source files in a directory and reports deviations from the
architectural rules defined in your rules/ directory.

Runs as a standalone script — separate from the coding agent loop.
Use it after a task completes, before a code review, or in CI.

Usage:
    python bestpractices.py                        # check git-changed files
    python bestpractices.py --all                  # check all source files
    python bestpractices.py --path src/            # check a specific directory
    python bestpractices.py --rules custom_rules/  # use a different rules dir
    python bestpractices.py --all --output file    # save report to reports/
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 8_096

DEFAULT_RULES_DIR = Path(__file__).parent / "rules"
DEFAULT_PATH      = Path.cwd()
REPORTS_DIR       = Path(__file__).parent / "reports"

SOURCE_EXTENSIONS = {".py", ".ts", ".js", ".tsx", ".jsx"}

# Rough token budget — keep file content under this to stay well within context
MAX_CONTENT_CHARS = 150_000


# ── File collection ───────────────────────────────────────────────────────────

def _get_changed_files(base_path: Path) -> list[Path]:
    """Return source files modified since the last git commit."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True,
            text=True,
            cwd=base_path,
        )
        staged = subprocess.run(
            ["git", "diff", "--name-only", "--cached"],
            capture_output=True,
            text=True,
            cwd=base_path,
        )
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
            cwd=base_path,
        )
    except FileNotFoundError:
        print("Warning: git not found. Falling back to checking all files.")
        return _get_all_files(base_path)

    names = set(
        result.stdout.splitlines()
        + staged.stdout.splitlines()
        + untracked.stdout.splitlines()
    )

    files = []
    for name in sorted(names):
        path = (base_path / name).resolve()
        if path.exists() and path.suffix in SOURCE_EXTENSIONS:
            files.append(path)
    return files


def _get_all_files(base_path: Path) -> list[Path]:
    """Return all source files under base_path."""
    return sorted(
        p for p in base_path.rglob("*")
        if p.is_file()
        and p.suffix in SOURCE_EXTENSIONS
        and ".venv" not in p.parts
        and "__pycache__" not in p.parts
        and ".git" not in p.parts
    )


# ── Rules loading ─────────────────────────────────────────────────────────────

def _load_rules(rules_dir: Path) -> dict[str, str]:
    """Load all .md files from the rules directory."""
    if not rules_dir.exists():
        print(f"Error: rules directory not found: {rules_dir}")
        sys.exit(1)

    rules = {
        path.stem: path.read_text(encoding="utf-8").strip()
        for path in sorted(rules_dir.glob("*.md"))
        if path.is_file()
    }

    if not rules:
        print(f"Error: no .md rule files found in {rules_dir}")
        sys.exit(1)

    return rules


# ── Claude call ───────────────────────────────────────────────────────────────

def _build_prompt(rules: dict[str, str], files: dict[Path, str]) -> str:
    rules_section = "\n\n---\n\n".join(
        f"### {name}.md\n\n{content}"
        for name, content in rules.items()
    )

    files_section = "\n\n---\n\n".join(
        f"### {path.name}\n\n```{path.suffix.lstrip('.')}\n{content}\n```"
        for path, content in files.items()
    )

    return (
        f"You are a senior engineer reviewing code for compliance with architectural rules.\n\n"
        f"## Architectural Rules\n\n{rules_section}\n\n"
        f"## Files to Review\n\n{files_section}\n\n"
        f"## Instructions\n\n"
        f"Review every file against every rule. For each violation found:\n"
        f"- State the file name\n"
        f"- State which rule is violated\n"
        f"- Quote the specific line(s) of code that violate the rule\n"
        f"- Suggest the exact fix\n\n"
        f"If a file is fully compliant, do not mention it.\n"
        f"If all files are compliant, say: ALL FILES COMPLIANT\n\n"
        f"Group violations by file. Be specific and actionable."
    )


def _run_analysis(rules: dict[str, str], files: dict[Path, str]) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    prompt = _build_prompt(rules, files)

    print(f"  Sending {len(files)} file(s) and {len(rules)} rule(s) to {MODEL}...")

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text.strip()


# ── Output ────────────────────────────────────────────────────────────────────

def _print_report(report: str, files: dict, rules: dict, mode: str) -> None:
    header = (
        f"\n{'='*60}\n"
        f"Best Practices Report\n"
        f"Files reviewed : {len(files)}\n"
        f"Rules applied  : {', '.join(rules.keys())}\n"
        f"{'='*60}\n"
    )

    full_report = header + "\n" + report + "\n"

    print(full_report)

    if mode == "file":
        REPORTS_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = REPORTS_DIR / f"report_{timestamp}.txt"
        report_path.write_text(full_report, encoding="utf-8")
        print(f"Report saved to {report_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bestpractices",
        description="Check source files against your architectural rules.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python bestpractices.py                    # check git-changed files
  python bestpractices.py --all              # check all source files
  python bestpractices.py --path src/        # check a specific directory
  python bestpractices.py --all --output file  # save report to reports/
        """,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all source files (default: only git-changed files).",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_PATH,
        metavar="DIR",
        help="Directory to scan (default: current directory).",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=DEFAULT_RULES_DIR,
        metavar="DIR",
        help="Directory containing rule .md files (default: rules/).",
    )
    parser.add_argument(
        "--output",
        choices=["terminal", "file"],
        default="terminal",
        help="Output mode: print to terminal or save to reports/ (default: terminal).",
    )

    args = parser.parse_args()

    # ── Load rules ────────────────────────────────────────────────────────────
    rules = _load_rules(args.rules)
    print(f"\nRules loaded: {', '.join(rules.keys())}")

    # ── Collect files ─────────────────────────────────────────────────────────
    if args.all:
        raw_files = _get_all_files(args.path)
        mode_label = "all source files"
    else:
        raw_files = _get_changed_files(args.path)
        mode_label = "git-changed files"

    if not raw_files:
        print(f"No {mode_label} found to review.")
        sys.exit(0)

    # ── Read file contents, warn if too large ─────────────────────────────────
    files: dict[Path, str] = {}
    total_chars = 0

    for path in raw_files:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"  Warning: could not read {path}: {e}")
            continue

        total_chars += len(content)
        if total_chars > MAX_CONTENT_CHARS:
            print(
                f"\nWarning: content exceeds {MAX_CONTENT_CHARS:,} chars. "
                f"Stopping at {len(files)} file(s) to stay within context limits. "
                f"Use --path to narrow the scope or run --changed instead of --all."
            )
            break

        files[path] = content

    print(f"Files to review ({mode_label}): {len(files)}")
    for path in files:
        print(f"  {path.relative_to(args.path)}")

    # ── Run analysis ──────────────────────────────────────────────────────────
    report = _run_analysis(rules, files)

    # ── Output ────────────────────────────────────────────────────────────────
    _print_report(report, files, rules, args.output)


if __name__ == "__main__":
    main()
