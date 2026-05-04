"""
Post-execution validation for the coding agent.

After every file write, validate the written code against all architectural
rule files in the rules/ directory. Validation is performed by a fast Claude
Haiku call — Claude reads the rules and the code and decides if it complies.

If violations are found, the error string is returned as the tool result so
the main agent sees its own violations and corrects the code before moving on.

Rules live in:  rules/<name>.md   (plain English architectural guidelines)
Validates:      .py, .ts, .js, .tsx, .jsx files only
Model used:     claude-haiku-4-5-20251001  (fast and cheap for validation)
"""

import os
from pathlib import Path

import anthropic

# ── Configuration ─────────────────────────────────────────────────────────────

RULES_DIR = Path(__file__).parent.parent / "rules"

VALIDATION_MODEL = "claude-haiku-4-5-20251001"

# Only validate these file types — no point checking configs, markdown, etc.
VALIDATED_EXTENSIONS = {".py", ".ts", ".js", ".tsx", ".jsx"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_rules() -> dict[str, str]:
    """
    Load all .md files from the rules/ directory.

    Returns a dict of {rule_name: content}. Empty dict if rules/ doesn't
    exist or contains no .md files — validation is silently skipped.
    """
    if not RULES_DIR.exists():
        return {}
    return {
        path.stem: path.read_text(encoding="utf-8").strip()
        for path in sorted(RULES_DIR.glob("*.md"))
        if path.is_file()
    }


# ── Public API ────────────────────────────────────────────────────────────────

def validate_file(file_path: Path, file_content: str) -> str | None:
    """
    Validate file content against all rules in rules/.

    Makes a single Claude Haiku API call with the rules and the file content.
    Returns None if compliant, or a violation report string if not.

    Only runs for source code file types — skips everything else silently.

    Args:
        file_path    : Resolved path of the file that was just written.
        file_content : The content that was written to the file.

    Returns:
        None if the file complies with all rules, or a string describing
        every violation found. The string is returned as a tool error so
        the main agent can see and fix the issues.
    """
    if file_path.suffix not in VALIDATED_EXTENSIONS:
        return None

    rules = _load_rules()
    if not rules:
        return None

    rules_text = "\n\n---\n\n".join(
        f"### {name}.md\n\n{content}"
        for name, content in rules.items()
    )

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    response = client.messages.create(
        model=VALIDATION_MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": (
                    f"You are a strict code reviewer enforcing architectural rules.\n\n"
                    f"## Rules all code must follow\n\n{rules_text}\n\n"
                    f"## File just written: {file_path.name}\n\n"
                    f"```\n{file_content}\n```\n\n"
                    f"Does this file comply with ALL the rules above?\n\n"
                    f"If fully compliant, respond with exactly: COMPLIANT\n\n"
                    f"If not compliant, respond with: VIOLATIONS FOUND\n"
                    f"Then list each violation on its own line, referencing the "
                    f"specific rule it breaks and exactly what needs to change."
                ),
            }
        ],
    )

    result = response.content[0].text.strip()

    if result.startswith("COMPLIANT"):
        return None

    return (
        f"Validation failed for {file_path.name}. Fix all violations before proceeding:\n\n"
        f"{result}"
    )
