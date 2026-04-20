# Python style skill

## Type hints
- All function signatures must have type hints on parameters and return type
- Use `X | None` instead of `Optional[X]`
- Use `list[X]` / `dict[K, V]` instead of `List[X]` / `Dict[K, V]` (Python 3.10+)

## Structure
- Use dataclasses (`@dataclass`) for structured data, not plain dicts
- Keep functions under 30 lines — extract helpers if they grow larger
- No global mutable state
- Prefer `pathlib.Path` over `os.path` string manipulation

## Formatting
- 4-space indentation
- Max line length 100 characters
- Use f-strings for string formatting, not `.format()` or `%`
- One blank line between methods, two between top-level definitions
