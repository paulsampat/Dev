# Testing skill

## Framework
- Use pytest for all tests
- Tests go in `tests/` mirroring the source structure (e.g. `tests/test_users.py` for `routers/users.py`)
- Test files must be prefixed `test_`, test functions must be prefixed `test_`

## Conventions
- One test file per module being tested
- Use `pytest.fixture` for shared setup — define fixtures in `tests/conftest.py`
- Prefer `assert` statements over unittest-style assertions
- Name tests descriptively: `test_create_user_returns_201`, not `test_create`

## Coverage
- Every public function needs at least one happy-path test
- Add a test for the main failure/edge case too (invalid input, missing resource)
- For FastAPI routes, use `httpx.AsyncClient` with `app` as the transport — do not spin up a real server
