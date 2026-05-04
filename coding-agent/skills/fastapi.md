# FastAPI Skill

**Purpose:** Enforce FastAPI conventions and best practices when building or reviewing a FastAPI service.
**Use when:** Generating a new FastAPI project, adding routes or routers, or reviewing existing FastAPI code.
**Invoke with:** `--skill fastapi` or `/fastapi`

---

## Project structure
- Entry point is `main.py` with an `app = FastAPI()` instance
- Routers go in `routers/` — one file per domain (e.g. `routers/users.py`)
- Pydantic models go in `models/` — separate request and response schemas

## Conventions
- Use `APIRouter` with a `prefix` and `tags` for every router
- All path operation functions must be `async def`
- Use `Annotated` + `Depends` for dependency injection, not bare `= Depends(...)`
- Return Pydantic response models explicitly — set `response_model=` on every route
- Raise `HTTPException` with appropriate status codes; never return error dicts

## Always include
- A `GET /health` endpoint on the root app returning `{"status": "ok"}`
- Lifespan context manager for startup/shutdown logic (not deprecated `@app.on_event`)
