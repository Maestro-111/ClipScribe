---
description: Prepare or verify the ClipScribe development environment.
argument-hint: "[optional setup goal]"
---

# Setup Dev

Use this workflow to prepare or inspect the local development environment.

1. Read `CLAUDE.md` or `AGENTS.md` before making changes.
2. Confirm the requested setup scope from the user input: `$ARGUMENTS`.
3. Run Python project commands from `backend/` unless explicitly using the root `Makefile`.
4. Prefer lightweight checks first:

```bash
uv --version
uv run python --version
```

5. Install project dependencies when requested:

```bash
uv sync
```

6. Install development tools when requested:

```bash
uv sync --extra dev
```

7. Install frontend dependencies when the setup scope includes the dashboard:

```bash
(cd ../frontend && pnpm install && pnpm gen:api)
```

`pnpm gen:api` requires the FastAPI app to be running on `localhost:8000`.

8. Apply database migrations when preparing a fresh database:

```bash
uv run alembic upgrade head
```

9. Do not run heavyweight setup without explicit user approval. The root `Makefile` setup/checkpoint/clean targets are stale after the backend move, so verify or fix them before relying on them.

10. Check environment variables only at the level needed for the task:
   - `OPENAI_API_KEY` for OpenAI-powered scene, taxonomy, and parser work.
   - `POSTGRESQL_URL` when `database.backend` is `postgresql`.
   - `SQLITE_URL` only when using SQLite.

Report what was verified, what changed, and any missing setup.
