---
description: Prepare or verify the ClipScribe development environment.
argument-hint: "[optional setup goal]"
---

# Setup Dev

Use this workflow to prepare or inspect the local development environment.

1. Read `CLAUDE.md` or `AGENTS.md` before making changes.
2. Confirm the requested setup scope from the user input: `$ARGUMENTS`.
3. Prefer lightweight checks first:

```bash
uv --version
uv run python --version
```

4. Install project dependencies when requested:

```bash
uv sync
```

5. Install development tools when requested:

```bash
uv sync --extra dev
```

6. Do not run heavyweight setup without explicit user approval:

```bash
make setup
make checkpoints
```

7. Check environment variables only at the level needed for the task:
   - `OPENAI_API_KEY` for OpenAI-powered scene, taxonomy, and parser work.
   - `POSTGRESQL_URL` when `database.backend` is `postgresql`.
   - `SQLITE_URL` only when using SQLite.

Report what was verified, what changed, and any missing setup.
