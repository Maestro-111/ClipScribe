---
description: Run or prepare a parser-only ClipScribe evaluation.
argument-hint: "<run_id> [video name/platform details]"
---

# Run Parser

Use this workflow when the user wants parser-only evaluation for an existing extraction run.

1. Confirm the `run_id` and platform details from `$ARGUMENTS`.
2. Run backend commands from `backend/`.
3. Check `src/clip_scribe/configs/clip_scribe.yaml` for `database.backend`.
4. Ensure the matching database environment exists:
   - PostgreSQL: `POSTGRESQL_URL`.
   - SQLite: `SQLITE_URL` or default `sqlite:///data/clip_scribe.db`.
5. Apply migrations (`uv run alembic upgrade head`) before parser work against a fresh database.
6. Verify the run exists before running the parser.
7. Prefer existing builder/parser code. Treat `main.py` as a temporary hardcoded scratch entry point, not a stable CLI.
8. Do not add task-specific scripts. If a reusable entry point is needed, modify the existing CLI/entry-point design instead.
9. Run the parser only after the run and platform inputs are clear.

Report the run id, database backend, output report path, and any failures.
