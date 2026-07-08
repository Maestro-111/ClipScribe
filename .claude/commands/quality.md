---
description: Run the standard ClipScribe quality checks.
argument-hint: "[optional changed-file scope]"
---

# Quality

Run quality checks for the editable project code. Do not inspect or modify `backend/src/dino/groundingdino/**` or `backend/src/sam2/**`.

Run all commands below from the `backend/` directory (the Python project lives in `backend/` in this monorepo). pre-commit in particular must be invoked from `backend/`, since it discovers `backend/.pre-commit-config.yaml` from the current directory but executes hooks from the git root.

1. Identify the relevant changed files from `$ARGUMENTS` or `git status --short`.
2. Run the test suite:

```bash
uv run pytest -q
```

3. Run mypy on editable core modules:

```bash
uv run mypy --config-file=pyproject.toml --explicit-package-bases src/clip_scribe src/extractor src/ocr src/parser
```

4. If frontend files changed, run the frontend typecheck from `frontend/`:

```bash
(cd ../frontend && pnpm typecheck)
```

For full frontend validation, run:

```bash
(cd ../frontend && pnpm build)
```

5. Run pre-commit when the user asks for full validation or before preparing a commit:

```bash
uv run pre-commit run --all-files
```

6. If a backend tool is missing, recommend `uv sync --extra dev`; if a frontend tool is missing, recommend `pnpm install` from `frontend/` rather than installing ad hoc packages.
7. If failures are unrelated to the task, report them separately and avoid broad refactors.

Summarize the commands run and the remaining failures.
