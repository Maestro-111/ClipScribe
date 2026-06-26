---
description: Run the standard ClipScribe quality checks.
argument-hint: "[optional changed-file scope]"
---

# Quality

Run quality checks for the editable project code. Do not inspect or modify `src/dino/groundingdino/**` or `src/sam2/**`.

1. Identify the relevant changed files from `$ARGUMENTS` or `git status --short`.
2. Run the test suite:

```bash
uv run pytest -q
```

3. Run mypy on editable core modules:

```bash
uv run mypy --config-file=pyproject.toml --explicit-package-bases src/clip_scribe src/extractor src/ocr src/parser
```

4. Run pre-commit when the user asks for full validation or before preparing a commit:

```bash
uv run pre-commit run --all-files
```

5. If a tool is missing, recommend `uv sync --extra dev` rather than installing ad hoc packages.
6. If failures are unrelated to the task, report them separately and avoid broad refactors.

Summarize the commands run and the remaining failures.
