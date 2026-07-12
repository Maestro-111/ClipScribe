---
description: Typecheck ClipScribe and fix typing issues with minimal code churn.
argument-hint: "[optional module or file]"
---

# Typecheck

Use this workflow for mypy and typing cleanup. Run the commands below from the `backend/` directory (the Python project lives in `backend/` in this monorepo).

1. Keep changes inside existing modules whenever practical. Do not create new abstraction layers just to satisfy mypy.
2. Prefer precise annotations over `Any`; use `TypedDict`, dataclasses, Pydantic models, protocols, or concrete generics when they match existing code.
3. Avoid blanket `# type: ignore`. Use narrow ignores only when an external library type stub is wrong or missing.
4. Run:

```bash
uv run mypy --config-file=pyproject.toml --explicit-package-bases src/clip_scribe src/extractor src/ocr src/parser
```

5. Fix only issues connected to `$ARGUMENTS` unless the user asked for a full typing pass.
6. Re-run the typecheck after edits and report any residual failures.
