"""Identifier generation for runs and jobs.

ULIDs are used (rather than UUID4) because they are lexicographically
sortable by creation time — so ``ORDER BY run_id`` / ``job_id`` yields
chronological order, which the jobs-list UI and run history rely on. The
string form is 26 chars, URL-safe, and exposed directly in API paths.
"""

from __future__ import annotations

from ulid import ULID


def new_ulid() -> str:
    """Return a fresh, time-sortable ULID as a 26-char string."""
    return str(ULID())
