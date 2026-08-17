"""Shared test fixtures.

Pins storage to the local backend for the whole suite so a developer's ambient
repo-root ``.env`` (e.g. ``CLIPSCRIBE_STORAGE_BACKEND=gcs``) can't push tests
down the GCS path — which would sign URLs, reach the network, or even write to a
real bucket. Mirrors the ``CLIPSCRIBE_JOB_BACKEND`` pin already done per-test in
test_api_jobs. Tests that exercise the GCS backend do so by constructing the
classes directly with a fake client, not through the environment.
"""

import pytest

from app import settings as settings_mod


@pytest.fixture(autouse=True)
def _pin_local_storage(monkeypatch):
    monkeypatch.setenv("CLIPSCRIBE_STORAGE_BACKEND", "local")
    monkeypatch.delenv("CLIPSCRIBE_GCS_BUCKET", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    # Auth is user-scoped (jobs.created_by), but the suite sends no bearer
    # token. Enable the dev anonymous-local fallback so current_user_id resolves
    # every request to DEFAULT_USER_ID ("local") — exercising the real auth
    # dependency rather than stubbing it. Seeds therefore use created_by="local".
    monkeypatch.setenv("CLIPSCRIBE_ALLOW_ANONYMOUS_LOCAL", "true")
    settings_mod.get_settings.cache_clear()
    yield
    settings_mod.get_settings.cache_clear()
