"""Pydantic models for exploratory analysis.

Two groups:

- Deterministic inputs (metrics, OCR text cards, per-shot context), computed in
  ``metrics.py`` from the persisted ``ExtractionSummary`` with no LLM. They are
  the numbers the report is built on, and are also exposed to the API.
- ``ExploreReport``, the persisted per-run report (``exploration_results``).
  The LLM writes its prose and labels; every number and timestamp in it comes
  from the deterministic inputs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Bump when ExploreReport changes shape; stored with each report.
EXPLORE_SCHEMA_VERSION = 1

TextCategory = Literal["brand", "cta", "price", "url", "caption", "other"]
Prominence = Literal["high", "medium", "low"]


class TimeSpan(BaseModel):
    start_sec: float
    end_sec: float


class ObjectStats(BaseModel):
    """One resolved object label aggregated across the run."""

    label: str
    # Distinct tracked identities (global ids) carrying this label.
    instance_count: int
    # Union of the label's lifespans, so overlapping instances aren't
    # double-counted.
    screen_time_sec: float
    first_seen_sec: float
    avg_screen_coverage: float


class RunMetrics(BaseModel):
    duration_sec: float
    shot_count: int
    avg_shot_sec: float
    cuts_per_minute: float
    # Fractions of the runtime (0-1).
    speech_coverage: float
    text_coverage: float
    # None when the run has no speech.
    words_per_minute: float | None
    # None when faces weren't detected at all (``no_objects`` runs), as
    # opposed to 0.0 for "detected, none found".
    face_coverage: float | None
    objects_tracked: bool
    # Sorted by screen time, longest first. Empty for ``no_objects`` runs.
    objects: list[ObjectStats]


class OcrTextCard(BaseModel):
    """One distinct piece of on-screen text, deduplicated across OCR noise."""

    # Most frequent raw variant of the text.
    text: str
    # Contiguous on-screen intervals, in order.
    spans: list[TimeSpan]
    total_sec: float


class ShotContext(BaseModel):
    """Everything extracted for one shot, aligned on the shot's time range."""

    shot_index: int
    start_sec: float
    end_sec: float
    description: str | None
    objects: list[str]
    on_screen_text: list[str]
    speech: str


# ── Report ──────────────────────────────────────────────────────────────────


class Chapter(BaseModel):
    title: str
    start_sec: float
    end_sec: float
    summary: str


class TimedText(BaseModel):
    text: str
    start_sec: float


class TranscriptDigest(BaseModel):
    summary: str
    topics: list[str]
    # Each claim is anchored to the start of the speech segment it came from.
    key_claims: list[TimedText]


class TextCard(OcrTextCard):
    """An OCR text card plus the LLM's cleaned reading and category."""

    clean_text: str
    category: TextCategory


class OnScreenTextDigest(BaseModel):
    summary: str
    cards: list[TextCard]


class VisualSubject(BaseModel):
    """A tracked object label; numbers from metrics, prominence from the LLM."""

    label: str
    screen_time_sec: float
    first_seen_sec: float
    prominence: Prominence


class VisualDigest(BaseModel):
    summary: str
    # Empty for ``no_objects`` runs: the summary then comes from scene
    # descriptions alone.
    subjects: list[VisualSubject]


class Finding(BaseModel):
    observation: str
    evidence_secs: list[float]


class ExploreReport(BaseModel):
    schema_version: int = EXPLORE_SCHEMA_VERSION
    summary: str
    chapters: list[Chapter]
    # None when the run has no data for that modality (no speech, no on-screen
    # text, nothing visual) or its digest failed.
    transcript: TranscriptDigest | None
    on_screen_text: OnScreenTextDigest | None
    visuals: VisualDigest | None
    # Filled by the findings stage (stage C); empty until then.
    findings: list[Finding] = Field(default_factory=list)
