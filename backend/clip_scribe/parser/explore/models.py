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

from enum import Enum
from typing import Literal, TypedDict

from pydantic import BaseModel, Field

# Bump when ExploreReport changes shape; stored with each report.
EXPLORE_SCHEMA_VERSION = 1


class TextCategory(str, Enum):
    """What an on-screen text card is for.

    The single source of truth for the categories: the on-screen text prompt, the
    API schema, and the frontend types are all derived from this enum, and
    ``TEXT_CATEGORY_DESCRIPTIONS`` must describe every member (tested). To add
    or rename one, edit here, then update ``CATEGORY_STYLE`` in the frontend
    (the typecheck fails until every category has a style).
    """

    BRAND = "brand"
    PRODUCT = "product"
    SLOGAN = "slogan"
    CTA = "cta"
    PRICE = "price"
    PROMO = "promo"
    URL = "url"
    SOCIAL = "social"
    LEGAL = "legal"
    CAPTION = "caption"
    OTHER = "other"


# Shown to the LLM, so write each as a short noun phrase it can match text to.
# Must not contain braces: the prompt is later filled with ``str.format_map``.
TEXT_CATEGORY_DESCRIPTIONS: dict[TextCategory, str] = {
    TextCategory.BRAND: "brand or company names",
    TextCategory.PRODUCT: "product, model, or service names",
    TextCategory.SLOGAN: "slogans, taglines, and campaign names",
    TextCategory.CTA: "calls to action, e.g. buy now, learn more",
    TextCategory.PRICE: "prices, discounts, financing, and incentives",
    TextCategory.PROMO: "time-limited or event messaging, e.g. sale ends soon",
    TextCategory.URL: "web addresses",
    TextCategory.SOCIAL: "social media handles and hashtags",
    TextCategory.LEGAL: "disclaimers, fine print, and terms",
    TextCategory.CAPTION: "subtitles or descriptive text",
    TextCategory.OTHER: "anything that fits none of the above",
}

Prominence = Literal["high", "medium", "low"]


class SummarySentencesNum(TypedDict):
    transcript_summary_sentences_num: int
    text_summary_sentences_num: int
    visual_summary_sentences_num: int
    overview_summary_sentences_num: int


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
    metrics: RunMetrics
