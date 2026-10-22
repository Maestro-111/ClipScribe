"""Build the exploratory report for one run (stage B: digests + overview).

Flow: the deterministic inputs from ``metrics.py`` feed three digest calls run
in parallel (transcript, on-screen text, visuals), then one synthesis call
writes the overview summary and chapters.

These are plain structured-output LLM calls, not agents: every input is known
up front. The LLM refers to segments, cards, shots, and labels by index or
exact label; this module maps those back to real timestamps and numbers and
drops anything that doesn't resolve, so the report never carries an invented
figure.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from clip_scribe.parser.explore.metrics import (
    build_shot_context,
    build_text_cards,
    compute_run_metrics,
    run_duration,
)
from clip_scribe.parser.explore.models import (
    Chapter,
    ExploreReport,
    OcrTextCard,
    OnScreenTextDigest,
    Prominence,
    RunMetrics,
    ShotContext,
    TextCard,
    TextCategory,
    TimedText,
    TranscriptDigest,
    VisualDigest,
    VisualSubject,
)
from clip_scribe.parser.explore.prompts import (
    ON_SCREEN_TEXT_PROMPT,
    SYNTHESIS_PROMPT,
    TRANSCRIPT_PROMPT,
    VISUALS_PROMPT,
    _SHARED_RULES,
)
from clip_scribe.utils.clip_scribe_cancel import (
    CancellationToken,
    NullCancellationToken,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from clip_scribe.extractor.extractor_models import (
        AudioSegment,
        ExtractionSummary,
    )
    from clip_scribe.utils.clip_scribe_prompts import _PromptLimitsParametrizer

logger = logging.getLogger("clip_scribe")

_Out = TypeVar("_Out", bound=BaseModel)
_T = TypeVar("_T")


# ── LLM structured outputs (index-based; resolved by the code below) ─────────


class _ClaimRef(BaseModel):
    text: str
    segment: int = Field(description="Index of the transcript segment.")


class _TranscriptOutput(BaseModel):
    summary: str
    topics: list[str]
    key_claims: list[_ClaimRef]


class _CardLabel(BaseModel):
    card: int = Field(description="Index of the text card.")
    clean_text: str
    category: TextCategory


class _OnScreenTextOutput(BaseModel):
    summary: str
    cards: list[_CardLabel]


class _SubjectRating(BaseModel):
    label: str = Field(description="Exact object label from the input.")
    prominence: Prominence


class _VisualsOutput(BaseModel):
    summary: str
    subjects: list[_SubjectRating]


class _ChapterRef(BaseModel):
    title: str
    first_shot: int = Field(description="Index of the chapter's first shot.")
    last_shot: int = Field(description="Index of the chapter's last shot.")
    summary: str


class _SynthesisOutput(BaseModel):
    summary: str
    chapters: list[_ChapterRef]


# ── Explorer ─────────────────────────────────────────────────────────────────


class VideoExplorer:
    def __init__(
        self,
        model: BaseChatModel,
        *,
        prompt_parametrizer: _PromptLimitsParametrizer,
        cancel_token: CancellationToken | None = None,
    ) -> None:
        self.model = model
        self._cancel = cancel_token or NullCancellationToken()
        self.prompt_parametrizer = prompt_parametrizer

    def _limit(self, key: str) -> int:
        """A list-length limit from the same params the prompts are filled from,
        so what the prompt asks for and what the code enforces can't differ."""
        return self.prompt_parametrizer.params[key]

    def _structured(self, schema: type[_Out], system: str, payload: object) -> _Out:
        """One structured-output call; the input is sent as compact JSON."""
        result = self.model.with_structured_output(schema).invoke(
            [
                SystemMessage(content=system),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ]
        )
        assert isinstance(result, schema)
        return result

    def explore(
        self, summary: ExtractionSummary, *, objects_tracked: bool
    ) -> ExploreReport:
        duration = run_duration(summary)
        metrics = compute_run_metrics(summary, objects_tracked=objects_tracked)
        cards = build_text_cards(summary["text_events"], duration)
        shots = build_shot_context(summary, cards)
        audio = sorted(summary["audio_segments"], key=lambda a: a["start"])

        self._cancel.check()

        with ThreadPoolExecutor(max_workers=3) as pool:
            transcript_f = pool.submit(
                self._safe, "transcript", lambda: self._transcript(audio)
            )
            text_f = pool.submit(
                self._safe, "on-screen text", lambda: self._on_screen_text(cards)
            )
            visuals_f = pool.submit(
                self._safe, "visuals", lambda: self._visuals(metrics, shots)
            )

            transcript = transcript_f.result()
            on_screen_text = text_f.result()
            visuals = visuals_f.result()

        # Last cheap bail before the overview call.
        self._cancel.check()

        overview = self._structured(
            _SynthesisOutput,
            self.prompt_parametrizer.parametrize(SYNTHESIS_PROMPT, _SHARED_RULES),
            {
                "metrics": metrics.model_dump(exclude={"objects"}),
                "transcript_summary": transcript.summary if transcript else None,
                "on_screen_text_summary": on_screen_text.summary
                if on_screen_text
                else None,
                "visuals_summary": visuals.summary if visuals else None,
                "shots": [
                    {"index": i, **shot.model_dump(exclude={"shot_index"})}
                    for i, shot in enumerate(shots)
                ],
            },
        )

        return ExploreReport(
            summary=overview.summary,
            chapters=_resolve_chapters(
                overview.chapters,
                shots,
                max_chapters=self._limit("overview_summary_chapters_num"),
            ),
            transcript=transcript,
            on_screen_text=on_screen_text,
            visuals=visuals,
            metrics=metrics,
        )

    def _safe(self, name: str, fn: Callable[[], _T | None]) -> _T | None:
        """Run one digest; a failure drops that section instead of the report.

        Only digests degrade this way. The overview call is not wrapped: a
        report without a summary isn't worth persisting.
        """
        try:
            return fn()
        except Exception:  # noqa: BLE001 - logged; the section is left empty
            logger.exception("ClipScribe Explorer: %s digest failed", name)
            return None

    def _transcript(self, audio: list[AudioSegment]) -> TranscriptDigest | None:
        if not audio:
            return None
        out = self._structured(
            _TranscriptOutput,
            self.prompt_parametrizer.parametrize(TRANSCRIPT_PROMPT, _SHARED_RULES),
            [
                {
                    "index": i,
                    "start_sec": round(a["start"], 2),
                    "end_sec": round(a["end"], 2),
                    "text": a["text"].strip(),
                }
                for i, a in enumerate(audio)
            ],
        )
        claims = [
            TimedText(text=c.text, start_sec=round(audio[c.segment]["start"], 3))
            for c in out.key_claims
            if 0 <= c.segment < len(audio)
        ]
        # The prompt only asks for these limits; enforce them here.
        return TranscriptDigest(
            summary=out.summary,
            topics=out.topics[: self._limit("transcript_topics_num")],
            key_claims=claims[: self._limit("transcript_key_claims_num")],
        )

    def _on_screen_text(self, cards: list[OcrTextCard]) -> OnScreenTextDigest | None:
        if not cards:
            return None
        # The LLM cleans and categorizes only the cards visible the longest, to
        # bound the call; every card stays in the digest. Sent cards keep their
        # original index so the answers map back without translation.
        sent = set(
            sorted(range(len(cards)), key=lambda i: -cards[i].total_sec)[
                : self._limit("on_screen_text_cards_num")
            ]
        )
        out = self._structured(
            _OnScreenTextOutput,
            self.prompt_parametrizer.parametrize(ON_SCREEN_TEXT_PROMPT, _SHARED_RULES),
            [
                {
                    "index": i,
                    "text": card.text,
                    "visible": [[s.start_sec, s.end_sec] for s in card.spans],
                }
                for i, card in enumerate(cards)
                if i in sent
            ],
        )
        labels = {label.card: label for label in out.cards if label.card in sent}
        return OnScreenTextDigest(
            summary=out.summary,
            cards=[
                TextCard(
                    **card.model_dump(),
                    # Cards the LLM wasn't sent (or skipped) keep their raw
                    # reading.
                    clean_text=labels[i].clean_text if i in labels else card.text,
                    category=labels[i].category if i in labels else "other",
                )
                for i, card in enumerate(cards)
            ],
        )

    def _visuals(
        self, metrics: RunMetrics, shots: list[ShotContext]
    ) -> VisualDigest | None:
        scenes = [
            {"index": i, "description": s.description}
            for i, s in enumerate(shots)
            if s.description
        ]
        if not metrics.objects and not scenes:
            return None
        # ``metrics.objects`` is sorted by screen time. The LLM rates only the
        # top ones to bound the call; every tracked object stays in the digest.
        rated = metrics.objects[: self._limit("visuals_subjects_num")]
        out = self._structured(
            _VisualsOutput,
            self.prompt_parametrizer.parametrize(VISUALS_PROMPT, _SHARED_RULES),
            {
                "scenes": scenes,
                "objects": [
                    {
                        "label": o.label,
                        "screen_time_sec": o.screen_time_sec,
                        "avg_screen_coverage": o.avg_screen_coverage,
                    }
                    for o in rated
                ],
            },
        )
        ratings = {r.label.casefold(): r.prominence for r in out.subjects}
        return VisualDigest(
            summary=out.summary,
            subjects=[
                VisualSubject(
                    label=o.label,
                    screen_time_sec=o.screen_time_sec,
                    first_seen_sec=o.first_seen_sec,
                    # Objects that weren't rated (beyond the top ones, or skipped
                    # by the model) default to low.
                    prominence=ratings.get(o.label.casefold(), "low"),
                )
                for o in metrics.objects
            ],
        )


def _resolve_chapters(
    refs: list[_ChapterRef], shots: list[ShotContext], *, max_chapters: int
) -> list[Chapter]:
    """Map shot-index chapters to time ranges, repairing order and overlap.

    Indices are clamped to the shot list; a chapter starting inside the
    previous one is pushed past it, and one left empty is dropped. At most
    ``max_chapters`` are kept, in order, so an over-long answer loses its
    trailing chapters rather than being reshuffled.
    """
    if not shots:
        return []
    last_index = len(shots) - 1
    chapters: list[Chapter] = []
    next_free = 0
    for ref in sorted(refs, key=lambda r: r.first_shot):
        first = max(next_free, min(max(ref.first_shot, 0), last_index))
        last = min(max(ref.last_shot, first), last_index)
        if first > last or first < next_free:
            continue
        chapters.append(
            Chapter(
                title=ref.title,
                start_sec=shots[first].start_sec,
                end_sec=shots[last].end_sec,
                summary=ref.summary,
            )
        )
        next_free = last + 1
        if next_free > last_index or len(chapters) >= max_chapters:
            break
    return chapters
