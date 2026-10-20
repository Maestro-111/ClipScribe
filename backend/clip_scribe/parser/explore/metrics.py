"""Deterministic inputs for exploratory analysis (no LLM).

Everything here is a pure function over the persisted ``ExtractionSummary``
(``reader.get_extraction_summary``), so it is cheap, reproducible, and testable
with plain dict fixtures. The LLM stages consume these results and only ever
phrase them; every number in the report comes from here.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from clip_scribe.parser.explore.models import (
    ObjectStats,
    OcrTextCard,
    RunMetrics,
    ShotContext,
    TimeSpan,
)

if TYPE_CHECKING:
    from clip_scribe.extractor.extractor_models import (
        AudioSegment,
        ExtractionSummary,
        TextEvent,
    )

# OCR reads of the same on-screen text vary between frames ("DRIVE ITHOME" vs
# "DRIVE IT HOME", "$8.800" vs "$8,800"). Lines whose normalized keys are at
# least this similar are treated as the same text.
_TEXT_SIMILARITY = 0.8
# Below this normalized length a fuzzy match is too likely to merge unrelated
# words ("jeep"/"deep"), so short keys only merge on exact equality.
_MIN_FUZZY_KEY_LEN = 5
# Single characters are almost always OCR noise.
_MIN_KEY_LEN = 2


def _round(value: float) -> float:
    return round(value, 3)


def run_duration(summary: ExtractionSummary) -> float:
    duration = summary["global_stats"].get("video_duration")
    if duration:
        return float(duration)
    shots = summary["shot_boundaries"]
    return float(max((s["end"] for s in shots), default=0.0))


def _union_length(spans: Iterable[tuple[float, float]], limit: float) -> float:
    """Total length covered by ``spans`` clipped to ``[0, limit]``."""
    clipped = sorted(
        (max(0.0, start), min(limit, end)) for start, end in spans if end > start
    )
    total = 0.0
    cur_start: float | None = None
    cur_end = 0.0
    for start, end in clipped:
        if cur_start is None or start > cur_end:
            if cur_start is not None:
                total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_start is not None:
        total += cur_end - cur_start
    return total


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _text_key(raw: str) -> str:
    return re.sub(r"[^0-9a-z]", "", raw.casefold())


@dataclass
class _CardBuilder:
    keys: list[str]
    variants: Counter[str] = field(default_factory=Counter)
    seconds: set[int] = field(default_factory=set)

    def matches(self, key: str) -> bool:
        if key in self.keys:
            return True
        if len(key) < _MIN_FUZZY_KEY_LEN:
            return False
        return any(
            len(k) >= _MIN_FUZZY_KEY_LEN
            and SequenceMatcher(None, key, k).ratio() >= _TEXT_SIMILARITY
            for k in self.keys
        )


def build_text_cards(
    text_events: list[TextEvent], duration_sec: float
) -> list[OcrTextCard]:
    """Collapse per-second OCR lines into distinct, timed text cards.

    Cards are ordered by first appearance. A card's spans are its runs of
    consecutive seconds; each second covers ``[s, s + 1)``, clipped to the
    video duration.
    """
    builders: list[_CardBuilder] = []
    for event in sorted(text_events, key=lambda e: e["second"]):
        for raw in event["text"]:
            text = raw.strip()
            key = _text_key(text)
            if len(key) < _MIN_KEY_LEN:
                continue
            builder = next((b for b in builders if b.matches(key)), None)
            if builder is None:
                builder = _CardBuilder(keys=[key])
                builders.append(builder)
            elif key not in builder.keys:
                builder.keys.append(key)
            builder.variants[text] += 1
            builder.seconds.add(event["second"])

    limit = duration_sec if duration_sec > 0 else float("inf")
    cards: list[OcrTextCard] = []
    for builder in builders:
        spans: list[TimeSpan] = []
        run_start: int | None = None
        prev = 0
        for second in sorted(builder.seconds):
            if run_start is None:
                run_start = second
            elif second != prev + 1:
                spans.append(
                    TimeSpan(start_sec=run_start, end_sec=min(prev + 1, limit))
                )
                run_start = second
            prev = second
        if run_start is not None:
            spans.append(TimeSpan(start_sec=run_start, end_sec=min(prev + 1, limit)))
        cards.append(
            OcrTextCard(
                # Counter.most_common keeps first-seen order on ties.
                text=builder.variants.most_common(1)[0][0],
                spans=spans,
                total_sec=_round(sum(s.end_sec - s.start_sec for s in spans)),
            )
        )
    return cards


def _object_stats(summary: ExtractionSummary) -> list[ObjectStats]:
    # The persisted columns are nullable: an occurrence without a lifespan
    # can't be placed in time and is skipped; a missing coverage is left out
    # of the average rather than counted as zero.
    by_label: dict[str, list[tuple[int, float, float, float | None]]] = {}
    for obj in summary["visual_objects"]:
        for occ in obj["occurrences"]:
            start, end = occ["lifespan"][0], occ["lifespan"][1]
            if start is None or end is None:
                continue
            by_label.setdefault(obj["label"], []).append(
                (obj["global_id"], start, end, occ["screen_coverage"])
            )

    stats: list[ObjectStats] = []
    for label, occs in by_label.items():
        coverages = [c for _, _, _, c in occs if c is not None]
        stats.append(
            ObjectStats(
                label=label,
                instance_count=len({gid for gid, _, _, _ in occs}),
                screen_time_sec=_round(
                    _union_length(((s, e) for _, s, e, _ in occs), float("inf"))
                ),
                first_seen_sec=_round(min(s for _, s, _, _ in occs)),
                avg_screen_coverage=_round(sum(coverages) / len(coverages))
                if coverages
                else 0.0,
            )
        )
    return sorted(stats, key=lambda o: (-o.screen_time_sec, o.first_seen_sec))


def compute_run_metrics(
    summary: ExtractionSummary, *, objects_tracked: bool
) -> RunMetrics:
    """Run-level numbers for the KPI strip and the LLM stages.

    ``objects_tracked`` is the run's extraction mode (``with_objects``): object
    and face detection only run then, so their absence otherwise means
    "unknown", not "none".
    """
    duration = run_duration(summary)
    minutes = duration / 60 if duration > 0 else 0.0
    shot_count = len(summary["shot_boundaries"]) or int(
        summary["global_stats"].get("total_shots") or 0
    )

    audio = summary["audio_segments"]
    speech_sec = _union_length(((a["start"], a["end"]) for a in audio), duration)
    words = sum(len(a["text"].split()) for a in audio)

    text_seconds = {
        e["second"]
        for e in summary["text_events"]
        if any(len(_text_key(line)) >= _MIN_KEY_LEN for line in e["text"])
    }

    face_coverage: float | None = None

    if objects_tracked:
        face_seconds = {
            int(d["timestamp_sec"])
            for d in summary["frame_detections"]
            if d["source"] == "mtcnn"
        }
        face_coverage = (
            _round(min(1.0, len(face_seconds) / duration)) if duration else 0.0
        )

    return RunMetrics(
        duration_sec=_round(duration),
        shot_count=shot_count,
        avg_shot_sec=_round(duration / shot_count) if shot_count else 0.0,
        cuts_per_minute=_round((shot_count - 1) / minutes)
        if minutes and shot_count > 1
        else 0.0,
        speech_coverage=_round(speech_sec / duration) if duration else 0.0,
        text_coverage=_round(min(1.0, len(text_seconds) / duration))
        if duration
        else 0.0,
        words_per_minute=_round(words / (speech_sec / 60)) if speech_sec > 0 else None,
        face_coverage=face_coverage,
        objects_tracked=objects_tracked,
        objects=_object_stats(summary) if objects_tracked else [],
    )


def _assign_speech(
    audio: list[AudioSegment], shots: list[tuple[int, float, float]]
) -> dict[int, list[str]]:
    """Give each speech segment to the shot it overlaps most, so a segment that
    straddles a cut is quoted once rather than in both shots."""
    by_shot: dict[int, list[str]] = {}
    for segment in sorted(audio, key=lambda a: a["start"]):
        best = max(
            shots,
            key=lambda s: _overlap(segment["start"], segment["end"], s[1], s[2]),
            default=None,
        )
        if (
            best is None
            or _overlap(segment["start"], segment["end"], best[1], best[2]) <= 0
        ):
            continue
        by_shot.setdefault(best[0], []).append(segment["text"].strip())
    return by_shot


def build_shot_context(
    summary: ExtractionSummary, text_cards: list[OcrTextCard]
) -> list[ShotContext]:
    """One row per shot: scene description, objects, on-screen text, speech.

    On-screen text uses the deduplicated ``text_cards`` (a card is listed for
    every shot it overlaps), so OCR noise doesn't leak into the LLM input.
    """
    shots = [
        (s["index"], float(s["start"]), float(s["end"]))
        for s in sorted(summary["shot_boundaries"], key=lambda s: s["start"])
    ]
    descriptions = {
        d["shot_index"]: d["description"] for d in summary["scene_descriptions"]
    }

    objects_by_shot: dict[int, list[tuple[float, str]]] = {}

    for obj in summary["visual_objects"]:
        for occ in obj["occurrences"]:
            objects_by_shot.setdefault(occ["shot_index"], []).append(
                (occ["lifespan"][0], obj["label"])
            )

    speech_by_shot = _assign_speech(summary["audio_segments"], shots)

    context: list[ShotContext] = []
    for index, start, end in shots:
        labels = [label for _, label in sorted(objects_by_shot.get(index, []))]
        context.append(
            ShotContext(
                shot_index=index,
                start_sec=_round(start),
                end_sec=_round(end),
                description=descriptions.get(index),
                objects=list(dict.fromkeys(labels)),
                on_screen_text=[
                    card.text
                    for card in text_cards
                    if any(
                        _overlap(span.start_sec, span.end_sec, start, end) > 0
                        for span in card.spans
                    )
                ],
                speech=" ".join(speech_by_shot.get(index, [])),
            )
        )
    return context
