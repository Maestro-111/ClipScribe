"""Exploratory report: index resolution, skipped/failed sections, persistence.

The LLM is stubbed at ``VideoExplorer._structured`` so these tests pin down the
code that turns index-based LLM output into a report with real timestamps.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import create_engine

from clip_scribe.db.reader import ClipScribeReaderDB
from clip_scribe.db.schema import metadata_obj
from clip_scribe.db.writer import ClipScribeWriterDB
from clip_scribe.parser.explore import explorer as explorer_mod
from clip_scribe.parser.explore.explorer import VideoExplorer
from clip_scribe.parser.explore.metrics import compute_run_metrics
from clip_scribe.parser.explore.models import ExploreReport
from clip_scribe.parser.explore.prompts import (
    _SHARED_RULES,
    ON_SCREEN_TEXT_PROMPT,
    SYNTHESIS_PROMPT,
    TRANSCRIPT_PROMPT,
    VISUALS_PROMPT,
)
from clip_scribe.utils.clip_scribe_prompts import _PromptLimitsParametrizer

from test.test_explore_metrics import _occ, _summary


_PROMPT_PARAMS = {
    "transcript_topics_num": 3,
    "transcript_key_claims_num": 4,
    "overview_summary_sentence_num": 5,
    "overview_summary_chapters_num": 7,
    "on_screen_text_cards_num": 20,
    "visuals_subjects_num": 15,
}


def _stub_outputs() -> dict[type, Any]:
    return {
        explorer_mod._TranscriptOutput: explorer_mod._TranscriptOutput(
            summary="talks",
            topics=["cars"],
            key_claims=[
                explorer_mod._ClaimRef(text="claim A", segment=1),
                # Out of range: dropped rather than given a made-up time.
                explorer_mod._ClaimRef(text="ghost", segment=9),
            ],
        ),
        explorer_mod._OnScreenTextOutput: explorer_mod._OnScreenTextOutput(
            summary="branding",
            cards=[
                explorer_mod._CardLabel(card=0, clean_text="Jeep", category="brand"),
                explorer_mod._CardLabel(
                    card=1, clean_text="Drive It Home", category="cta"
                ),
                explorer_mod._CardLabel(card=7, clean_text="ghost", category="url"),
            ],
        ),
        explorer_mod._VisualsOutput: explorer_mod._VisualsOutput(
            summary="a car",
            subjects=[
                explorer_mod._SubjectRating(label="CAR", prominence="high"),
                explorer_mod._SubjectRating(label="unicorn", prominence="high"),
            ],
        ),
        explorer_mod._SynthesisOutput: explorer_mod._SynthesisOutput(
            summary="an ad",
            chapters=[
                explorer_mod._ChapterRef(
                    title="Close", first_shot=2, last_shot=9, summary="end"
                ),
                explorer_mod._ChapterRef(
                    title="Open", first_shot=0, last_shot=1, summary="start"
                ),
                # Overlaps "Close" entirely: dropped.
                explorer_mod._ChapterRef(
                    title="Dup", first_shot=2, last_shot=2, summary="dup"
                ),
            ],
        ),
    }


class _StubExplorer(VideoExplorer):
    def __init__(
        self,
        outputs: dict[type, Any],
        fail: set[type] | None = None,
        params: dict[str, int] | None = None,
    ):
        super().__init__(
            model=None,  # type: ignore[arg-type]  # never called
            prompt_parametrizer=_PromptLimitsParametrizer(
                params=params or _PROMPT_PARAMS
            ),
        )
        self.outputs = outputs
        self.fail = fail or set()
        self.calls: list[type] = []
        self.payloads: dict[type, Any] = {}

    def _structured(self, schema, system, payload):  # type: ignore[no-untyped-def]
        self.calls.append(schema)
        self.payloads[schema] = payload
        if schema in self.fail:
            raise RuntimeError("llm down")
        return self.outputs[schema]


def _objects_summary() -> Any:
    return _summary(
        visual_objects=[
            {
                "global_id": 1,
                "label": "car",
                "occurrences": [_occ(shot_index=2, lifespan=[7.0, 12.0])],
            },
            {
                "global_id": 2,
                "label": "logo",
                "occurrences": [_occ(shot_index=0, lifespan=[0.0, 1.0])],
            },
        ]
    )


def test_report_resolves_indices_to_real_data():
    report = _StubExplorer(_stub_outputs()).explore(
        _objects_summary(), objects_tracked=True
    )

    assert report.summary == "an ad"
    # The metrics the digests were built from are stored with the report.
    assert report.metrics.shot_count == 3
    assert [o.label for o in report.metrics.objects] == ["car", "logo"]
    # Claims carry the timestamp of the referenced speech segment.
    assert [(c.text, c.start_sec) for c in report.transcript.key_claims] == [
        ("claim A", 8.0)
    ]
    cards = report.on_screen_text.cards
    assert [(c.text, c.clean_text, c.category) for c in cards] == [
        ("Jeep", "Jeep", "brand"),
        ("DRIVE IT HOME", "Drive It Home", "cta"),
        # Not labeled by the LLM: raw text, category other.
        ("HOLIDAY EVENT", "HOLIDAY EVENT", "other"),
    ]
    # Subjects mirror the tracked objects; numbers come from metrics.
    subjects = {s.label: s for s in report.visuals.subjects}
    assert subjects.keys() == {"car", "logo"}
    assert subjects["car"].prominence == "high"
    assert subjects["car"].screen_time_sec == 5.0
    assert subjects["logo"].prominence == "low"
    # Chapters are sorted, clamped to real shots, and non-overlapping.
    assert [(c.title, c.start_sec, c.end_sec) for c in report.chapters] == [
        ("Open", 0.0, 7.0),
        ("Close", 7.0, 12.0),
    ]
    assert report.findings == []


def test_empty_modalities_skip_their_llm_calls():
    summary = _summary(audio_segments=[], text_events=[], scene_descriptions=[])
    stub = _StubExplorer(_stub_outputs())
    report = stub.explore(summary, objects_tracked=False)

    assert stub.calls == [explorer_mod._SynthesisOutput]
    assert report.transcript is None
    assert report.on_screen_text is None
    assert report.visuals is None


def test_no_objects_visuals_use_scene_descriptions_only():
    report = _StubExplorer(_stub_outputs()).explore(_summary(), objects_tracked=False)
    assert report.visuals is not None
    assert report.visuals.subjects == []


def test_failed_digest_drops_only_that_section():
    stub = _StubExplorer(_stub_outputs(), fail={explorer_mod._TranscriptOutput})
    report = stub.explore(_summary(), objects_tracked=False)

    assert report.transcript is None
    assert report.on_screen_text is not None
    assert report.summary == "an ad"


def test_failed_overview_fails_the_report():
    stub = _StubExplorer(_stub_outputs(), fail={explorer_mod._SynthesisOutput})
    with pytest.raises(RuntimeError):
        stub.explore(_summary(), objects_tracked=False)


def test_exploration_result_round_trip_replaces(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'explore.db'}")
    metadata_obj.create_all(engine)
    writer = ClipScribeWriterDB(engine=engine)
    reader = ClipScribeReaderDB(engine=engine)

    first = ExploreReport(
        summary="v1",
        chapters=[],
        transcript=None,
        on_screen_text=None,
        visuals=None,
        metrics=compute_run_metrics(_summary(), objects_tracked=False),
    )
    second = first.model_copy(update={"summary": "v2"})
    writer.save_exploration_result("r1", 1, first.model_dump(mode="json"), "m")
    writer.save_exploration_result("r1", 1, second.model_dump(mode="json"), "m")

    row = reader.get_exploration_result("r1")
    assert row is not None
    assert ExploreReport.model_validate(row["payload"]).summary == "v2"
    assert row["model"] == "m"
    assert reader.get_exploration_result("missing") is None


@pytest.mark.parametrize(
    "prompt",
    [TRANSCRIPT_PROMPT, ON_SCREEN_TEXT_PROMPT, VISUALS_PROMPT, SYNTHESIS_PROMPT],
)
def test_prompts_fill_every_placeholder_and_end_with_shared_rules(prompt):
    text = _PromptLimitsParametrizer(params=_PROMPT_PARAMS).parametrize(
        prompt, _SHARED_RULES
    )

    assert "{" not in text and "}" not in text
    assert text.endswith(_SHARED_RULES)


def test_prompt_limits_come_from_params():
    parametrizer = _PromptLimitsParametrizer(params=_PROMPT_PARAMS)

    transcript = parametrizer.parametrize(TRANSCRIPT_PROMPT, _SHARED_RULES)
    overview = parametrizer.parametrize(SYNTHESIS_PROMPT, _SHARED_RULES)

    assert "up to 3 short topic labels" in transcript
    assert "up to 4 concrete claims" in transcript
    assert "5 sentences" in overview
    assert "into up to 7 consecutive chapters" in overview


def test_list_limits_are_enforced_in_code():
    outputs = _stub_outputs()
    outputs[explorer_mod._TranscriptOutput] = explorer_mod._TranscriptOutput(
        summary="talks",
        topics=["a", "b", "c", "d", "e"],
        key_claims=[
            explorer_mod._ClaimRef(text=f"claim {i}", segment=0) for i in range(6)
        ],
    )
    params = {**_PROMPT_PARAMS, "overview_summary_chapters_num": 1}
    report = _StubExplorer(outputs, params=params).explore(
        _summary(), objects_tracked=False
    )

    # The stub returns more than the limits (3 topics, 4 claims, 1 chapter).
    assert report.transcript.topics == ["a", "b", "c"]
    assert [c.text for c in report.transcript.key_claims] == [
        "claim 0",
        "claim 1",
        "claim 2",
        "claim 3",
    ]
    # Chapters are cut from the end, keeping the earliest.
    assert [c.title for c in report.chapters] == ["Open"]


def test_text_cards_beyond_the_llm_limit_stay_in_the_report():
    # Three cards (Jeep 3s, DRIVE IT HOME 2s, HOLIDAY EVENT 2s); the LLM only
    # cleans the single longest one.
    outputs = _stub_outputs()
    outputs[explorer_mod._OnScreenTextOutput] = explorer_mod._OnScreenTextOutput(
        summary="branding",
        cards=[
            explorer_mod._CardLabel(card=0, clean_text="Jeep", category="brand"),
            # Never sent to the model: must not be applied.
            explorer_mod._CardLabel(card=1, clean_text="ghost", category="url"),
        ],
    )
    stub = _StubExplorer(
        outputs, params={**_PROMPT_PARAMS, "on_screen_text_cards_num": 1}
    )
    report = stub.explore(_summary(), objects_tracked=False)

    assert [c["index"] for c in stub.payloads[explorer_mod._OnScreenTextOutput]] == [0]
    cards = report.on_screen_text.cards
    assert [c.text for c in cards] == ["Jeep", "DRIVE IT HOME", "HOLIDAY EVENT"]
    assert [(c.clean_text, c.category) for c in cards] == [
        ("Jeep", "brand"),
        ("DRIVE IT HOME", "other"),
        ("HOLIDAY EVENT", "other"),
    ]


def test_objects_beyond_the_llm_limit_stay_in_the_report():
    # car (5s) is the longest, then logo (1s); only the top one is rated.
    stub = _StubExplorer(
        _stub_outputs(), params={**_PROMPT_PARAMS, "visuals_subjects_num": 1}
    )
    report = stub.explore(_objects_summary(), objects_tracked=True)

    sent = stub.payloads[explorer_mod._VisualsOutput]["objects"]
    assert [o["label"] for o in sent] == ["car"]
    subjects = {s.label: s.prominence for s in report.visuals.subjects}
    # The stub rates "CAR" high; the unsent logo defaults to low but is kept.
    assert subjects == {"car": "high", "logo": "low"}
