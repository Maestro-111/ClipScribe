from types import SimpleNamespace
from unittest import mock

from clip_scribe.extractor.taxonomy_config import ProfilesPile, TaxonomyProfile


def _generated_profile(video_type: str) -> TaxonomyProfile:
    return TaxonomyProfile(
        video_type=video_type,
        focus_categories=["Main subjects", "Environment"],
        guidance_ratio="Generate 60% main subjects and 40% environment.",
        example_items=["person", "building"],
    )


def test_profiles_pile_has_no_general_profile() -> None:
    profiles = ProfilesPile(model="test-model")

    assert profiles.has_profile("car commercial")
    assert not profiles.has_profile("general")
    assert not profiles.has_profile("general profile")


def test_resolve_or_create_reuses_known_profile_without_agent_call() -> None:
    profiles = ProfilesPile(model="test-model")

    with mock.patch(
        "clip_scribe.extractor.taxonomy_config.Runner.run_sync"
    ) as run_sync:
        resolved_type = profiles.resolve_or_create(
            "  Car   Commercial ", scene_context="A vehicle on a road."
        )

    assert resolved_type == "car commercial"
    run_sync.assert_not_called()


def test_resolve_or_create_generates_unknown_explicit_type_once() -> None:
    profiles = ProfilesPile(model="test-model")
    result = SimpleNamespace(final_output=_generated_profile("ignored agent value"))

    with mock.patch(
        "clip_scribe.extractor.taxonomy_config.Runner.run_sync", return_value=result
    ) as run_sync:
        first = profiles.resolve_or_create(
            " Travel   Vlog ",
            scene_context="A host walks through a city.",
            dino_prompt="person . building .",
        )
        second = profiles.resolve_or_create(
            "travel vlog", scene_context="The host enters a restaurant."
        )

    assert first == "travel vlog"
    assert second == first
    assert run_sync.call_count == 1


def test_resolve_or_create_infers_missing_video_type() -> None:
    profiles = ProfilesPile(model="test-model")
    result = SimpleNamespace(final_output=_generated_profile("Cooking Tutorial"))

    with mock.patch(
        "clip_scribe.extractor.taxonomy_config.Runner.run_sync", return_value=result
    ):
        resolved_type = profiles.resolve_or_create(
            None,
            scene_context="A chef demonstrates a recipe in a kitchen.",
            user_hints=["chef", "pan"],
        )

    assert resolved_type == "cooking tutorial"
    assert profiles.has_profile("cooking tutorial")
