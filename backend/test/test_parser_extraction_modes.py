"""Parser behavior for extraction modes with and without object data."""

from unittest import mock

from clip_scribe.parser.abcd.evaluator_base import BaseEvaluator
from clip_scribe.parser.abcd.youtube.criteria import get_feature_configs


class _Evaluator(BaseEvaluator):
    @property
    def platform(self) -> str:
        return "test"

    def get_feature_configs(self) -> list[dict]:
        return [
            {
                "id": "baseline",
                "mode": "baseline",
                "tool_group": None,
            },
            {
                "id": "audio",
                "mode": "agentic",
                "tool_group": "audio",
            },
            {
                "id": "objects",
                "mode": "agentic",
                "tool_group": "visual",
                "requires_object_data": True,
            },
        ]

    def evaluate_baseline(self, feature_id: str, run_id: str) -> bool:
        return True

    def _resolve_placeholders(self, text: str, **extra) -> str:
        return text

    def build_feature_result(
        self,
        feature: dict,
        video_name: str,
        evaluation: bool,
        llm_prompt: str | None,
        llm_explanation: str | None,
    ) -> str:
        return feature["id"]

    def _evaluate_agentic_feature(
        self,
        feature: dict,
        run_id: str,
        video_name: str,
        field_context: str | None = None,
    ) -> str:
        return feature["id"]


def test_no_objects_mode_skips_only_object_dependent_features():
    reader = mock.MagicMock()
    reader.get_field_descriptions.return_value = []
    evaluator = _Evaluator(
        reader_db=reader,
        model=mock.MagicMock(),
        platform_config=mock.MagicMock(),
        agentic_eval=mock.MagicMock(),
        object_data_available=False,
    )

    assert set(evaluator.evaluate_all("run-id", "video.mp4")) == {
        "baseline",
        "audio",
    }


def test_youtube_object_questions_are_marked_as_object_dependent():
    marked_ids = {
        feature["id"]
        for feature in get_feature_configs()
        if feature.get("requires_object_data")
    }

    assert marked_ids == {
        "b_brand_visuals",
        "b_brand_visuals_1st_5_secs",
        "b_product_visuals",
        "b_product_visuals_1st_5_secs",
        "c_presence_of_people",
        "c_presence_of_people_1st_5_secs",
        "c_visible_face",
        "c_visible_face_close_up",
    }
