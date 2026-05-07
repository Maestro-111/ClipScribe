"""YouTube ABCD criteria evaluator - thin subclass of BaseEvaluator."""

from src.parser.evaluator_base import BaseEvaluator
from src.parser.youtube.criteria import get_feature_configs
from src.parser.youtube.baseline import evaluate_baseline as _evaluate_baseline
from src.parser.youtube.models import YouTubeFeatureResult


class YouTubeEvaluator(BaseEvaluator):
    """Evaluates all YouTube ABCD features for a video run."""

    @property
    def platform(self) -> str:
        return "youtube"

    @property
    def platform_context(self) -> str:
        return "YouTube ABCD criteria"

    def get_feature_configs(self) -> list[dict]:
        return get_feature_configs()

    def evaluate_baseline(self, feature_id: str, run_id: str) -> bool:
        return _evaluate_baseline(feature_id, self.reader_db, run_id)

    def _resolve_placeholders(self, text: str, **extra) -> str:
        config_data = self.platform_config.model_dump()

        branded_products = ", ".join(config_data.get("branded_products", []))
        branded_products_categories = ", ".join(
            config_data.get("branded_products_categories", [])
        )
        call_to_actions = ", ".join(config_data.get("call_to_actions", []))

        return text.format(
            brand_name=config_data.get("brand_name", ""),
            branded_products=branded_products,
            branded_products_categories=branded_products_categories,
            call_to_actions=call_to_actions,
            **extra,
        )

    def build_feature_result(
        self,
        feature: dict,
        video_name: str,
        evaluation: bool,
        llm_prompt: str | None,
        llm_explanation: str | None,
    ) -> YouTubeFeatureResult:
        return YouTubeFeatureResult(
            video_name=video_name,
            platform=self.platform,
            feature_category=feature["category"],
            feature_name=feature["name"],
            feature_criteria=self._resolve_placeholders(feature["criteria"]),
            evaluation=evaluation,
            llm_prompt=llm_prompt,
            llm_explanation=llm_explanation,
        )
