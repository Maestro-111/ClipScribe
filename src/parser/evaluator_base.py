"""Base evaluator with generic orchestration for platform-specific evaluators."""

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.clib_scribe_db import ClipScribeReaderDB
from src.parser.tools import build_tools, TOOL_GROUP_TABLES
from src.parser.agent import build_agent, run_agent
from src.parser.models import BaseFeatureResult, BaseAgentEvaluation

from src.clip_scribe.platform_configs import BasePlatformConf


class BaseEvaluator(ABC):
    """Base class for platform-specific feature evaluators."""

    def __init__(
        self,
        reader_db: ClipScribeReaderDB,
        model,
        platform_config: BasePlatformConf,
        agentic_eval: type[BaseAgentEvaluation],
        logger: logging.Logger,
        max_parallel_agents: int = 5,
        recursion_limit: int = 25,
    ):
        self.reader_db = reader_db
        self.model = model
        self.platform_config = platform_config
        self.agentic_eval = agentic_eval
        self.logger = logger
        self.max_parallel_agents = max_parallel_agents
        self.recursion_limit = recursion_limit

    @property
    @abstractmethod
    def platform(self) -> str:
        """Return the platform identifier (e.g. 'youtube')."""
        ...

    @property
    def platform_context(self) -> str:
        """Return context string for the agent system prompt. Override per platform."""
        return "video criteria"

    @abstractmethod
    def get_feature_configs(self) -> list[dict]:
        """Return the list of feature configuration dicts for this platform."""
        ...

    @abstractmethod
    def evaluate_baseline(self, feature_id: str, run_id: str) -> bool:
        """Evaluate a single baseline feature. Implemented per platform."""
        ...

    @abstractmethod
    def _resolve_placeholders(self, text: str, **extra) -> str:
        """Resolve template placeholders in text using platform_config."""
        ...

    @abstractmethod
    def build_feature_result(
        self,
        feature: dict,
        video_name: str,
        evaluation: bool,
        llm_prompt: str | None,
        llm_explanation: str | None,
    ) -> BaseFeatureResult:
        """Build a platform-specific feature result from a feature config and evaluation output.

        Args:
            feature: The feature config dict from get_feature_configs()
            video_name: Name of the video
            evaluation: Boolean evaluation result
            llm_prompt: The prompt sent to the LLM (None for baseline)
            llm_explanation: The LLM's explanation (None for baseline)
        """
        ...

    def _build_field_context(self, tool_group: str) -> str | None:
        """Fetch field descriptions for the tables relevant to a tool_group
        and format them into a readable reference string for the agent."""
        table_names = TOOL_GROUP_TABLES.get(tool_group, [])

        if not table_names:
            return None

        all_descriptions: list[dict] = []
        for table_name in table_names:
            all_descriptions.extend(self.reader_db.get_field_descriptions(table_name))

        if not all_descriptions:
            return None

        # Group by table for readable formatting
        by_table: dict[str, list[dict]] = {}
        for desc in all_descriptions:
            by_table.setdefault(desc["table_name"], []).append(desc)

        lines: list[str] = []
        for table_name, columns in by_table.items():
            lines.append(f"\nTable: {table_name}")
            for col in columns:
                lines.append(f"  - {col['column_name']}: {col['description']}")

        return "\n".join(lines)

    def _evaluate_agentic_feature(
        self,
        feature: dict,
        run_id: str,
        video_name: str,
        field_context: str | None = None,
    ) -> BaseFeatureResult:
        feature_id = feature["id"]
        self.logger.info(f"Evaluating agentic feature: {feature_id}")

        tool_group = feature["tool_group"]
        tools = build_tools(self.reader_db, run_id, tool_group)
        agent = build_agent(self.model, tools)

        criteria = feature.get("criteria", "")
        question = self._resolve_placeholders(feature["question"], criteria=criteria)
        instructions_list = feature.get("instructions", [])
        instructions = "\n".join(
            [
                self._resolve_placeholders(instr, criteria=criteria)
                for instr in instructions_list
            ]
        )
        llm_prompt = f"{question}\n\n{instructions}"

        time_scope = 5.0 if feature.get("type") == "first_5_secs_video" else None

        try:
            agent_result = run_agent(
                agent,
                question,
                instructions,
                agentic_eval=self.agentic_eval,
                platform_context=self.platform_context,
                time_scope=time_scope,
                field_context=field_context,
                recursion_limit=self.recursion_limit,
            )

            return self.build_feature_result(
                feature=feature,
                video_name=video_name,
                evaluation=agent_result.evaluation,
                llm_prompt=llm_prompt,
                llm_explanation=agent_result.explanation,
            )

        except Exception as e:
            self.logger.error(
                f"Error evaluating agentic feature {feature_id}: {str(e)}"
            )
            return self.build_feature_result(
                feature=feature,
                video_name=video_name,
                evaluation=False,
                llm_prompt=llm_prompt,
                llm_explanation=f"Error during evaluation: {str(e)}",
            )

    def _evaluate_baseline_feature(
        self, feature: dict, run_id: str, video_name: str
    ) -> BaseFeatureResult:
        feature_id = feature["id"]
        self.logger.info(f"Evaluating baseline feature: {feature_id}")

        try:
            evaluation = self.evaluate_baseline(feature_id, run_id)

            return self.build_feature_result(
                feature=feature,
                video_name=video_name,
                evaluation=evaluation,
                llm_prompt=None,
                llm_explanation=None,
            )

        except Exception as e:
            self.logger.error(
                f"Error evaluating baseline feature {feature_id}: {str(e)}"
            )
            return self.build_feature_result(
                feature=feature,
                video_name=video_name,
                evaluation=False,
                llm_prompt=None,
                llm_explanation=None,
            )

    def evaluate_all(self, run_id: str, video_name: str) -> list[BaseFeatureResult]:
        self.logger.info(f"Starting {self.platform} evaluation for run {run_id}")

        features = self.get_feature_configs()

        baseline_features = [f for f in features if f["mode"] == "baseline"]
        agentic_features = [f for f in features if f["mode"] == "agentic"]

        results: list[BaseFeatureResult] = []
        self.logger.info(f"Evaluating {len(baseline_features)} baseline features...")

        for feature in baseline_features:
            result = self._evaluate_baseline_feature(feature, run_id, video_name)
            results.append(result)

        self.logger.info(
            f"Evaluating {len(agentic_features)} agentic features in parallel "
            f"(max {self.max_parallel_agents} concurrent)..."
        )

        # Pre-compute field contexts on the main thread to avoid concurrent
        # SQLite access from worker threads.
        unique_tool_groups = {f["tool_group"] for f in agentic_features}
        field_contexts: dict[str, str | None] = {
            tg: self._build_field_context(tg) for tg in unique_tool_groups
        }

        with ThreadPoolExecutor(max_workers=self.max_parallel_agents) as executor:
            future_to_feature = {
                executor.submit(
                    self._evaluate_agentic_feature,
                    feature,
                    run_id,
                    video_name,
                    field_contexts.get(feature["tool_group"]),
                ): feature
                for feature in agentic_features
            }

            for future in as_completed(future_to_feature):
                feature = future_to_feature[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(
                        f"Error in parallel evaluation of {feature['id']}: {str(e)}"
                    )
                    results.append(
                        self.build_feature_result(
                            feature=feature,
                            video_name=video_name,
                            evaluation=False,
                            llm_prompt=None,
                            llm_explanation=f"Parallel execution error: {str(e)}",
                        )
                    )

        self.logger.info(
            f"Completed evaluation of {len(results)} features for run {run_id}"
        )
        return results
