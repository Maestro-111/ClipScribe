"""Base evaluator with generic orchestration for platform-specific evaluators."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

from clip_scribe.db import ClipScribeReaderDB
from clip_scribe.parser.abcd.tools import build_tools, TOOL_GROUP_TABLES

from clip_scribe.parser.agent_factory import AgentRunner, AgentBuilder
from clip_scribe.parser.abcd.models import BaseFeatureResult, BaseAgentEvaluation

from clip_scribe.clip_scribe_core.abcd_platform_configs import BasePlatformConf
from clip_scribe.utils.clip_scribe_cancel import (
    CancellationToken,
    JobCanceled,
    NullCancellationToken,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

logger = logging.getLogger("clip_scribe")


class BaseEvaluator(ABC):
    """Base class for platform-specific feature evaluators."""

    def __init__(
        self,
        reader_db: ClipScribeReaderDB,
        model: ChatOpenAI,
        platform_config: BasePlatformConf,
        agentic_eval: type[BaseAgentEvaluation],
        max_parallel_agents: int = 5,
        recursion_limit: int = 25,
        cancel_token: CancellationToken | None = None,
        object_data_available: bool = True,
    ):
        """Set up an evaluator for one platform.

        Args:
            reader_db: Sync reader used by the baseline checks and agent tools.
            model: Chat model the criterion agents run on.
            platform_config: Platform settings (brand name, products, CTAs) that
                fill the criteria placeholders.
            agentic_eval: Result model the agent's JSON answer is parsed into.
            max_parallel_agents: Concurrent LLM-judged criteria.
            recursion_limit: Maximum reasoning steps per agent run.
            cancel_token: Polled between criteria; never cancels when omitted.
            object_data_available: False skips criteria that need object tracking.
        """
        self.reader_db = reader_db
        self.model = model
        self.platform_config = platform_config
        self.agentic_eval = agentic_eval
        self.max_parallel_agents = max_parallel_agents
        self.recursion_limit = recursion_limit
        self.object_data_available = object_data_available
        # Cooperative-cancel token, polled between criteria in evaluate_all.
        self._cancel = cancel_token or NullCancellationToken()

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
        """Judge one criterion with a ReAct agent over its tool group's data.

        Any failure becomes a failed result that carries the error text, so one
        criterion cannot abort the whole run.
        """
        feature_id = feature["id"]
        logger.info(f"ClipScribe Parser: Evaluating agentic feature: {feature_id}")

        tool_group = feature["tool_group"]

        tools = build_tools(self.reader_db, run_id, tool_group)
        agent = AgentBuilder.build_agent(model=self.model, tools=tools)

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

        agent_runner = AgentRunner(agent=agent)

        try:
            agent_result = agent_runner.run_abcd_agent(
                question,
                instructions,
                agentic_eval=self.agentic_eval,
                platform_context=self.platform_context,
                time_scope=time_scope,
                field_context=field_context,
                recursion_limit=self.recursion_limit,
                run_name=feature_id,
            )

            return self.build_feature_result(
                feature=feature,
                video_name=video_name,
                evaluation=agent_result.evaluation,
                llm_prompt=llm_prompt,
                llm_explanation=agent_result.explanation,
            )

        except Exception as e:
            logger.error(
                f"ClipScribe Parser: Error evaluating agentic feature {feature_id}: {str(e)}"
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
        """Check one rule-based criterion (no LLM); a failure counts as not passed."""
        feature_id = feature["id"]
        logger.info(f"ClipScribe Parser: Evaluating baseline feature: {feature_id}")

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
            logger.error(
                f"ClipScribe Parser: Error evaluating baseline feature {feature_id}: {str(e)}"
            )
            return self.build_feature_result(
                feature=feature,
                video_name=video_name,
                evaluation=False,
                llm_prompt=None,
                llm_explanation=None,
            )

    def evaluate_all(self, run_id: str, video_name: str) -> list[BaseFeatureResult]:
        """Evaluate every criterion for a run and return the results.

        Baseline criteria run first, one by one; LLM-judged ones then run in
        parallel (up to ``max_parallel_agents``). Criteria that need object data are
        skipped when objects were not extracted. Raises ``JobCanceled`` if the job is
        canceled between criteria.
        """
        logger.info(
            f"ClipScribe Parser: Starting {self.platform} evaluation for run {run_id}"
        )

        features = self.get_feature_configs()

        if not self.object_data_available:
            skipped_features = [
                feature for feature in features if feature.get("requires_object_data")
            ]
            features = [
                feature
                for feature in features
                if not feature.get("requires_object_data")
            ]
            logger.info(
                "ClipScribe Parser: Skipping %d object-dependent criteria because "
                "object extraction was disabled",
                len(skipped_features),
            )

        baseline_features = [f for f in features if f["mode"] == "baseline"]
        agentic_features = [f for f in features if f["mode"] == "agentic"]

        results: list[BaseFeatureResult] = []
        logger.info(
            f"ClipScribe Parser: Evaluating {len(baseline_features)} baseline features..."
        )

        for feature in baseline_features:
            self._cancel.check()
            result = self._evaluate_baseline_feature(feature, run_id, video_name)
            results.append(result)

        logger.info(
            f"ClipScribe Parser: Evaluating {len(agentic_features)} agentic features in parallel "
            f"(max {self.max_parallel_agents} concurrent)..."
        )

        # Pre-compute field contexts on the main thread to avoid concurrent
        # SQLite access from worker threads.
        unique_tool_groups = {f["tool_group"] for f in agentic_features}
        field_contexts: dict[str, str | None] = {}
        for tg in unique_tool_groups:
            try:
                field_contexts[tg] = self._build_field_context(tg)
            except Exception as e:
                logger.warning(
                    "ClipScribe Parser: Failed to build field context for tool group '%s': %s",
                    tg,
                    e,
                )
                field_contexts[tg] = None

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
                # If a cancel arrived, stop collecting and prevent queued agents
                # from starting. In-flight agents (<= max_parallel_agents) run to
                # completion when the executor's context manager waits on exit —
                # they can't be thread-killed — but no new LLM work is dispatched.
                try:
                    self._cancel.check()
                except JobCanceled:
                    for pending in future_to_feature:
                        pending.cancel()
                    raise

                feature = future_to_feature[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(
                        f"ClipScribe Parser: Error in parallel evaluation of {feature['id']}: {str(e)}"
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

        logger.info(
            f"ClipScribe Parser: Completed evaluation of {len(results)} features for run {run_id}"
        )
        return results
