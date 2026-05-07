"""YouTube ABCD criteria evaluator orchestrator."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import ChatOpenAI

from src.utils.clib_scribe_db import ClipScribeReaderDB
from src.parser.models import FeatureResult
from src.parser.youtube.criteria import get_feature_configs
from src.parser.youtube.baseline import evaluate_baseline
from src.parser.youtube.tools import build_tools
from src.parser.youtube.agent import build_agent, run_agent

from src.clip_scribe.platform_configs import BasePlatformConf


class YouTubeEvaluator:
    """Evaluates all YouTube ABCD features for a video run."""

    def __init__(
        self,
        reader_db: ClipScribeReaderDB,
        model: str,
        platform_config: BasePlatformConf,
        logger: logging.Logger,
        max_parallel_agents: int = 5,
    ):
        """
        Initialize the YouTube evaluator.

        Args:
            reader_db: Database reader instance
            model: LLM model name (e.g., "gpt-4o-mini")
            platform_config: Platform-specific config (brand_name, products, etc.)
            logger: Logger instance
            max_parallel_agents: Maximum number of parallel agent executions
        """
        self.reader_db = reader_db
        self.model_name = model
        self.platform_config = platform_config
        self.logger = logger
        self.max_parallel_agents = max_parallel_agents

        # Initialize ChatOpenAI model
        self.llm = ChatOpenAI(model=model, temperature=0)

    def _resolve_placeholders(self, text: str) -> str:
        """
        Resolve template placeholders in text using platform_config.

        Args:
            text: Text with placeholders like {brand_name}

        Returns:
            Text with placeholders replaced
        """
        # Handle list fields by joining with commas
        branded_products = ", ".join(self.platform_config.get("branded_products", []))
        branded_products_categories = ", ".join(
            self.platform_config.get("branded_products_categories", [])
        )
        call_to_actions = ", ".join(self.platform_config.get("call_to_actions", []))

        return text.format(
            brand_name=self.platform_config.get("brand_name", ""),
            branded_products=branded_products,
            branded_products_categories=branded_products_categories,
            call_to_actions=call_to_actions,
        )

    def _evaluate_agentic_feature(
        self, feature: dict, run_id: str, video_name: str
    ) -> FeatureResult:
        """
        Evaluate a single agentic feature using LangGraph agent.

        Args:
            feature: Feature configuration dict
            run_id: Run identifier
            video_name: Video name

        Returns:
            FeatureResult with evaluation and explanation
        """
        feature_id = feature["id"]
        self.logger.info(f"Evaluating agentic feature: {feature_id}")

        # Build tools for this feature
        tool_group = feature["tool_group"]
        tools = build_tools(self.reader_db, run_id, tool_group)

        # Build agent
        agent = build_agent(self.llm, tools)

        # Resolve placeholders in question and instructions
        question = self._resolve_placeholders(feature["question"])
        instructions_list = feature.get("instructions", [])
        instructions = "\n".join(
            [self._resolve_placeholders(instr) for instr in instructions_list]
        )

        # Run agent
        try:
            agent_result = run_agent(agent, question, instructions)

            return FeatureResult(
                video_name=video_name,
                platform="youtube",
                feature_category=feature["category"],
                feature_name=feature["name"],
                feature_criteria=self._resolve_placeholders(feature["criteria"]),
                evaluation=agent_result.evaluation,
                llm_prompt=f"{question}\n\n{instructions}",
                llm_explanation=agent_result.explanation,
            )

        except Exception as e:
            self.logger.error(
                f"Error evaluating agentic feature {feature_id}: {str(e)}"
            )
            return FeatureResult(
                video_name=video_name,
                platform="youtube",
                feature_category=feature["category"],
                feature_name=feature["name"],
                feature_criteria=self._resolve_placeholders(feature["criteria"]),
                evaluation=False,
                llm_prompt=f"{question}\n\n{instructions}",
                llm_explanation=f"Error during evaluation: {str(e)}",
            )

    def _evaluate_baseline_feature(
        self, feature: dict, run_id: str, video_name: str
    ) -> FeatureResult:
        """
        Evaluate a single baseline feature using deterministic query.

        Args:
            feature: Feature configuration dict
            run_id: Run identifier
            video_name: Video name

        Returns:
            FeatureResult with evaluation (no LLM prompt/explanation)
        """
        feature_id = feature["id"]
        self.logger.info(f"Evaluating baseline feature: {feature_id}")

        try:
            evaluation = evaluate_baseline(feature_id, self.reader_db, run_id)

            return FeatureResult(
                video_name=video_name,
                platform="youtube",
                feature_category=feature["category"],
                feature_name=feature["name"],
                feature_criteria=self._resolve_placeholders(feature["criteria"]),
                evaluation=evaluation,
                llm_prompt=None,
                llm_explanation=None,
            )

        except Exception as e:
            self.logger.error(
                f"Error evaluating baseline feature {feature_id}: {str(e)}"
            )
            return FeatureResult(
                video_name=video_name,
                platform="youtube",
                feature_category=feature["category"],
                feature_name=feature["name"],
                feature_criteria=self._resolve_placeholders(feature["criteria"]),
                evaluation=False,
                llm_prompt=None,
                llm_explanation=None,
            )

    def evaluate_all(self, run_id: str, video_name: str) -> list[FeatureResult]:
        """
        Evaluate all YouTube ABCD features for a video run.

        Args:
            run_id: Run identifier
            video_name: Video name

        Returns:
            List of FeatureResult for all 23 features
        """

        self.logger.info(f"Starting YouTube ABCD evaluation for run {run_id}")

        # Load feature configs
        features = get_feature_configs()

        # Separate baseline and agentic features
        baseline_features = [f for f in features if f["mode"] == "baseline"]
        agentic_features = [f for f in features if f["mode"] == "agentic"]

        results = []

        # Evaluate baseline features sequentially (they're fast)
        self.logger.info(f"Evaluating {len(baseline_features)} baseline features...")
        for feature in baseline_features:
            result = self._evaluate_baseline_feature(feature, run_id, video_name)
            results.append(result)

        # Evaluate agentic features in parallel
        self.logger.info(
            f"Evaluating {len(agentic_features)} agentic features in parallel (max {self.max_parallel_agents} concurrent)..."
        )

        with ThreadPoolExecutor(max_workers=self.max_parallel_agents) as executor:
            # Submit all agentic feature evaluations
            future_to_feature = {
                executor.submit(
                    self._evaluate_agentic_feature, feature, run_id, video_name
                ): feature
                for feature in agentic_features
            }

            # Collect results as they complete
            for future in as_completed(future_to_feature):
                feature = future_to_feature[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(
                        f"Error in parallel evaluation of {feature['id']}: {str(e)}"
                    )
                    # Add a failed result
                    results.append(
                        FeatureResult(
                            video_name=video_name,
                            platform="youtube",
                            feature_category=feature["category"],
                            feature_name=feature["name"],
                            feature_criteria=self._resolve_placeholders(
                                feature["criteria"]
                            ),
                            evaluation=False,
                            llm_prompt=None,
                            llm_explanation=f"Parallel execution error: {str(e)}",
                        )
                    )

        self.logger.info(
            f"Completed evaluation of {len(results)} features for run {run_id}"
        )
        return results
