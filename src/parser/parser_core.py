"""Core parser for video information extraction and evaluation."""

import logging
from pathlib import Path

from src.utils.clib_scribe_db import ClipScribeReaderDB
from src.parser.evaluator_base import BaseEvaluator
from src.parser.youtube import *  # noqa ignore
from src.clip_scribe.platform_configs import BasePlatformConf
from langchain_openai import ChatOpenAI


class VideoInformationParser:
    """Main entry point for parsing and evaluating video metadata."""

    def __init__(
        self,
        reader_db: ClipScribeReaderDB,
        agent: dict,
        platform_name: str,
        platform_config: BasePlatformConf,
        output_dir: str,
        logger: logging.Logger,
        max_parallel_agents: int = 5,
        recursion_limit: int = 20,
    ):
        """
        Initialize the video information parser.

        Args:
            reader_db: Database reader instance
            agent: agent configuration
            platform_config: Platform-specific configuration (brand_name, products, etc.)
            output_dir: Output directory for parser reports
            logger: Logger instance
            max_parallel_agents: Maximum number of parallel agent executions
        """

        self.reader_db = reader_db
        self.platform_name = platform_name
        self.platform_config = platform_config
        self.output_dir = output_dir
        self.logger = logger
        self.max_parallel_agents = max_parallel_agents
        self.recursion_limit = recursion_limit
        self.model = self.create_agent_model(agent)

    def __repr__(self) -> str:
        return f"VideoInformationParser(model={self.model.model_name}, output_dir={self.output_dir})"

    @staticmethod
    def create_agent_model(agent: dict):
        llm = agent.get("llm", "gpt-4o-mini")
        temperature = agent.get("temperature", 0.0)
        max_tokens = agent.get("max_tokens", 1000)
        max_retries = agent.get("max_retries", 7)

        model = ChatOpenAI(
            model=llm,
            temperature=temperature,
            timeout=30,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

        return model

    def create_report_name(self) -> str:
        report_name = ""
        if self.platform_name == "youtube":
            report_name = "abcd"
        return report_name

    def create_evaluator(self) -> BaseEvaluator | None:
        evaluator = None

        if self.platform_name == "youtube":
            evaluator = YouTubeEvaluator(  # noqa ignore
                reader_db=self.reader_db,
                model=self.model,
                platform_config=self.platform_config,
                agentic_eval=YouTubeAgentEvaluation,  # noqa ignore
                logger=self.logger,
                max_parallel_agents=self.max_parallel_agents,
                recursion_limit=self.recursion_limit,
            )

        return evaluator

    def create_report_writer(self, output_path):
        report_writer = None

        if self.platform_name == "youtube":
            report_writer = YouTubeReportWriter(output_path)  # noqa ignore

        return report_writer

    def parse(self, run_id: str, video_name: str) -> str:
        """
        Parse and evaluate video information for all platform criteria.

        Args:
            run_id: Run identifier from database
            video_name: Name of the video

        Returns:
            Path to the generated CSV report
        """

        self.logger.info(f"Starting parser for run {run_id}, video '{video_name}'")

        report_name = self.create_report_name()
        output_path = Path(self.output_dir) / video_name / f"{report_name}_report.csv"

        evaluator = self.create_evaluator()
        report_writer = self.create_report_writer(output_path)

        if evaluator is not None:
            results = evaluator.evaluate_all(run_id, video_name)
            report_writer.write_report(results)

            self.logger.info(f"Parser completed. Report written to: {output_path}")

        return str(output_path)
