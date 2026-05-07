"""Core parser for video information extraction and evaluation."""

import logging
from pathlib import Path

from src.utils.clib_scribe_db import ClipScribeReaderDB
from src.parser.youtube.evaluator import YouTubeEvaluator
from src.parser.report_writer import write_report
from src.clip_scribe.platform_configs import BasePlatformConf


class VideoInformationParser:
    """Main entry point for parsing and evaluating video metadata."""

    def __init__(
        self,
        reader_db: ClipScribeReaderDB,
        model: str,
        platform_name: str,
        platform_config: BasePlatformConf,
        output_dir: str,
        logger: logging.Logger,
        max_parallel_agents: int = 5,
    ):
        """
        Initialize the video information parser.

        Args:
            reader_db: Database reader instance
            model: LLM model name for agentic evaluation
            platform_config: Platform-specific configuration (brand_name, products, etc.)
            output_dir: Output directory for parser reports
            logger: Logger instance
            max_parallel_agents: Maximum number of parallel agent executions
        """
        self.reader_db = reader_db
        self.model = model
        self.platform_name = platform_name
        self.platform_config = platform_config
        self.output_dir = output_dir
        self.logger = logger
        self.max_parallel_agents = max_parallel_agents

    def __repr__(self) -> str:
        return (
            f"VideoInformationParser(model={self.model}, output_dir={self.output_dir})"
        )

    def create_report_name(self) -> str:
        report_name = ""
        if self.platform_name == "youtube":
            report_name = "abcd"
        return report_name

    def create_evaluator(self):
        evaluator = None

        if self.platform_name == "youtube":
            evaluator = YouTubeEvaluator(
                reader_db=self.reader_db,
                model=self.model,
                platform_config=self.platform_config,
                logger=self.logger,
                max_parallel_agents=self.max_parallel_agents,
            )

        return evaluator

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

        # Creat evaluator
        evaluator = self.create_evaluator()
        report_name = self.create_report_name()

        output_path = Path(self.output_dir) / video_name / f"{report_name}_report.csv"

        if evaluator is not None:
            # Evaluate all features
            results = evaluator.evaluate_all(run_id, video_name)
            write_report(results, str(output_path))

            self.logger.info(f"Parser completed. Report written to: {output_path}")

        return str(output_path)
