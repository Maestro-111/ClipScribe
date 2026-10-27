"""Core parser for video information extraction and evaluation."""

import logging
import re
from pathlib import Path

from clip_scribe.db import ClipScribeReaderDB, ClipScribeWriterDB
from clip_scribe.parser.abcd.evaluator_base import BaseEvaluator
from clip_scribe.parser.explore.explorer import VideoExplorer
from clip_scribe.parser.explore.models import SummarySentencesNum
from clip_scribe.parser.abcd.youtube import (
    YouTubeAgentEvaluation,
    YouTubeEvaluator,
    YouTubeReportWriter,
)
from clip_scribe.clip_scribe_core.platform_configs import BasePlatformConf
from clip_scribe.utils.clip_scribe_cancel import (
    CancellationToken,
    NullCancellationToken,
)
from clip_scribe.utils.clip_scribe_prompts import _PromptLimitsParametrizer
from clip_scribe.utils.progress import NullProgressReporter, Phase, ProgressReporter
from langchain_openai import ChatOpenAI

logger = logging.getLogger("clip_scribe")


def _safe_path_segment(value: str, fallback: str) -> str:
    """Reduce ``value`` to one filesystem-safe path segment, or ``fallback`` if
    nothing is left.
    """
    name = Path(value).name
    segment = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return segment or fallback


class VideoInformationParser:
    """Main entry point for parsing and evaluating video metadata."""

    def __init__(
        self,
        agent: dict,
        explore_max_tokens: int,
        explore_transcript_topics_num: int,
        explore_transcript_key_claims_num: int,
        explore_overview_summary_chapters_num: int,
        explore_summary_sentences_num: SummarySentencesNum,
        explore_on_screen_text_cards_num: int,
        explore_visuals_subjects_num: int,
        platform_name: str | None,
        platform_config: BasePlatformConf | None,
        output_dir: str,
        max_parallel_agents: int = 5,
        recursion_limit: int = 20,
        progress_reporter: ProgressReporter | None = None,
        cancel_token: CancellationToken | None = None,
        extraction_mode: str = "with_objects",
    ):
        """
        Initialize the video information parser.

        Args:
            agent: agent configuration
            platform_name: Platform key used to select the evaluator/report writer
            platform_config: Platform-specific configuration (brand_name, products, etc.)
            output_dir: Output directory for parser reports
            max_parallel_agents: Maximum number of parallel agent executions
            recursion_limit: Maximum LangGraph reasoning steps per agent run
            progress_reporter: Optional progress event sink; defaults to no-op
            explore_max_tokens: Output budget for each exploratory-report call
                (they write longer structured output than an ABCD verdict)
        """

        self.platform_name = platform_name
        self.platform_config = platform_config
        self.output_dir = output_dir
        self.max_parallel_agents = max_parallel_agents
        self.recursion_limit = recursion_limit
        self.model = self.create_agent_model(agent)
        self.explore_model = self.model.model_copy(
            update={"max_tokens": explore_max_tokens}
        )
        self.progress = progress_reporter or NullProgressReporter()
        self.object_data_available = extraction_mode == "with_objects"
        # Cooperative-cancel token, passed on to the evaluator so it can stop
        # between criteria. Null (never canceled) for CLI/tests.
        self._cancel = cancel_token or NullCancellationToken()
        self.explorer_prompt_parametrizer = _PromptLimitsParametrizer(
            params={
                "transcript_topics_num": explore_transcript_topics_num,
                "transcript_key_claims_num": explore_transcript_key_claims_num,
                "overview_summary_chapters_num": explore_overview_summary_chapters_num,
                "on_screen_text_cards_num": explore_on_screen_text_cards_num,
                "visuals_subjects_num": explore_visuals_subjects_num,
                # Spelled out: mypy can't unpack a TypedDict into dict[str, int].
                "transcript_summary_sentences_num": explore_summary_sentences_num[
                    "transcript_summary_sentences_num"
                ],
                "text_summary_sentences_num": explore_summary_sentences_num[
                    "text_summary_sentences_num"
                ],
                "visual_summary_sentences_num": explore_summary_sentences_num[
                    "visual_summary_sentences_num"
                ],
                "overview_summary_sentences_num": explore_summary_sentences_num[
                    "overview_summary_sentences_num"
                ],
            }
        )

    def __repr__(self) -> str:
        """Model and output directory, for logs."""
        return f"VideoInformationParser(model={self.model.model_name}, output_dir={self.output_dir})"

    @staticmethod
    def create_agent_model(agent: dict) -> ChatOpenAI:
        """Build the chat model from the parser ``agent`` config (llm, temperature,
        max_tokens, max_retries).
        """
        llm = agent.get("llm", "gpt-5.4")
        temperature = agent.get("temperature", 0.0)
        max_tokens = agent.get("max_tokens", 1000)
        max_retries = agent.get("max_retries", 7)

        model = ChatOpenAI(
            model=llm,
            temperature=temperature,
            timeout=30,
            max_completion_tokens=max_tokens,
            max_retries=max_retries,
        )

        return model

    def create_report_name(self) -> str:
        """Base name of the report files (``abcd`` for YouTube, empty otherwise)."""
        report_name = ""
        if self.platform_name == "youtube":
            report_name = "abcd"
        return report_name

    def create_evaluator(self, reader_db) -> BaseEvaluator | None:
        """The platform's ABCD evaluator, or ``None`` if the platform has none."""
        evaluator = None

        if self.platform_name == "youtube" and self.platform_config is not None:
            evaluator = YouTubeEvaluator(  # noqa ignore
                reader_db=reader_db,
                model=self.model,
                platform_config=self.platform_config,
                agentic_eval=YouTubeAgentEvaluation,  # noqa ignore
                max_parallel_agents=self.max_parallel_agents,
                recursion_limit=self.recursion_limit,
                cancel_token=self._cancel,
                object_data_available=self.object_data_available,
            )

        return evaluator

    def create_report_writer(self, output_path: Path, scores_output_path: Path):
        """The platform's CSV report writer, or ``None`` if the platform has none."""
        report_writer = None

        if self.platform_name == "youtube":
            report_writer = YouTubeReportWriter(output_path, scores_output_path)  # noqa ignore

        return report_writer

    def explore(
        self,
        run_id: str,
        video_name: str,
        reader_db: ClipScribeReaderDB,
        writer_db: ClipScribeWriterDB,
    ) -> None:
        """Build and persist the exploratory report for a persisted run.

        Findings (stage C) are not generated yet; the report's ``findings``
        stays empty.
        """

        logger.info(
            f"ClipScribe Parser: Starting exploration parser for run {run_id}, video '{video_name}'"
        )

        self._cancel.check()
        self.progress.phase_started(Phase.PARSE)

        summary = reader_db.get_extraction_summary(run_id)

        if summary is None:
            raise ValueError(f"run_id '{run_id}' not found in database")

        explorer = VideoExplorer(
            self.explore_model,
            prompt_parametrizer=self.explorer_prompt_parametrizer,
            cancel_token=self._cancel,
        )
        report = explorer.explore(summary, objects_tracked=self.object_data_available)

        writer_db.save_exploration_result(
            run_id,
            report.schema_version,
            report.model_dump(mode="json"),
            model=self.explore_model.model_name,
        )

        logger.info(
            "ClipScribe Parser: Exploration completed for run %s: %d chapters",
            run_id,
            len(report.chapters),
        )

        self.progress.phase_completed(Phase.PARSE, {"chapters": len(report.chapters)})

    def abcd(
        self,
        run_id: str,
        video_name: str,
        reader_db: ClipScribeReaderDB,
        writer_db: ClipScribeWriterDB,
    ) -> str:
        """
        Parse and evaluate video information for all platform ABCD criteria.

        Args:
            run_id: Run identifier from database
            video_name: Name of the video
            reader_db: connection to reader database
            writer_db: writer used to persist per-criterion results to the
                abcd_results table

        Returns:
            Path to the generated CSV report
        """

        if self.platform_name is None or self.platform_config is None:
            raise ValueError("ABCD parsing requires a platform and platform config")

        logger.info(
            f"ClipScribe Parser: Starting ABCD parser for run {run_id}, video '{video_name}'"
        )

        report_name = self.create_report_name()

        report_dir = Path(self.output_dir) / _safe_path_segment(run_id, "run")
        report_output_path = report_dir / f"{report_name}_report.csv"
        scores_output_path = report_dir / f"{report_name}_scores.csv"

        evaluator = self.create_evaluator(reader_db)
        report_writer = self.create_report_writer(
            report_output_path, scores_output_path
        )

        # Last cheap bail before the LLM-heavy evaluation begins.
        self._cancel.check()
        self.progress.phase_started(Phase.PARSE)

        if evaluator is not None:
            results = evaluator.evaluate_all(run_id, video_name)
            report_writer.write_results(results)
            writer_db.save_abcd_results(run_id, self.platform_name, results)
            logger.info(
                f"ClipScribe Parser: ABCD Parser completed. Report: {report_output_path}, Scores: {scores_output_path}"
            )
            self.progress.phase_completed(Phase.PARSE, {"criteria_count": len(results)})
        else:
            self.progress.phase_completed(Phase.PARSE, {"criteria_count": 0})

        return str(report_output_path)
