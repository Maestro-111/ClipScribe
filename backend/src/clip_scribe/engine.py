from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.utils.ids import new_ulid
from src.utils.clip_scribe_artifacts import ArtifactUploader, run_artifact_dir
from src.utils.clip_scribe_cancel import CancellationToken, JobCanceled
from src.utils.progress import Phase, ProgressEvent, ProgressReporter

if TYPE_CHECKING:
    from src.db import ClipScribeReaderDB, ClipScribeWriterDB
    from src.extractor.extractor_core import (
        ExtractionSummary,
        VideoInformationExtractor,
    )
    from src.parser.parser_core import VideoInformationParser

logger = logging.getLogger("clip_scribe")


class ClipScribeEngine:

    """
    Coordinates extraction, persistence, parsing, and job-level progress events.
    """

    def __init__(
        self,
        mode: str,
        video_name: str,
        video_path: str,
        video_type: str | None,
        extractor: VideoInformationExtractor | None,
        parser: VideoInformationParser | None,
        reader_db: ClipScribeReaderDB,
        writer_db: ClipScribeWriterDB,
        progress_reporter: ProgressReporter,
        artifact_uploader: ArtifactUploader,
        cancel_token: CancellationToken,
    ):
        self.mode = mode

        self.video_name = video_name
        self.video_path = video_path
        self.video_type = video_type

        self.extractor = extractor
        self.parser = parser
        self.writer_db = writer_db
        self.reader_db = reader_db
        self.progress = progress_reporter
        self.artifact_uploader = artifact_uploader
        # Cooperative-cancel token, shared with the extractor/parser. Null (never
        # canceled) for CLI/tests; a Redis-backed token in the web paths.
        self._cancel = cancel_token

        # Populated when the extractor's run is persisted; surfaced on
        # job.completed so live subscribers can navigate to the finished run.
        self.run_id: str = ""

    def __repr__(self) -> str:
        return (
            f"This is ClibScribe - smartest video processor! "
            f"mode : {self.mode}, "
            f"(extractor={self.extractor}, parser={self.parser})"
        )

    def _phases_for_mode(self) -> list[str]:
        """The ordered phase names a UI should expect for the current mode."""
        extract_phases = [
            Phase.SCENE_DETECTION,
            Phase.AUDIO,
            Phase.SHOT_PROCESSING,
            Phase.FINALIZE,
        ]
        if self.mode == "extract":
            return extract_phases
        if self.mode == "parse":
            return [Phase.PARSE]
        return [*extract_phases, Phase.PARSE]

    def run(self, run_id: str = ""):
        # For extract/full the run is new — mint the id up front (ULID) so the
        # extractor keys its artifact dir + raw detections by it before the run
        # row exists. For parse mode the id refers to an existing run, use it.
        if self.mode in ("full", "extract") and not run_id:
            run_id = new_ulid()
        self.run_id = run_id
        self.progress.emit(
            ProgressEvent.JOB_STARTED,
            {
                "video_name": self.video_name,
                "mode": self.mode,
                "phases": self._phases_for_mode(),
            },
        )
        try:
            # Bail before doing any heavy work if a cancel already landed (e.g.
            # the job sat queued and was canceled just as it dequeued).
            self._cancel.check()
            if self.mode == "full":
                # run the whole pipeline, usual UI run
                self.parse_extract()
            elif self.mode == "extract":
                # only the extraction engine, i.e. it won't save run to db
                logger.warning(
                    "ClipScribe is running in extraction mode only"
                    "This mode will only process the video and save the artifacts"
                    "Its meant to run locally only"
                )
                self.extract()
            elif self.mode == "parse":
                # only parse engine, i.e. we expect to have an existing run_id data in db
                logger.warning(
                    "ClipScribe is running in parse mode only"
                    "This mode will only process existing artifacts of run_id"
                    "Its meant to run locally only for agent debugging"
                )
                self.parse(run_id)
            else:
                raise ValueError(
                    f"Invalid mode: {self.mode}. Supported modes: 'full', 'extract', 'parse'"
                )
        except JobCanceled:
            # A cancel is not a failure: emit a distinct terminal event (ends the
            # SSE stream) and re-raise so run_job_core records 'canceled'.
            logger.info("Job canceled; stopping pipeline for run %s", self.run_id)
            self.progress.emit(ProgressEvent.JOB_CANCELED, {"run_id": self.run_id})
            raise
        except Exception as e:
            self.progress.emit(ProgressEvent.JOB_FAILED, {"error": str(e)})
            raise
        else:
            self.progress.emit(ProgressEvent.JOB_COMPLETED, {"run_id": self.run_id})

    def extract(self) -> ExtractionSummary:
        if self.extractor is None:
            raise ValueError("No extractor defined; extract method cannot be called")

        video_metadata = self._run_extractor()

        # Artifacts (mp4, viz PNGs, summary json) are fully written by now;
        # push the run's bundle (no-op unless remote_artifact_write=true).
        self.artifact_uploader.upload_run_artifacts(
            self.run_id, run_artifact_dir(self.run_id)
        )

        return video_metadata

    def parse(self, run_id: str):
        if self.parser is None:
            raise ValueError("No parser defined; parse method cannot be called")

        try:
            if run_id:
                if self.reader_db.get_run(run_id) is None:
                    raise ValueError(f"run_id '{run_id}' not found in database")
                self._run_parser(run_id)
            else:
                raise ValueError("run_id is required for parse mode")

        finally:
            self.writer_db.close()
            self.reader_db.close()

    def parse_extract(self) -> None:
        """

        Run the whole Clib Scribe engine: extract all information from video + parse it

        :return:
        """

        assert self.extractor is not None

        if not getattr(self, "run_id", None):
            raise ValueError(
                "No run_id defined, hence parse_extract was called outside of run!"
            )

        video_metadata = self.extract()

        metadata_descriptions = self.extractor.get_schema_descriptions()
        self._save_metadata_to_db(video_metadata, metadata_descriptions)

        # Extraction is persisted; bail here rather than starting the parser's
        # ~30 LLM criteria if a cancel arrived during extraction.
        self._cancel.check()
        self.parse(self.run_id)

        return

    def _run_extractor(self) -> ExtractionSummary:
        assert self.extractor is not None
        try:
            metadata = self.extractor.extract(
                video_name=self.video_name,
                video_path=self.video_path,
                video_type=self.video_type,
                run_id=self.run_id,
            )
            if metadata is None:
                raise RuntimeError("Extractor returned no metadata")

            logger.info("Extraction finished successfully.")
            return metadata
        except KeyboardInterrupt:
            logger.error("\n!!! Interrupted by User. Saving video... !!!")
            raise
        except JobCanceled:
            # Expected cooperative stop, not an error; cleanup() still runs in
            # the finally below, releasing the capture + video writer.
            logger.info("Extraction canceled; releasing resources.")
            raise
        except Exception as e:
            logger.error(f"_extract_information error occurred: {e}", exc_info=True)
            raise
        finally:
            self.extractor.cleanup()
            logger.info("Extraction Done!")

    def _save_metadata_to_db(
        self,
        video_metadata: ExtractionSummary,
        field_descriptions: dict[str, dict[str, str]],
    ) -> None:
        # run_id is already authoritative on self (set in run()); save_run just
        # persists under it. Nothing to return.
        self.writer_db.save_run(
            run_id=self.run_id,
            video_name=self.video_name,
            video_path=self.video_path,
            video_type=self.video_type,
            video_metadata=video_metadata,
            field_descriptions=field_descriptions,
        )

    def _run_parser(self, run_id: str) -> None:
        """
        Parse and evaluate video information.

        Args:
            run_id: Run identifier from database
        """

        assert self.parser is not None

        report_path = self.parser.parse(
            run_id=run_id,
            reader_db=self.reader_db,
            video_name=self.video_name,
            writer_db=self.writer_db,
        )

        logger.info("Parsing complete!")
        logger.info(f"Parser local report generated: {report_path}")
