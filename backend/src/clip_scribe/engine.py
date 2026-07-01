from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.utils.ids import new_ulid
from src.utils.artifacts import ArtifactUploader, run_artifact_dir
from src.utils.progress import Phase, ProgressEvent, ProgressReporter

if TYPE_CHECKING:
    from src.db import ClipScribeReaderDB, ClipScribeWriterDB
    from src.extractor.extractor_core import ExtractionSummary, VideoInformationExtractor
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
            if self.mode == "full":
                # run the whole pipeline
                self.parse_extract()
            elif self.mode == "extract":
                # only the extraction engine, i.e. it won't save run to db
                self.extract()
            elif self.mode == "parse":
                # onyl parse engine, i.e. we expect to have an existing run_id data in db
                self.parse(run_id)
            else:
                raise ValueError(
                    f"Invalid mode: {self.mode}. Supported modes: 'full', 'extract', 'parse'"
                )
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

        video_metadata = self.extract()

        metadata_descriptions = self.extractor.get_schema_descriptions()
        self._save_metadata_to_db(video_metadata, metadata_descriptions)
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
        except Exception as e:
            logger.error(f"_extract_information error occurred: {e}", exc_info=True)
            raise
        finally:
            self.extractor.cleanup()
            logger.info("Done!")

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

        logger.info(f"Parser report generated: {report_path}")
