from src.extractor.extractor_core import ExtractionSummary, VideoInformationExtractor
from src.parser.parser_core import VideoInformationParser
from src.utils.progress import (
    NullProgressReporter,
    Phase,
    ProgressEvent,
    ProgressReporter,
)


class ClipScribeEngine:

    """
    Core
    """

    def __init__(
        self,
        mode: str,
        logger,
        video_name: str,
        video_path: str,
        video_type: str | None,
        extractor: VideoInformationExtractor | None,
        parser: VideoInformationParser | None,
        reader_db=None,
        writer_db=None,
        progress_reporter: ProgressReporter | None = None,
    ):
        self.mode = mode

        self.video_name = video_name
        self.video_path = video_path
        self.video_type = video_type

        self.extractor = extractor
        self.parser = parser
        self.logger = logger
        self.writer_db = writer_db
        self.reader_db = reader_db
        self.progress = progress_reporter or NullProgressReporter()

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
                self.parse_extract()
            elif self.mode == "extract":
                self.extract()
            elif self.mode == "parse":
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

    def extract(self) -> ExtractionSummary | None:
        if self.extractor is None:
            raise ValueError("No extractor defined; extract method cannot be called")

        video_metadata: ExtractionSummary | None = None

        try:
            video_metadata = self._run_extractor()
        except Exception:  # can be None, will skip later
            pass

        return video_metadata

    def parse(self, run_id: str):
        if self.parser is None:
            raise ValueError("No parser defined; parse method cannot be called")

        try:
            if run_id:
                if self.reader_db is None:
                    raise Exception("parse called without a reader_db")
                if self.reader_db.get_run(run_id) is None:
                    raise ValueError(f"run_id '{run_id}' not found in database")
                self._run_parser(run_id)
            else:
                self.logger.warning("Empty run_id is given to parse!")

        except Exception as e:
            raise e

        finally:
            if self.writer_db:
                self.writer_db.close()
            if self.reader_db:
                self.reader_db.close()

    def parse_extract(self) -> None:
        """

        Run the whole Clib Scribe engine: extract all information from video + parse it

        :return:
        """

        assert self.extractor is not None

        video_metadata = self.extract()

        if video_metadata:
            metadata_descriptions = self.extractor.get_schema_descriptions()
            run_id = self._save_metadata_to_db(video_metadata, metadata_descriptions)
            self.run_id = run_id
            self.parse(run_id)

        else:
            self.logger.warning("No video metadata to parse")

        return

    def _run_extractor(self) -> ExtractionSummary | None:
        assert self.extractor is not None
        try:
            metadata = self.extractor.extract(
                video_name=self.video_name,
                video_path=self.video_path,
                video_type=self.video_type,
            )

            self.logger.info("Extraction finished successfully.")
            return metadata
        except KeyboardInterrupt:
            self.logger.error("\n!!! Interrupted by User. Saving video... !!!")
            return None
        except Exception as e:
            self.logger.error(
                f"_extract_information error occurred: {e}", exc_info=True
            )
            return None
        finally:
            self.extractor.cleanup()
            self.logger.info("Done!")

    def _save_metadata_to_db(
        self,
        video_metadata: ExtractionSummary,
        field_descriptions: dict[str, dict[str, str]],
    ) -> str:
        if self.writer_db is None:
            raise Exception("_save_metadata_to_db called without a writer_db")

        run_id = self.writer_db.save_run(
            video_name=self.video_name,
            video_path=self.video_path,
            video_type=self.video_type,
            video_metadata=video_metadata,
            field_descriptions=field_descriptions,
        )
        return run_id

    def _run_parser(self, run_id: str) -> None:
        """
        Parse and evaluate video information.

        Args:
            run_id: Run identifier from database
        """

        assert self.reader_db is not None
        assert self.parser is not None

        report_path = self.parser.parse(
            run_id=run_id, reader_db=self.reader_db, video_name=self.video_name
        )

        self.logger.info(f"Parser report generated: {report_path}")
