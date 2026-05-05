class ClipScribeEngine:

    """
    Core
    """

    def __init__(self, extractor, parser, logger, reader_db=None, writer_db=None):
        self.extractor = extractor
        self.parser = parser
        self.logger = logger

        self.writer_db = writer_db
        self.reader_db = reader_db

    def __repr__(self) -> str:
        return (
            f"This is ClibScribe - smartest video processor! "
            f"(extractor={self.extractor}, parser={self.parser})"
        )

    def run(self) -> int | None:
        video_metadata: dict | None = {}

        try:
            video_metadata = self._extract_information()
        except Exception:
            pass

        if video_metadata:
            try:
                self._save_metadata_to_db(video_metadata)
                metadata_descriptions = self.extractor.get_schema_descriptions()
                self._save_field_descriptions(metadata_descriptions)
            except Exception as e:
                self.logger.error(f"Failed to save data: {e}", exc_info=True)
                return 0
            finally:
                # Close DB connections once after all operations are done
                if self.writer_db:
                    self.writer_db.close()
                if self.reader_db:
                    self.reader_db.close()
            try:
                self._parse_information()
            except Exception:
                return 0

        else:
            self.logger.warning("No video metadata to parse")

        return 1

    def _extract_information(self) -> dict | None:
        try:
            metadata = self.extractor.extract()
            self.logger.info("Extraction finished successfully.")
            return metadata
        except KeyboardInterrupt:
            self.logger.error("\n!!! Interrupted by User. Saving video... !!!")
            return None
        except Exception as e:
            self.logger.error(
                f"_extract_information error occurred: {e}", exc_info=True
            )
            raise e
        finally:
            self.extractor.cleanup()
            self.logger.info("Done!")

    def _save_metadata_to_db(self, video_metadata: dict) -> None:
        if self.writer_db is None:
            return

        self.writer_db.save_run(
            video_name=self.extractor.video_name,
            video_path=self.extractor.video_path,
            video_type=self.extractor.video_type,
            video_metadata=video_metadata,
        )

    def _save_field_descriptions(self, descriptions: dict) -> None:
        if self.writer_db is None:
            return

        self.writer_db.save_field_descriptions(descriptions)

    def _parse_information(self):
        pass
