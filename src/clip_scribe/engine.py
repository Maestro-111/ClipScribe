class ClipScribeEngine:

    """
    Core
    """

    def __init__(self, extractor, parser, logger):
        self.extractor = extractor
        self.parser = parser
        self.logger = logger

    def __repr__(self) -> str:
        return (
            f"This is ClibScribe - smartest video processor! "
            f"(extractor={self.extractor}, parser={self.parser})"
        )

    def run(self) -> None:
        video_metadata: dict | None = {}
        try:
            video_metadata = self._extract_information()
        except Exception as e:
            self.logger.error(e)

        if video_metadata:
            self._parse_information(video_metadata)
        else:
            self.logger.warning("No video metadata to parse")

        return None

    def _extract_information(self) -> dict | None:
        try:
            metadata = self.extractor.extract()
            self.logger.info("Extraction finished successfully.")
            return metadata
        except KeyboardInterrupt:
            self.logger.error("\n!!! Interrupted by User. Saving video... !!!")
            return None
        except Exception as e:
            # exc_info=True automatically captures and formats the traceback
            self.logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            raise e
        finally:
            self.extractor.cleanup()
            self.logger.info("Done!")

    def _parse_information(self, video_metadata):
        try:
            parsed_metadata = self.parser.parse(video_metadata)
            self.logger.info("Extraction finished successfully.")
            return parsed_metadata
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            raise e
        finally:
            self.logger.info("Done!")
