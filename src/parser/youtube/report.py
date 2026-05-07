import csv
from pathlib import Path
from src.parser.youtube.models import YouTubeFeatureResult


class YouTubeReportWriter:
    def __init__(self, output_path: str):
        self.output_path = output_path

    def write_report(self, results: list[YouTubeFeatureResult]) -> None:
        """
        Write feature evaluation results to CSV file.

        Args:
            results: List of FeatureResult objects

        Creates directory if it doesn't exist and writes CSV with columns:
        - video_name
        - platform
        - feature_category
        - feature_name
        - feature_criteria
        - evaluation
        - llm_prompt
        - llm_explanation
        """

        # Ensure output directory exists

        output_file = Path(self.output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Define CSV columns
        fieldnames = [
            "video_name",
            "platform",
            "feature_category",
            "feature_name",
            "feature_criteria",
            "evaluation",
            "llm_prompt",
            "llm_explanation",
        ]

        # Write CSV
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                writer.writerow(
                    {
                        "video_name": result.video_name,
                        "platform": result.platform,
                        "feature_category": result.feature_category,
                        "feature_name": result.feature_name,
                        "feature_criteria": result.feature_criteria,
                        "evaluation": result.evaluation,
                        "llm_prompt": result.llm_prompt or "",
                        "llm_explanation": result.llm_explanation or "",
                    }
                )
