import csv
from collections import Counter
from pathlib import Path
from src.parser.youtube.models import YouTubeFeatureResult

CATEGORY_ORDER = ["Attract", "Brand", "Connect", "Direct"]


class YouTubeReportWriter:
    def __init__(self, report_output_path: Path, scores_output_path: Path):
        self.report_output_path = report_output_path
        self.scores_output_path = scores_output_path

    def write_results(self, results: list[YouTubeFeatureResult]) -> None:
        """
        Write feature evaluation results and ABCD scores to CSV files.

        Produces two files in the same directory:
        - <report_name>_report.csv: per-feature detail rows
        - <report_name>_scores.csv: per-category and total scores

        Args:
            results: List of FeatureResult objects
        """

        self.report_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.scores_output_path.parent.mkdir(parents=True, exist_ok=True)

        self._write_detail_report(self.report_output_path, results)
        self._write_scores(self.scores_output_path, results)

    @staticmethod
    def _write_detail_report(
        output_file: Path, results: list[YouTubeFeatureResult]
    ) -> None:
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

    @staticmethod
    def _write_scores(scores_file: Path, results: list[YouTubeFeatureResult]) -> None:
        total_counts: Counter = Counter()
        passed_counts: Counter = Counter()

        for result in results:
            cat = result.feature_category
            total_counts[cat] += 1
            if result.evaluation:
                passed_counts[cat] += 1

        fieldnames = ["category", "passed", "total", "score"]

        total_passed = 0
        total_all = 0

        with open(scores_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for cat in CATEGORY_ORDER:
                passed = passed_counts.get(cat, 0)
                total = total_counts.get(cat, 0)
                total_passed += passed
                total_all += total
                writer.writerow(
                    {
                        "category": cat,
                        "passed": passed,
                        "total": total,
                        "score": round(passed / total, 2) if total else 0.0,
                    }
                )

            writer.writerow(
                {
                    "category": "Total",
                    "passed": total_passed,
                    "total": total_all,
                    "score": round(total_passed / total_all, 2) if total_all else 0.0,
                }
            )
