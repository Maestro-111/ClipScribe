"""Baseline (deterministic) feature evaluators for YouTube ABCD criteria."""

from src.utils.clib_scribe_db import ClipScribeReaderDB


def evaluate_baseline(
    feature_id: str, reader_db: ClipScribeReaderDB, run_id: str
) -> bool:
    """
    Evaluate a baseline feature using deterministic DB queries.

    Args:
        feature_id: The feature identifier (e.g., "a_dynamic_start")
        reader_db: Database reader instance
        run_id: Run identifier for the video

    Returns:
        Boolean evaluation result
    """
    match feature_id:
        case "a_dynamic_start":
            stats = reader_db.get_global_stats(run_id)
            if not stats:
                return False
            return bool(stats.get("dynamic_start_detected", 0))

        case "a_quick_pacing":
            stats = reader_db.get_global_stats(run_id)
            if not stats:
                return False
            return bool(stats.get("qp_general_detected", 0))

        case "a_quick_pacing_1st_secs":
            stats = reader_db.get_global_stats(run_id)
            if not stats:
                return False
            return bool(stats.get("qp_intro_detected", 0))

        case "a_supers":
            text_events = reader_db.get_text_events(run_id)
            return len(text_events) > 0

        case "c_overall_pacing":
            stats = reader_db.get_global_stats(run_id)
            if not stats:
                return False
            avg_shot_duration = stats.get("avg_shot_duration")
            if avg_shot_duration is None:
                return False
            return avg_shot_duration > 2.0

        case "d_audio_speech_early_1st_5_secs":
            # Query for audio segments in first 5 seconds
            audio_segments = reader_db.get_audio_segments(run_id, max_start_time=5.0)
            return len(audio_segments) > 0

        case _:
            raise ValueError(f"Unknown baseline feature: {feature_id}")
