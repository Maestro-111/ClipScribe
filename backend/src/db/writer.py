"""Write access for ClipScribe database."""

import json
import uuid
import logging

from sqlalchemy import Engine

from .engine import ClipScribeBaseDB, _upsert_ignore
from .schema import (
    runs_table,
    global_stats_table,
    visual_object_occurrences_table,
    text_events_table,
    audio_segments_table,
    scene_descriptions_table,
    field_descriptions_table,
)

logger = logging.getLogger("clip_scribe")


def _native(val):
    """Convert numpy scalars to Python native types for DB compatibility."""
    if val is None:
        return None
    if hasattr(val, "item"):
        return val.item()
    return val


class ClipScribeWriterDB(ClipScribeBaseDB):
    def __init__(self, engine: Engine):
        super().__init__(engine)
        logger.info("ClipScribeWriterDB ready.")

    def save_run(
        self,
        video_name: str,
        video_path: str,
        video_type: str,
        video_metadata: dict,
        field_descriptions: dict,
    ) -> str:
        run_id = str(uuid.uuid4())

        with self._engine.begin() as conn:
            self._insert_run(conn, run_id, video_name, video_path, video_type)
            self._insert_global_stats(
                conn, run_id, video_metadata.get("global_stats", {})
            )
            self._insert_visual_objects(
                conn, run_id, video_metadata.get("visual_objects", [])
            )
            self._insert_text_events(
                conn, run_id, video_metadata.get("text_events", [])
            )
            self._insert_audio_segments(
                conn, run_id, video_metadata.get("audio_segments", [])
            )
            self._insert_scene_descriptions(
                conn, run_id, video_metadata.get("scene_descriptions", [])
            )
            self._insert_field_descriptions(conn, field_descriptions)

        logger.info(f"Saved run {run_id} for '{video_name}'")
        return run_id

    def _insert_run(
        self, conn, run_id: str, video_name: str, video_path: str, video_type: str
    ) -> None:
        conn.execute(
            runs_table.insert().values(
                run_id=run_id,
                video_name=video_name,
                video_path=video_path,
                video_type=video_type,
            )
        )

    def _insert_global_stats(self, conn, run_id: str, global_stats: dict) -> None:
        ds = global_stats.get("dynamic_start", {})
        qpi = global_stats.get("quick_pacing_intro", {})
        qpg = global_stats.get("quick_pacing_general", {})

        conn.execute(
            global_stats_table.insert().values(
                run_id=run_id,
                total_shots=global_stats.get("total_shots"),
                video_duration=global_stats.get("video_duration"),
                avg_shot_duration=global_stats.get("avg_shot_duration"),
                dynamic_start_detected=int(ds.get("detected", False)),
                dynamic_start_first_shot_dur=ds.get("first_shot_duration"),
                dynamic_start_criteria=ds.get("criteria"),
                qp_intro_detected=int(qpi.get("detected", False)),
                qp_intro_shot_count=qpi.get("shot_count"),
                qp_intro_shots=json.dumps(qpi.get("shots", [])),
                qp_intro_criteria=qpi.get("criteria"),
                qp_general_detected=int(qpg.get("detected", False)),
                qp_general_rapid_fire_segments=json.dumps(
                    qpg.get("rapid_fire_segments", [])
                ),
                qp_general_criteria=qpg.get("criteria"),
            )
        )

    def _insert_visual_objects(self, conn, run_id: str, visual_objects: list) -> None:
        rows = []
        for obj in visual_objects:
            g_id = obj.get("global_id")
            label = obj.get("label")
            for occ in obj.get("occurrences", []):
                lifespan = occ.get("lifespan", [None, None])
                rows.append(
                    {
                        "run_id": run_id,
                        "global_id": g_id,
                        "label": label,
                        "shot_index": occ.get("shot_index"),
                        "lifespan_start": _native(
                            lifespan[0] if len(lifespan) > 0 else None
                        ),
                        "lifespan_end": _native(
                            lifespan[1] if len(lifespan) > 1 else None
                        ),
                        "screen_coverage": _native(occ.get("screen_coverage")),
                        "velocity_px_sec": _native(occ.get("velocity_px_sec")),
                        "growth_factor": _native(occ.get("growth_factor")),
                        "direction": occ.get("direction"),
                        "centrality_score": _native(occ.get("centrality_score")),
                        "screen_time_ratio": _native(occ.get("screen_time_ratio")),
                        "quadrant": occ.get("quadrant"),
                    }
                )

        if rows:
            conn.execute(visual_object_occurrences_table.insert(), rows)

    def _insert_text_events(self, conn, run_id: str, text_events: list) -> None:
        rows = []
        for ev in text_events:
            second = ev.get("second")
            for line_index, line in enumerate(ev.get("text", [])):
                rows.append(
                    {
                        "run_id": run_id,
                        "second": second,
                        "line_index": line_index,
                        "text": line,
                    }
                )

        if rows:
            conn.execute(text_events_table.insert(), rows)

    def _insert_audio_segments(self, conn, run_id: str, audio_segments: list) -> None:
        rows = [
            {
                "run_id": run_id,
                "start_time": seg.get("start"),
                "end_time": seg.get("end"),
                "text": seg.get("text"),
                "confidence": seg.get("confidence"),
            }
            for seg in audio_segments
        ]

        if rows:
            conn.execute(audio_segments_table.insert(), rows)

    def _insert_scene_descriptions(
        self, conn, run_id: str, scene_descriptions: list
    ) -> None:
        rows = [
            {
                "run_id": run_id,
                "shot_index": desc.get("shot_index"),
                "start_time": desc.get("start_time"),
                "end_time": desc.get("end_time"),
                "description": desc.get("description"),
            }
            for desc in scene_descriptions
        ]

        if rows:
            conn.execute(scene_descriptions_table.insert(), rows)

    def _insert_field_descriptions(self, conn, descriptions: dict) -> None:
        rows = []
        for table_name, columns in descriptions.items():
            for column_name, description in columns.items():
                rows.append(
                    {
                        "table_name": table_name,
                        "column_name": column_name,
                        "description": description,
                    }
                )

        if rows:
            stmt = _upsert_ignore(
                self._engine,
                field_descriptions_table,
                rows,
                conflict_columns=["table_name", "column_name"],
            )
            conn.execute(stmt)
