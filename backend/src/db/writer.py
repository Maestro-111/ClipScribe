"""Write access for ClipScribe database."""

import json
import logging
from collections.abc import Sequence
from typing import Any, Mapping

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
    frame_detections_table,
    shot_boundaries_table,
    parser_results_table,
    jobs_table,
    chat_messages_table,
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
        run_id: str,
        video_name: str,
        video_path: str,
        video_type: str | None,
        video_metadata: Mapping[str, Any],
        field_descriptions: Mapping[str, Mapping[str, str]],
    ) -> str:
        """Persist a full extraction run under the caller-supplied ``run_id``.

        The id is generated up front by the engine (a ULID) so the extractor
        can key its artifact directory and raw detections by the same value
        before the run row exists.
        """
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
            self._insert_shot_boundaries(
                conn, run_id, video_metadata.get("shot_boundaries", [])
            )
            self._insert_frame_detections(
                conn, run_id, video_metadata.get("frame_detections", [])
            )
            self._insert_field_descriptions(conn, field_descriptions)

        logger.info(f"Saved run {run_id} for '{video_name}'")
        return run_id

    def _insert_run(
        self,
        conn,
        run_id: str,
        video_name: str,
        video_path: str,
        video_type: str | None,
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

    def _insert_shot_boundaries(self, conn, run_id: str, shot_boundaries: list) -> None:
        rows = [
            {
                "run_id": run_id,
                "shot_index": shot.get("index"),
                "start_sec": shot.get("start"),
                "end_sec": shot.get("end"),
                "duration_sec": shot.get("duration"),
            }
            for shot in shot_boundaries
        ]

        if rows:
            conn.execute(shot_boundaries_table.insert(), rows)

    def _insert_frame_detections(self, conn, run_id: str, detections: list) -> None:
        rows = [
            {
                "run_id": run_id,
                "shot_index": det.get("shot_index"),
                "frame_idx": det.get("frame_idx"),
                "timestamp_sec": _native(det.get("timestamp_sec")),
                "source": det.get("source"),
                "label": det.get("label"),
                "text": det.get("text"),
                "box_x1": _native(det.get("box_x1")),
                "box_y1": _native(det.get("box_y1")),
                "box_x2": _native(det.get("box_x2")),
                "box_y2": _native(det.get("box_y2")),
                "confidence": _native(det.get("confidence")),
                "object_id": _native(det.get("object_id")),
            }
            for det in detections
        ]

        if rows:
            conn.execute(frame_detections_table.insert(), rows)

    def create_job(
        self,
        *,
        job_id: str,
        mode: str,
        status: str = "queued",
        parent_job_id: str | None = None,
        run_id: str | None = None,
        video_name: str | None = None,
        video_path: str | None = None,
        video_type: str | None = None,
        device: str | None = None,
        platform: str | None = None,
        params_json: dict | None = None,
        created_by: str | None = None,
    ) -> None:
        """Insert a new orchestration job row (``created_at`` is DB-defaulted).

        Orchestration state, kept separate from ``runs`` (extractor output).
        ``run_id`` may be minted up front so the row links to a run before the
        extractor writes it (web-app-plan §4). ``parent_job_id`` links a child
        run to its batch parent; NULL marks a parent/standalone job.
        """
        with self._engine.begin() as conn:
            conn.execute(
                jobs_table.insert(),
                {
                    "job_id": job_id,
                    "parent_job_id": parent_job_id,
                    "run_id": run_id,
                    "status": status,
                    "mode": mode,
                    "video_name": video_name,
                    "video_path": video_path,
                    "video_type": video_type,
                    "device": device,
                    "platform": platform,
                    "params_json": params_json,
                    "created_by": created_by,
                },
            )
        logger.info(f"Created job {job_id} (mode={mode}, status={status})")

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        run_id: str | None = None,
        celery_task_id: str | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
        error_text: str | None = None,
    ) -> None:
        """Update mutable orchestration fields on a job.

        Only fields passed as non-``None`` are written, so callers can advance
        one part of the lifecycle (e.g. just ``status`` + ``started_at``)
        without clobbering the rest.
        """
        values = {
            key: value
            for key, value in {
                "status": status,
                "run_id": run_id,
                "celery_task_id": celery_task_id,
                "started_at": started_at,
                "finished_at": finished_at,
                "error_text": error_text,
            }.items()
            if value is not None
        }
        if not values:
            return

        with self._engine.begin() as conn:
            conn.execute(
                jobs_table.update()
                .where(jobs_table.c.job_id == job_id)
                .values(**values)
            )

    def update_job_if_status(
        self,
        job_id: str,
        *,
        allowed_statuses: Sequence[str],
        status: str | None = None,
        run_id: str | None = None,
        celery_task_id: str | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
        error_text: str | None = None,
    ) -> bool:
        values = {
            key: value
            for key, value in {
                "status": status,
                "run_id": run_id,
                "celery_task_id": celery_task_id,
                "started_at": started_at,
                "finished_at": finished_at,
                "error_text": error_text,
            }.items()
            if value is not None
        }
        if not values:
            return False

        with self._engine.begin() as conn:
            result = conn.execute(
                jobs_table.update()
                .where(jobs_table.c.job_id == job_id)
                .where(jobs_table.c.status.in_(allowed_statuses))
                .values(**values)
            )
            return result.rowcount > 0

    def delete_job(self, job_id: str) -> bool:
        """Delete a job row. Returns True if a row was deleted, False if not found."""
        with self._engine.begin() as conn:
            result = conn.execute(
                jobs_table.delete().where(jobs_table.c.job_id == job_id)
            )
            return result.rowcount > 0

    def delete_run(self, run_id: str) -> None:
        """Delete all persisted data for a run across every run-keyed table.

        Used when a run is superseded (e.g. an in-place retry mints a new
        ``run_id`` and the old run's rows must not linger). One transaction; no
        FK constraints are declared, so table order is immaterial. The ``jobs``
        table is intentionally untouched — orchestration rows are managed
        separately.
        """
        run_keyed = (
            runs_table,
            global_stats_table,
            visual_object_occurrences_table,
            text_events_table,
            audio_segments_table,
            scene_descriptions_table,
            frame_detections_table,
            shot_boundaries_table,
            parser_results_table,
            chat_messages_table,
        )
        with self._engine.begin() as conn:
            for table in run_keyed:
                conn.execute(table.delete().where(table.c.run_id == run_id))

    def reset_job_for_retry(self, job_id: str, *, run_id: str) -> bool:
        """Reset a terminal job back to ``queued`` for an in-place retry.

        Assigns a fresh ``run_id`` and clears the terminal fields so a re-run
        starts clean. Guarded to terminal statuses so a concurrent transition
        can't clobber a live job. Returns True if the row was reset. Unlike
        :meth:`update_job`, this writes explicit NULLs (that method skips
        ``None`` values, so it can't clear a field).
        """
        with self._engine.begin() as conn:
            result = conn.execute(
                jobs_table.update()
                .where(jobs_table.c.job_id == job_id)
                .where(jobs_table.c.status.in_(("completed", "failed", "canceled")))
                .values(
                    status="queued",
                    run_id=run_id,
                    started_at=None,
                    finished_at=None,
                    error_text=None,
                    celery_task_id=None,
                )
            )
            return result.rowcount > 0

    def add_chat_message(
        self,
        *,
        run_id: str,
        session_id: str,
        role: str,
        content: str,
        tool_calls: list[str] | None = None,
    ) -> None:
        """Append one advisory-chat message to a session transcript (§13)."""
        with self._engine.begin() as conn:
            conn.execute(
                chat_messages_table.insert(),
                {
                    "run_id": run_id,
                    "session_id": session_id,
                    "role": role,
                    "content": content,
                    "tool_calls_json": tool_calls,
                },
            )

    def delete_chat_session(self, run_id: str, session_id: str) -> int:
        """Delete all messages in one chat session. Returns rows removed."""
        with self._engine.begin() as conn:
            result = conn.execute(
                chat_messages_table.delete()
                .where(chat_messages_table.c.run_id == run_id)
                .where(chat_messages_table.c.session_id == session_id)
            )
            return result.rowcount

    def save_parser_results(self, run_id: str, platform: str, results: list) -> None:
        """Persist per-criterion parser evaluations for a run.

        ``results`` are platform feature-result models (e.g.
        ``YouTubeFeatureResult``); feature fields are read via ``getattr`` so
        platforms without them still persist the common columns.
        """
        rows = [
            {
                "run_id": run_id,
                "platform": getattr(result, "platform", platform),
                "feature_category": getattr(result, "feature_category", None),
                "feature_name": getattr(result, "feature_name", None),
                "feature_criteria": getattr(result, "feature_criteria", None),
                "evaluation": bool(getattr(result, "evaluation", False)),
                "llm_prompt": getattr(result, "llm_prompt", None),
                "llm_explanation": getattr(result, "llm_explanation", None),
                "langsmith_run_id": getattr(result, "langsmith_run_id", None),
            }
            for result in results
        ]

        with self._engine.begin() as conn:
            conn.execute(
                parser_results_table.delete()
                .where(parser_results_table.c.run_id == run_id)
                .where(parser_results_table.c.platform == platform)
            )

            if rows:
                conn.execute(parser_results_table.insert(), rows)

        logger.info(f"Saved {len(rows)} parser results for run {run_id}")

    def _insert_field_descriptions(
        self, conn, descriptions: Mapping[str, Mapping[str, str]]
    ) -> None:
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
