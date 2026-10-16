"""Write access for ClipScribe database."""

import json
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Engine
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine

from .engine import (
    ClipScribeAsyncBaseDB,
    ClipScribeBaseDB,
    _upsert_ignore,
    retry_transient,
    retry_transient_async,
)
from .schema import (
    audio_segments_table,
    chat_messages_table,
    field_descriptions_table,
    frame_detections_table,
    global_stats_table,
    jobs_table,
    parser_results_table,
    runs_table,
    scene_descriptions_table,
    shot_boundaries_table,
    text_events_table,
    videos_table,
    visual_object_occurrences_table,
)

logger = logging.getLogger("clip_scribe")


def _native(value: Any) -> Any:
    """Convert numpy scalars to Python native types for DB compatibility."""
    if value is None:
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


class _WriterOperations:
    """Shared synchronous SQL operations for sync and async writers."""

    @staticmethod
    def save_run(
        conn: Connection,
        run_id: str,
        video_name: str,
        video_path: str,
        video_type: str | None,
        video_metadata: Mapping[str, Any],
        field_descriptions: Mapping[str, Mapping[str, str]],
        extraction_mode: str = "with_objects",
    ) -> str:
        _WriterOperations._insert_run(
            conn,
            run_id,
            video_name,
            video_path,
            video_type,
            extraction_mode,
        )
        _WriterOperations._insert_global_stats(
            conn,
            run_id,
            video_metadata.get("global_stats", {}),
        )
        _WriterOperations._insert_visual_objects(
            conn,
            run_id,
            video_metadata.get("visual_objects", []),
        )
        _WriterOperations._insert_text_events(
            conn,
            run_id,
            video_metadata.get("text_events", []),
        )
        _WriterOperations._insert_audio_segments(
            conn,
            run_id,
            video_metadata.get("audio_segments", []),
        )
        _WriterOperations._insert_scene_descriptions(
            conn,
            run_id,
            video_metadata.get("scene_descriptions", []),
        )
        _WriterOperations._insert_shot_boundaries(
            conn,
            run_id,
            video_metadata.get("shot_boundaries", []),
        )
        _WriterOperations._insert_frame_detections(
            conn,
            run_id,
            video_metadata.get("frame_detections", []),
        )
        _WriterOperations._insert_field_descriptions(conn, field_descriptions)
        return run_id

    @staticmethod
    def _insert_run(
        conn: Connection,
        run_id: str,
        video_name: str,
        video_path: str,
        video_type: str | None,
        extraction_mode: str,
    ) -> None:
        conn.execute(
            runs_table.insert().values(
                run_id=run_id,
                video_name=video_name,
                video_path=video_path,
                video_type=video_type,
                extraction_mode=extraction_mode,
            )
        )

    @staticmethod
    def _insert_global_stats(
        conn: Connection,
        run_id: str,
        global_stats: Mapping[str, Any],
    ) -> None:
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

    @staticmethod
    def _insert_visual_objects(
        conn: Connection,
        run_id: str,
        visual_objects: Sequence[Mapping[str, Any]],
    ) -> None:
        rows: list[dict[str, Any]] = []
        for obj in visual_objects:
            global_id = obj.get("global_id")
            label = obj.get("label")
            for occurrence in obj.get("occurrences", []):
                lifespan = occurrence.get("lifespan", [None, None])
                rows.append(
                    {
                        "run_id": run_id,
                        "global_id": global_id,
                        "label": label,
                        "shot_index": occurrence.get("shot_index"),
                        "lifespan_start": _native(
                            lifespan[0] if len(lifespan) > 0 else None
                        ),
                        "lifespan_end": _native(
                            lifespan[1] if len(lifespan) > 1 else None
                        ),
                        "screen_coverage": _native(occurrence.get("screen_coverage")),
                        "velocity_px_sec": _native(occurrence.get("velocity_px_sec")),
                        "growth_factor": _native(occurrence.get("growth_factor")),
                        "direction": occurrence.get("direction"),
                        "centrality_score": _native(occurrence.get("centrality_score")),
                        "screen_time_ratio": _native(
                            occurrence.get("screen_time_ratio")
                        ),
                        "quadrant": occurrence.get("quadrant"),
                    }
                )

        if rows:
            conn.execute(visual_object_occurrences_table.insert(), rows)

    @staticmethod
    def _insert_text_events(
        conn: Connection,
        run_id: str,
        text_events: Sequence[Mapping[str, Any]],
    ) -> None:
        rows: list[dict[str, Any]] = []
        for event in text_events:
            second = event.get("second")
            for line_index, line in enumerate(event.get("text", [])):
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

    @staticmethod
    def _insert_audio_segments(
        conn: Connection,
        run_id: str,
        audio_segments: Sequence[Mapping[str, Any]],
    ) -> None:
        rows = [
            {
                "run_id": run_id,
                "start_time": segment.get("start"),
                "end_time": segment.get("end"),
                "text": segment.get("text"),
                "confidence": segment.get("confidence"),
            }
            for segment in audio_segments
        ]

        if rows:
            conn.execute(audio_segments_table.insert(), rows)

    @staticmethod
    def _insert_scene_descriptions(
        conn: Connection,
        run_id: str,
        scene_descriptions: Sequence[Mapping[str, Any]],
    ) -> None:
        rows = [
            {
                "run_id": run_id,
                "shot_index": description.get("shot_index"),
                "start_time": description.get("start_time"),
                "end_time": description.get("end_time"),
                "description": description.get("description"),
            }
            for description in scene_descriptions
        ]

        if rows:
            conn.execute(scene_descriptions_table.insert(), rows)

    @staticmethod
    def _insert_shot_boundaries(
        conn: Connection,
        run_id: str,
        shot_boundaries: Sequence[Mapping[str, Any]],
    ) -> None:
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

    @staticmethod
    def _insert_frame_detections(
        conn: Connection,
        run_id: str,
        detections: Sequence[Mapping[str, Any]],
    ) -> None:
        rows = [
            {
                "run_id": run_id,
                "shot_index": detection.get("shot_index"),
                "frame_idx": detection.get("frame_idx"),
                "timestamp_sec": _native(detection.get("timestamp_sec")),
                "source": detection.get("source"),
                "label": detection.get("label"),
                "text": detection.get("text"),
                "box_x1": _native(detection.get("box_x1")),
                "box_y1": _native(detection.get("box_y1")),
                "box_x2": _native(detection.get("box_x2")),
                "box_y2": _native(detection.get("box_y2")),
                "confidence": _native(detection.get("confidence")),
                "object_id": _native(detection.get("object_id")),
            }
            for detection in detections
        ]

        if rows:
            conn.execute(frame_detections_table.insert(), rows)

    @staticmethod
    def _insert_field_descriptions(
        conn: Connection,
        descriptions: Mapping[str, Mapping[str, str]],
    ) -> None:
        rows = [
            {
                "table_name": table_name,
                "column_name": column_name,
                "description": description,
            }
            for table_name, columns in descriptions.items()
            for column_name, description in columns.items()
        ]

        if rows:
            statement = _upsert_ignore(
                conn.engine,
                field_descriptions_table,
                rows,
                conflict_columns=["table_name", "column_name"],
            )
            conn.execute(statement)

    @staticmethod
    def insert_video(
        conn: Connection,
        user_id: str,
        content_hash: str,
        stored_key: str,
        original_name: str,
        size_bytes: int | None,
    ) -> None:
        conn.execute(
            videos_table.insert().values(
                user_id=user_id,
                content_hash=content_hash,
                stored_key=stored_key,
                original_name=original_name,
                size_bytes=size_bytes,
            )
        )

    @staticmethod
    def touch_video(conn: Connection, user_id: str, content_hash: str) -> None:
        conn.execute(
            videos_table.update()
            .where(
                (videos_table.c.user_id == user_id)
                & (videos_table.c.content_hash == content_hash)
            )
            .values(last_seen_at=datetime.now(timezone.utc).isoformat())
        )

    @staticmethod
    def delete_video(conn: Connection, user_id: str, stored_key: str) -> None:
        conn.execute(
            videos_table.delete().where(
                (videos_table.c.user_id == user_id)
                & (videos_table.c.stored_key == stored_key)
            )
        )

    @staticmethod
    def create_job(
        conn: Connection,
        job_id: str,
        mode: str,
        status: str,
        parent_job_id: str | None,
        run_id: str | None,
        video_name: str | None,
        video_path: str | None,
        video_type: str | None,
        device: str | None,
        platform: str | None,
        params_json: dict[str, Any] | None,
        created_by: str | None,
    ) -> None:
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

    @staticmethod
    def _job_update_values(
        *,
        status: str | None,
        run_id: str | None,
        celery_task_id: str | None,
        started_at: str | None,
        finished_at: str | None,
        error_text: str | None,
    ) -> dict[str, Any]:
        return {
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

    @staticmethod
    def update_job(
        conn: Connection,
        job_id: str,
        status: str | None,
        run_id: str | None,
        celery_task_id: str | None,
        started_at: str | None,
        finished_at: str | None,
        error_text: str | None,
    ) -> None:
        values = _WriterOperations._job_update_values(
            status=status,
            run_id=run_id,
            celery_task_id=celery_task_id,
            started_at=started_at,
            finished_at=finished_at,
            error_text=error_text,
        )
        if values:
            conn.execute(
                jobs_table.update()
                .where(jobs_table.c.job_id == job_id)
                .values(**values)
            )

    @staticmethod
    def update_job_if_status(
        conn: Connection,
        job_id: str,
        allowed_statuses: Sequence[str],
        status: str | None,
        run_id: str | None,
        celery_task_id: str | None,
        started_at: str | None,
        finished_at: str | None,
        error_text: str | None,
    ) -> bool:
        values = _WriterOperations._job_update_values(
            status=status,
            run_id=run_id,
            celery_task_id=celery_task_id,
            started_at=started_at,
            finished_at=finished_at,
            error_text=error_text,
        )
        if not values:
            return False

        result = conn.execute(
            jobs_table.update()
            .where(jobs_table.c.job_id == job_id)
            .where(jobs_table.c.status.in_(allowed_statuses))
            .values(**values)
        )
        return result.rowcount > 0

    @staticmethod
    def delete_job(conn: Connection, job_id: str) -> bool:
        result = conn.execute(jobs_table.delete().where(jobs_table.c.job_id == job_id))
        if result.rowcount > 0:
            conn.execute(
                chat_messages_table.delete().where(
                    chat_messages_table.c.job_id == job_id
                )
            )
        return result.rowcount > 0

    @staticmethod
    def delete_run(conn: Connection, run_id: str) -> None:
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
        for table in run_keyed:
            conn.execute(table.delete().where(table.c.run_id == run_id))

    @staticmethod
    def reset_job_for_retry(conn: Connection, job_id: str, run_id: str) -> bool:
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

    @staticmethod
    def add_chat_message(
        conn: Connection,
        session_id: str,
        role: str,
        content: str,
        run_id: str | None,
        job_id: str | None,
        tool_calls: list[str] | None,
    ) -> None:
        conn.execute(
            chat_messages_table.insert(),
            {
                "run_id": run_id,
                "job_id": job_id,
                "session_id": session_id,
                "role": role,
                "content": content,
                "tool_calls_json": tool_calls,
            },
        )

    @staticmethod
    def delete_chat_session(conn: Connection, run_id: str, session_id: str) -> int:
        result = conn.execute(
            chat_messages_table.delete()
            .where(chat_messages_table.c.run_id == run_id)
            .where(chat_messages_table.c.session_id == session_id)
        )
        return result.rowcount

    @staticmethod
    def delete_job_chat_session(
        conn: Connection,
        job_id: str,
        session_id: str,
    ) -> int:
        result = conn.execute(
            chat_messages_table.delete()
            .where(chat_messages_table.c.job_id == job_id)
            .where(chat_messages_table.c.session_id == session_id)
        )
        return result.rowcount

    @staticmethod
    def save_parser_results(
        conn: Connection,
        run_id: str,
        platform: str,
        results: list[Any],
    ) -> None:
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

        conn.execute(
            parser_results_table.delete()
            .where(parser_results_table.c.run_id == run_id)
            .where(parser_results_table.c.platform == platform)
        )
        if rows:
            conn.execute(parser_results_table.insert(), rows)


class ClipScribeAsyncWriterDB(ClipScribeAsyncBaseDB):
    def __init__(self, async_engine: AsyncEngine) -> None:
        super().__init__(async_engine)
        self.writer_op = _WriterOperations()
        logger.info("ClipScribeAsyncWriterDB ready.")

    @retry_transient_async
    async def save_run(
        self,
        run_id: str,
        video_name: str,
        video_path: str,
        video_type: str | None,
        video_metadata: Mapping[str, Any],
        field_descriptions: Mapping[str, Mapping[str, str]],
        extraction_mode: str = "with_objects",
    ) -> str:
        """Persist a full extraction run in one transaction."""
        async with self._async_engine.begin() as conn:
            saved_run_id = await conn.run_sync(
                self.writer_op.save_run,
                run_id,
                video_name,
                video_path,
                video_type,
                video_metadata,
                field_descriptions,
                extraction_mode,
            )
        logger.info("Saved run %s for '%s'", run_id, video_name)
        return saved_run_id

    @retry_transient_async
    async def insert_video(
        self,
        *,
        user_id: str,
        content_hash: str,
        stored_key: str,
        original_name: str,
        size_bytes: int | None,
    ) -> None:
        async with self._async_engine.begin() as conn:
            await conn.run_sync(
                self.writer_op.insert_video,
                user_id,
                content_hash,
                stored_key,
                original_name,
                size_bytes,
            )
        logger.info("Registered video %s (%s)", stored_key, original_name)

    @retry_transient_async
    async def touch_video(self, user_id: str, content_hash: str) -> None:
        async with self._async_engine.begin() as conn:
            await conn.run_sync(self.writer_op.touch_video, user_id, content_hash)

    @retry_transient_async
    async def delete_video(self, user_id: str, stored_key: str) -> None:
        async with self._async_engine.begin() as conn:
            await conn.run_sync(self.writer_op.delete_video, user_id, stored_key)
        logger.info("Pruned orphaned video registry row %s", stored_key)

    @retry_transient_async
    async def create_job(
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
        params_json: dict[str, Any] | None = None,
        created_by: str | None = None,
    ) -> None:
        async with self._async_engine.begin() as conn:
            await conn.run_sync(
                self.writer_op.create_job,
                job_id,
                mode,
                status,
                parent_job_id,
                run_id,
                video_name,
                video_path,
                video_type,
                device,
                platform,
                params_json,
                created_by,
            )
        logger.info("Created job %s (mode=%s, status=%s)", job_id, mode, status)

    @retry_transient_async
    async def update_job(
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
        async with self._async_engine.begin() as conn:
            await conn.run_sync(
                self.writer_op.update_job,
                job_id,
                status,
                run_id,
                celery_task_id,
                started_at,
                finished_at,
                error_text,
            )

    @retry_transient_async
    async def update_job_if_status(
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
        async with self._async_engine.begin() as conn:
            return await conn.run_sync(
                self.writer_op.update_job_if_status,
                job_id,
                allowed_statuses,
                status,
                run_id,
                celery_task_id,
                started_at,
                finished_at,
                error_text,
            )

    @retry_transient_async
    async def delete_job(self, job_id: str) -> bool:
        async with self._async_engine.begin() as conn:
            return await conn.run_sync(self.writer_op.delete_job, job_id)

    @retry_transient_async
    async def delete_run(self, run_id: str) -> None:
        async with self._async_engine.begin() as conn:
            await conn.run_sync(self.writer_op.delete_run, run_id)

    @retry_transient_async
    async def reset_job_for_retry(self, job_id: str, *, run_id: str) -> bool:
        async with self._async_engine.begin() as conn:
            return await conn.run_sync(
                self.writer_op.reset_job_for_retry,
                job_id,
                run_id,
            )

    @retry_transient_async
    async def add_chat_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        run_id: str | None = None,
        job_id: str | None = None,
        tool_calls: list[str] | None = None,
    ) -> None:
        async with self._async_engine.begin() as conn:
            await conn.run_sync(
                self.writer_op.add_chat_message,
                session_id,
                role,
                content,
                run_id,
                job_id,
                tool_calls,
            )

    @retry_transient_async
    async def delete_chat_session(self, run_id: str, session_id: str) -> int:
        async with self._async_engine.begin() as conn:
            return await conn.run_sync(
                self.writer_op.delete_chat_session,
                run_id,
                session_id,
            )

    @retry_transient_async
    async def delete_job_chat_session(self, job_id: str, session_id: str) -> int:
        async with self._async_engine.begin() as conn:
            return await conn.run_sync(
                self.writer_op.delete_job_chat_session,
                job_id,
                session_id,
            )

    @retry_transient_async
    async def save_parser_results(
        self,
        run_id: str,
        platform: str,
        results: list[Any],
    ) -> None:
        async with self._async_engine.begin() as conn:
            await conn.run_sync(
                self.writer_op.save_parser_results,
                run_id,
                platform,
                results,
            )
        logger.info("Saved %d parser results for run %s", len(results), run_id)


class ClipScribeWriterDB(ClipScribeBaseDB):
    def __init__(self, engine: Engine) -> None:
        super().__init__(engine)
        self.writer_op = _WriterOperations()
        logger.info("ClipScribeWriterDB ready.")

    @retry_transient
    def save_run(
        self,
        run_id: str,
        video_name: str,
        video_path: str,
        video_type: str | None,
        video_metadata: Mapping[str, Any],
        field_descriptions: Mapping[str, Mapping[str, str]],
        extraction_mode: str = "with_objects",
    ) -> str:
        """Persist a full extraction run under the supplied ``run_id``."""
        with self._engine.begin() as conn:
            saved_run_id = self.writer_op.save_run(
                conn,
                run_id,
                video_name,
                video_path,
                video_type,
                video_metadata,
                field_descriptions,
                extraction_mode,
            )
        logger.info("Saved run %s for '%s'", run_id, video_name)
        return saved_run_id

    @retry_transient
    def insert_video(
        self,
        *,
        user_id: str,
        content_hash: str,
        stored_key: str,
        original_name: str,
        size_bytes: int | None,
    ) -> None:
        with self._engine.begin() as conn:
            self.writer_op.insert_video(
                conn,
                user_id,
                content_hash,
                stored_key,
                original_name,
                size_bytes,
            )
        logger.info("Registered video %s (%s)", stored_key, original_name)

    @retry_transient
    def touch_video(self, user_id: str, content_hash: str) -> None:
        with self._engine.begin() as conn:
            self.writer_op.touch_video(conn, user_id, content_hash)

    @retry_transient
    def delete_video(self, user_id: str, stored_key: str) -> None:
        with self._engine.begin() as conn:
            self.writer_op.delete_video(conn, user_id, stored_key)
        logger.info("Pruned orphaned video registry row %s", stored_key)

    @retry_transient
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
        params_json: dict[str, Any] | None = None,
        created_by: str | None = None,
    ) -> None:
        with self._engine.begin() as conn:
            self.writer_op.create_job(
                conn,
                job_id,
                mode,
                status,
                parent_job_id,
                run_id,
                video_name,
                video_path,
                video_type,
                device,
                platform,
                params_json,
                created_by,
            )
        logger.info("Created job %s (mode=%s, status=%s)", job_id, mode, status)

    @retry_transient
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
        with self._engine.begin() as conn:
            self.writer_op.update_job(
                conn,
                job_id,
                status,
                run_id,
                celery_task_id,
                started_at,
                finished_at,
                error_text,
            )

    @retry_transient
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
        with self._engine.begin() as conn:
            return self.writer_op.update_job_if_status(
                conn,
                job_id,
                allowed_statuses,
                status,
                run_id,
                celery_task_id,
                started_at,
                finished_at,
                error_text,
            )

    @retry_transient
    def delete_job(self, job_id: str) -> bool:
        with self._engine.begin() as conn:
            return self.writer_op.delete_job(conn, job_id)

    @retry_transient
    def delete_run(self, run_id: str) -> None:
        with self._engine.begin() as conn:
            self.writer_op.delete_run(conn, run_id)

    @retry_transient
    def reset_job_for_retry(self, job_id: str, *, run_id: str) -> bool:
        with self._engine.begin() as conn:
            return self.writer_op.reset_job_for_retry(conn, job_id, run_id)

    @retry_transient
    def add_chat_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        run_id: str | None = None,
        job_id: str | None = None,
        tool_calls: list[str] | None = None,
    ) -> None:
        with self._engine.begin() as conn:
            self.writer_op.add_chat_message(
                conn,
                session_id,
                role,
                content,
                run_id,
                job_id,
                tool_calls,
            )

    @retry_transient
    def delete_chat_session(self, run_id: str, session_id: str) -> int:
        with self._engine.begin() as conn:
            return self.writer_op.delete_chat_session(conn, run_id, session_id)

    @retry_transient
    def delete_job_chat_session(self, job_id: str, session_id: str) -> int:
        with self._engine.begin() as conn:
            return self.writer_op.delete_job_chat_session(conn, job_id, session_id)

    @retry_transient
    def save_parser_results(
        self,
        run_id: str,
        platform: str,
        results: list[Any],
    ) -> None:
        with self._engine.begin() as conn:
            self.writer_op.save_parser_results(conn, run_id, platform, results)
        logger.info("Saved %d parser results for run %s", len(results), run_id)
