"""Read-only database access for ClipScribe."""

import json
import logging

from sqlalchemy import Engine, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine

from .engine import ClipScribeAsyncBaseDB, ClipScribeBaseDB

logger = logging.getLogger("clip_scribe")


class _ReaderOperations:
    """Shared query and decoding logic for sync and async readers."""

    @staticmethod
    def get_run(conn: Connection, run_id: str) -> dict | None:
        result = conn.execute(
            text("SELECT * FROM runs WHERE run_id = :run_id"),
            {"run_id": run_id},
        )
        row = result.mappings().fetchone()
        return dict(row) if row else None

    @staticmethod
    def get_latest_run(conn: Connection) -> dict | None:
        result = conn.execute(
            text("SELECT * FROM runs ORDER BY created_at DESC LIMIT 1")
        )
        row = result.mappings().fetchone()
        return dict(row) if row else None

    @staticmethod
    def get_global_stats(conn: Connection, run_id: str) -> dict | None:
        result = conn.execute(
            text("SELECT * FROM global_stats WHERE run_id = :run_id"),
            {"run_id": run_id},
        )
        row = result.mappings().fetchone()
        if not row:
            return None
        stats = dict(row)
        if stats.get("qp_intro_shots"):
            stats["qp_intro_shots"] = json.loads(stats["qp_intro_shots"])
        if stats.get("qp_general_rapid_fire_segments"):
            stats["qp_general_rapid_fire_segments"] = json.loads(
                stats["qp_general_rapid_fire_segments"]
            )
        return stats

    @staticmethod
    def get_audio_segments(
        conn: Connection,
        run_id: str,
        max_start_time: float | None = None,
    ) -> list[dict]:
        query = "SELECT * FROM audio_segments WHERE run_id = :run_id"
        params: dict = {"run_id": run_id}

        if max_start_time is not None:
            query += " AND start_time < :max_start_time"
            params["max_start_time"] = max_start_time

        query += " ORDER BY start_time"
        result = conn.execute(text(query), params)
        return [dict(row) for row in result.mappings().fetchall()]

    @staticmethod
    def get_text_events(
        conn: Connection,
        run_id: str,
        max_second: int | None = None,
    ) -> list[dict]:
        query = "SELECT * FROM text_events WHERE run_id = :run_id"
        params: dict = {"run_id": run_id}

        if max_second is not None:
            query += " AND second < :max_second"
            params["max_second"] = max_second

        query += " ORDER BY second, line_index"
        result = conn.execute(text(query), params)
        return [dict(row) for row in result.mappings().fetchall()]

    @staticmethod
    def get_visual_objects(
        conn: Connection,
        run_id: str,
        label_contains: str | None = None,
        max_lifespan_start: float | None = None,
    ) -> list[dict]:
        query = "SELECT * FROM visual_object_occurrences WHERE run_id = :run_id"
        params: dict = {"run_id": run_id}

        if label_contains is not None:
            # Use ILIKE on PostgreSQL for case-insensitive match, LIKE on SQLite.
            if conn.dialect.name == "postgresql":
                query += " AND label ILIKE :label_pattern"
            else:
                query += " AND label LIKE :label_pattern"
            params["label_pattern"] = f"%{label_contains}%"

        if max_lifespan_start is not None:
            query += " AND lifespan_start < :max_lifespan_start"
            params["max_lifespan_start"] = max_lifespan_start

        query += " ORDER BY lifespan_start"
        result = conn.execute(text(query), params)
        return [dict(row) for row in result.mappings().fetchall()]

    @staticmethod
    def get_field_descriptions(
        conn: Connection,
        table_name: str | None = None,
    ) -> list[dict]:
        query = "SELECT table_name, column_name, description FROM field_descriptions"
        params: dict = {}

        if table_name is not None:
            query += " WHERE table_name = :table_name"
            params["table_name"] = table_name

        query += " ORDER BY table_name, column_name"
        result = conn.execute(text(query), params)
        return [dict(row) for row in result.mappings().fetchall()]

    @staticmethod
    def get_scene_descriptions(
        conn: Connection,
        run_id: str,
        max_start_time: float | None = None,
        max_end_time: float | None = None,
    ) -> list[dict]:
        query = "SELECT * FROM scene_descriptions WHERE run_id = :run_id"
        params: dict = {"run_id": run_id}

        if max_start_time is not None:
            query += " AND start_time < :max_start_time"
            params["max_start_time"] = max_start_time

        if max_end_time is not None:
            query += " AND end_time < :max_end_time"
            params["max_end_time"] = max_end_time

        query += " ORDER BY start_time"
        result = conn.execute(text(query), params)
        return [dict(row) for row in result.mappings().fetchall()]

    @staticmethod
    def list_runs(conn: Connection, limit: int, offset: int) -> list[dict]:
        result = conn.execute(
            text(
                "SELECT * FROM runs ORDER BY created_at DESC "
                "LIMIT :limit OFFSET :offset"
            ),
            {"limit": limit, "offset": offset},
        )
        return [dict(row) for row in result.mappings().fetchall()]

    @staticmethod
    def get_shot_boundaries(conn: Connection, run_id: str) -> list[dict]:
        result = conn.execute(
            text(
                "SELECT * FROM shot_boundaries WHERE run_id = :run_id "
                "ORDER BY shot_index"
            ),
            {"run_id": run_id},
        )
        return [dict(row) for row in result.mappings().fetchall()]

    @staticmethod
    def get_frame_detections(
        conn: Connection,
        run_id: str,
        from_sec: float | None = None,
        to_sec: float | None = None,
    ) -> list[dict]:
        query = "SELECT * FROM frame_detections WHERE run_id = :run_id"
        params: dict = {"run_id": run_id}

        if from_sec is not None:
            query += " AND timestamp_sec >= :from_sec"
            params["from_sec"] = from_sec
        if to_sec is not None:
            query += " AND timestamp_sec <= :to_sec"
            params["to_sec"] = to_sec

        query += " ORDER BY frame_idx, id"
        result = conn.execute(text(query), params)
        return [dict(row) for row in result.mappings().fetchall()]

    @staticmethod
    def get_parser_results(conn: Connection, run_id: str) -> list[dict]:
        result = conn.execute(
            text(
                "SELECT * FROM parser_results WHERE run_id = :run_id "
                "ORDER BY feature_category, feature_name"
            ),
            {"run_id": run_id},
        )
        return [dict(row) for row in result.mappings().fetchall()]

    @staticmethod
    def get_chat_messages(
        conn: Connection,
        scope_col: str,
        scope_val: str,
        session_id: str,
    ) -> list[dict]:
        """Return one transcript scoped by the internal run/job column literal."""
        result = conn.execute(
            text(
                "SELECT id, run_id, job_id, session_id, role, content, "
                "tool_calls_json, created_at FROM chat_messages "
                f"WHERE {scope_col} = :scope AND session_id = :session_id "
                "ORDER BY id"
            ),
            {"scope": scope_val, "session_id": session_id},
        )
        rows = [dict(row) for row in result.mappings().fetchall()]
        for row in rows:
            raw = row.get("tool_calls_json")
            if isinstance(raw, str) and raw:
                try:
                    row["tool_calls_json"] = json.loads(raw)
                except json.JSONDecodeError:
                    pass
        return rows

    @staticmethod
    def get_chat_sessions(
        conn: Connection,
        scope_col: str,
        scope_val: str,
    ) -> list[dict]:
        """Return chat summaries scoped by the internal run/job column literal."""
        result = conn.execute(
            text(
                "SELECT session_id, COUNT(*) AS message_count, "
                "MIN(created_at) AS started_at, MAX(created_at) AS last_at, "
                "(SELECT content FROM chat_messages c2 "
                f" WHERE c2.{scope_col} = c.{scope_col} "
                "  AND c2.session_id = c.session_id "
                " ORDER BY id LIMIT 1) AS title "
                f"FROM chat_messages c WHERE {scope_col} = :scope "
                "GROUP BY session_id ORDER BY MAX(id) DESC"
            ),
            {"scope": scope_val},
        )
        return [dict(row) for row in result.mappings().fetchall()]

    @staticmethod
    def get_job(conn: Connection, job_id: str) -> dict | None:
        result = conn.execute(
            text("SELECT * FROM jobs WHERE job_id = :job_id"),
            {"job_id": job_id},
        )
        row = result.mappings().fetchone()
        return _ReaderOperations._decode_job(dict(row)) if row else None

    @staticmethod
    def list_jobs(
        conn: Connection,
        user_id: str,
        status: str | None,
        limit: int,
        offset: int,
    ) -> list[dict]:
        query = "SELECT * FROM jobs WHERE created_by = :user_id"
        params: dict = {"user_id": user_id, "limit": limit, "offset": offset}

        if status is not None:
            query += " AND status = :status"
            params["status"] = status

        query += " ORDER BY created_at DESC, job_id DESC LIMIT :limit OFFSET :offset"
        result = conn.execute(text(query), params)
        return [
            _ReaderOperations._decode_job(dict(row))
            for row in result.mappings().fetchall()
        ]

    @staticmethod
    def list_parent_jobs(
        conn: Connection,
        user_id: str,
        limit: int,
        offset: int,
        status: str | None,
    ) -> list[dict]:
        query = """
            WITH child_status AS (
                SELECT
                    parent_job_id,
                    COUNT(*) AS child_count,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END)
                        AS completed_count,
                    SUM(CASE
                        WHEN status IN ('completed', 'failed', 'canceled')
                        THEN 1 ELSE 0
                    END) AS terminal_count,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END)
                        AS failed_count,
                    SUM(CASE WHEN status = 'canceled' THEN 1 ELSE 0 END)
                        AS canceled_count,
                    SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END)
                        AS running_count
                FROM jobs
                WHERE parent_job_id IS NOT NULL
                GROUP BY parent_job_id
            ),
            effective_parent_jobs AS (
                SELECT
                    p.*,
                    CASE
                        WHEN child_status.child_count IS NULL
                            THEN p.status
                        WHEN child_status.completed_count = child_status.child_count
                            THEN 'completed'
                        WHEN child_status.terminal_count = child_status.child_count
                            AND child_status.failed_count > 0
                            THEN 'failed'
                        WHEN child_status.terminal_count = child_status.child_count
                            AND child_status.canceled_count > 0
                            THEN 'canceled'
                        WHEN child_status.terminal_count = child_status.child_count
                            THEN 'completed'
                        WHEN child_status.running_count > 0
                            OR child_status.terminal_count > 0
                            THEN 'running'
                        ELSE 'queued'
                    END AS effective_status
                FROM jobs AS p
                LEFT JOIN child_status ON child_status.parent_job_id = p.job_id
                WHERE p.parent_job_id IS NULL AND p.created_by = :user_id
            )
            SELECT * FROM effective_parent_jobs
            """
        params: dict = {"user_id": user_id, "limit": limit, "offset": offset}
        if status is not None:
            query += " WHERE effective_status = :status"
            params["status"] = status
        query += " ORDER BY created_at DESC, job_id DESC LIMIT :limit OFFSET :offset"

        result = conn.execute(text(query), params)
        rows = []
        for row in result.mappings().fetchall():
            decoded = _ReaderOperations._decode_job(dict(row))
            decoded.pop("effective_status", None)
            rows.append(decoded)
        return rows

    @staticmethod
    def get_child_jobs(conn: Connection, parent_job_id: str) -> list[dict]:
        result = conn.execute(
            text(
                "SELECT * FROM jobs WHERE parent_job_id = :pid "
                "ORDER BY created_at ASC, job_id ASC"
            ),
            {"pid": parent_job_id},
        )
        return [
            _ReaderOperations._decode_job(dict(row))
            for row in result.mappings().fetchall()
        ]

    @staticmethod
    def get_run_siblings(
        conn: Connection,
        run_id: str,
        user_id: str,
    ) -> list[dict]:
        child = (
            conn.execute(
                text(
                    "SELECT parent_job_id FROM jobs "
                    "WHERE run_id = :rid AND created_by = :uid LIMIT 1"
                ),
                {"rid": run_id, "uid": user_id},
            )
            .mappings()
            .fetchone()
        )
        if child is None or child["parent_job_id"] is None:
            return []

        result = conn.execute(
            text(
                "SELECT job_id, parent_job_id, run_id, status, video_name "
                "FROM jobs WHERE parent_job_id = :pid AND run_id IS NOT NULL "
                "AND created_by = :uid "
                "ORDER BY created_at ASC, job_id ASC"
            ),
            {"pid": child["parent_job_id"], "uid": user_id},
        )
        return [dict(row) for row in result.mappings().fetchall()]

    @staticmethod
    def run_owner(conn: Connection, run_id: str) -> str | None:
        row = (
            conn.execute(
                text("SELECT created_by FROM jobs WHERE run_id = :rid LIMIT 1"),
                {"rid": run_id},
            )
            .mappings()
            .fetchone()
        )
        return row["created_by"] if row else None

    @staticmethod
    def get_video_by_hash(
        conn: Connection,
        user_id: str,
        content_hash: str,
    ) -> dict | None:
        result = conn.execute(
            text("SELECT * FROM videos " "WHERE user_id = :uid AND content_hash = :h"),
            {"uid": user_id, "h": content_hash},
        )
        row = result.mappings().fetchone()
        return dict(row) if row else None

    @staticmethod
    def get_video_by_key(
        conn: Connection,
        user_id: str,
        stored_key: str,
    ) -> dict | None:
        result = conn.execute(
            text("SELECT * FROM videos WHERE user_id = :uid AND stored_key = :k"),
            {"uid": user_id, "k": stored_key},
        )
        row = result.mappings().fetchone()
        return dict(row) if row else None

    @staticmethod
    def list_videos(conn: Connection, user_id: str) -> list[dict]:
        result = conn.execute(
            text(
                "SELECT * FROM videos WHERE user_id = :uid "
                "ORDER BY created_at DESC, stored_key DESC"
            ),
            {"uid": user_id},
        )
        return [dict(row) for row in result.mappings().fetchall()]

    @staticmethod
    def _decode_job(row: dict) -> dict:
        """Decode ``params_json`` when a driver returns it as raw text."""
        raw = row.get("params_json")
        if isinstance(raw, str):
            try:
                row["params_json"] = json.loads(raw)
            except (ValueError, TypeError):
                pass
        return row


class ClipScribeAsyncReaderDB(ClipScribeAsyncBaseDB):
    def __init__(self, async_engine: AsyncEngine) -> None:
        super().__init__(async_engine)
        self.reader_op = _ReaderOperations()
        logger.info("ClipScribeAsyncReaderDB ready")

    async def get_run(self, run_id: str) -> dict | None:
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(self.reader_op.get_run, run_id)

    async def get_latest_run(self) -> dict | None:
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(self.reader_op.get_latest_run)

    async def get_global_stats(self, run_id: str) -> dict | None:
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(self.reader_op.get_global_stats, run_id)

    async def get_audio_segments(
        self,
        run_id: str,
        max_start_time: float | None = None,
    ) -> list[dict]:
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(
                self.reader_op.get_audio_segments,
                run_id,
                max_start_time,
            )

    async def get_text_events(
        self,
        run_id: str,
        max_second: int | None = None,
    ) -> list[dict]:
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(
                self.reader_op.get_text_events,
                run_id,
                max_second,
            )

    async def get_visual_objects(
        self,
        run_id: str,
        label_contains: str | None = None,
        max_lifespan_start: float | None = None,
    ) -> list[dict]:
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(
                self.reader_op.get_visual_objects,
                run_id,
                label_contains,
                max_lifespan_start,
            )

    async def get_field_descriptions(
        self,
        table_name: str | None = None,
    ) -> list[dict]:
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(
                self.reader_op.get_field_descriptions,
                table_name,
            )

    async def get_scene_descriptions(
        self,
        run_id: str,
        max_start_time: float | None = None,
        max_end_time: float | None = None,
    ) -> list[dict]:
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(
                self.reader_op.get_scene_descriptions,
                run_id,
                max_start_time,
                max_end_time,
            )

    async def list_runs(self, limit: int = 50, offset: int = 0) -> list[dict]:
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(self.reader_op.list_runs, limit, offset)

    async def get_shot_boundaries(self, run_id: str) -> list[dict]:
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(self.reader_op.get_shot_boundaries, run_id)

    async def get_frame_detections(
        self,
        run_id: str,
        from_sec: float | None = None,
        to_sec: float | None = None,
    ) -> list[dict]:
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(
                self.reader_op.get_frame_detections,
                run_id,
                from_sec,
                to_sec,
            )

    async def get_parser_results(self, run_id: str) -> list[dict]:
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(self.reader_op.get_parser_results, run_id)

    async def get_chat_messages(self, run_id: str, session_id: str) -> list[dict]:
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(
                self.reader_op.get_chat_messages,
                "run_id",
                run_id,
                session_id,
            )

    async def get_job_chat_messages(
        self,
        job_id: str,
        session_id: str,
    ) -> list[dict]:
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(
                self.reader_op.get_chat_messages,
                "job_id",
                job_id,
                session_id,
            )

    async def get_chat_sessions(self, run_id: str) -> list[dict]:
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(
                self.reader_op.get_chat_sessions,
                "run_id",
                run_id,
            )

    async def get_job_chat_sessions(self, job_id: str) -> list[dict]:
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(
                self.reader_op.get_chat_sessions,
                "job_id",
                job_id,
            )

    async def get_job(self, job_id: str) -> dict | None:
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(self.reader_op.get_job, job_id)

    async def list_jobs(
        self,
        user_id: str,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(
                self.reader_op.list_jobs,
                user_id,
                status,
                limit,
                offset,
            )

    async def list_parent_jobs(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
    ) -> list[dict]:
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(
                self.reader_op.list_parent_jobs,
                user_id,
                limit,
                offset,
                status,
            )

    async def get_child_jobs(self, parent_job_id: str) -> list[dict]:
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(
                self.reader_op.get_child_jobs,
                parent_job_id,
            )

    async def get_run_siblings(self, run_id: str, user_id: str) -> list[dict]:
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(
                self.reader_op.get_run_siblings,
                run_id,
                user_id,
            )

    async def run_owner(self, run_id: str) -> str | None:
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(self.reader_op.run_owner, run_id)

    async def get_video_by_hash(
        self,
        user_id: str,
        content_hash: str,
    ) -> dict | None:
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(
                self.reader_op.get_video_by_hash,
                user_id,
                content_hash,
            )

    async def get_video_by_key(
        self,
        user_id: str,
        stored_key: str,
    ) -> dict | None:
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(
                self.reader_op.get_video_by_key,
                user_id,
                stored_key,
            )

    async def list_videos(self, user_id: str) -> list[dict]:
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(self.reader_op.list_videos, user_id)


class ClipScribeReaderDB(ClipScribeBaseDB):
    def __init__(self, engine: Engine) -> None:
        super().__init__(engine)
        self.reader_op = _ReaderOperations()
        logger.info("ClipScribeReaderDB ready")

    def get_run(self, run_id: str) -> dict | None:
        """Fetch a specific run by its run_id."""
        with self._engine.connect() as conn:
            return self.reader_op.get_run(conn, run_id)

    def get_latest_run(self) -> dict | None:
        """Fetch the most recent run from the runs table."""
        with self._engine.connect() as conn:
            return self.reader_op.get_latest_run(conn)

    def get_global_stats(self, run_id: str) -> dict | None:
        """Fetch global statistics for a specific run."""
        with self._engine.connect() as conn:
            return self.reader_op.get_global_stats(conn, run_id)

    def get_audio_segments(
        self,
        run_id: str,
        max_start_time: float | None = None,
    ) -> list[dict]:
        """Fetch audio transcript segments for a run."""
        with self._engine.connect() as conn:
            return self.reader_op.get_audio_segments(conn, run_id, max_start_time)

    def get_text_events(
        self,
        run_id: str,
        max_second: int | None = None,
    ) -> list[dict]:
        """Fetch OCR text events for a run."""
        with self._engine.connect() as conn:
            return self.reader_op.get_text_events(conn, run_id, max_second)

    def get_visual_objects(
        self,
        run_id: str,
        label_contains: str | None = None,
        max_lifespan_start: float | None = None,
    ) -> list[dict]:
        """Fetch visual object occurrences for a run."""
        with self._engine.connect() as conn:
            return self.reader_op.get_visual_objects(
                conn,
                run_id,
                label_contains,
                max_lifespan_start,
            )

    def get_field_descriptions(self, table_name: str | None = None) -> list[dict]:
        """Fetch field descriptions from the field_descriptions table."""
        with self._engine.connect() as conn:
            return self.reader_op.get_field_descriptions(conn, table_name)

    def get_scene_descriptions(
        self,
        run_id: str,
        max_start_time: float | None = None,
        max_end_time: float | None = None,
    ) -> list[dict]:
        """Fetch scene descriptions for a run."""
        with self._engine.connect() as conn:
            return self.reader_op.get_scene_descriptions(
                conn,
                run_id,
                max_start_time,
                max_end_time,
            )

    def list_runs(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """List runs, most recent first (jobs / run-history view)."""
        with self._engine.connect() as conn:
            return self.reader_op.list_runs(conn, limit, offset)

    def get_shot_boundaries(self, run_id: str) -> list[dict]:
        """Fetch per-shot temporal boundaries for a run (timeline view)."""
        with self._engine.connect() as conn:
            return self.reader_op.get_shot_boundaries(conn, run_id)

    def get_frame_detections(
        self,
        run_id: str,
        from_sec: float | None = None,
        to_sec: float | None = None,
    ) -> list[dict]:
        """Fetch raw frame detections, optionally within a time window."""
        with self._engine.connect() as conn:
            return self.reader_op.get_frame_detections(
                conn,
                run_id,
                from_sec,
                to_sec,
            )

    def get_parser_results(self, run_id: str) -> list[dict]:
        """Fetch persisted per-criterion parser evaluations for a run."""
        with self._engine.connect() as conn:
            return self.reader_op.get_parser_results(conn, run_id)

    def get_chat_messages(self, run_id: str, session_id: str) -> list[dict]:
        """Fetch a per-run chat session transcript, oldest first."""
        with self._engine.connect() as conn:
            return self.reader_op.get_chat_messages(
                conn,
                "run_id",
                run_id,
                session_id,
            )

    def get_job_chat_messages(self, job_id: str, session_id: str) -> list[dict]:
        """Fetch a job-level chat session transcript, oldest first."""
        with self._engine.connect() as conn:
            return self.reader_op.get_chat_messages(
                conn,
                "job_id",
                job_id,
                session_id,
            )

    def get_chat_sessions(self, run_id: str) -> list[dict]:
        """List per-run chat sessions, most recently active first."""
        with self._engine.connect() as conn:
            return self.reader_op.get_chat_sessions(conn, "run_id", run_id)

    def get_job_chat_sessions(self, job_id: str) -> list[dict]:
        """List job-level chat sessions, most recently active first."""
        with self._engine.connect() as conn:
            return self.reader_op.get_chat_sessions(conn, "job_id", job_id)

    def get_job(self, job_id: str) -> dict | None:
        """Fetch a single orchestration job by id."""
        with self._engine.connect() as conn:
            return self.reader_op.get_job(conn, job_id)

    def list_jobs(
        self,
        user_id: str,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """List a user's jobs, newest first and optionally by status."""
        with self._engine.connect() as conn:
            return self.reader_op.list_jobs(conn, user_id, status, limit, offset)

    def list_parent_jobs(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
    ) -> list[dict]:
        """List a user's top-level jobs using their effective status."""
        with self._engine.connect() as conn:
            return self.reader_op.list_parent_jobs(
                conn,
                user_id,
                limit,
                offset,
                status,
            )

    def get_child_jobs(self, parent_job_id: str) -> list[dict]:
        """List child runs of a batch job in submission order."""
        with self._engine.connect() as conn:
            return self.reader_op.get_child_jobs(conn, parent_job_id)

    def get_run_siblings(self, run_id: str, user_id: str) -> list[dict]:
        """List sibling runs from the same user-owned batch."""
        with self._engine.connect() as conn:
            return self.reader_op.get_run_siblings(conn, run_id, user_id)

    def run_owner(self, run_id: str) -> str | None:
        """Return the owner of the job that produced a run."""
        with self._engine.connect() as conn:
            return self.reader_op.run_owner(conn, run_id)

    def get_video_by_hash(self, user_id: str, content_hash: str) -> dict | None:
        """Look up a registered video by its content hash."""
        with self._engine.connect() as conn:
            return self.reader_op.get_video_by_hash(conn, user_id, content_hash)

    def get_video_by_key(self, user_id: str, stored_key: str) -> dict | None:
        """Look up a registered video by its storage key."""
        with self._engine.connect() as conn:
            return self.reader_op.get_video_by_key(conn, user_id, stored_key)

    def list_videos(self, user_id: str) -> list[dict]:
        """List all registered videos for a user, newest first."""
        with self._engine.connect() as conn:
            return self.reader_op.list_videos(conn, user_id)
