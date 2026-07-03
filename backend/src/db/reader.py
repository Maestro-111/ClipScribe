"""Read-only database access for ClipScribe."""

import json
import logging

from sqlalchemy import Engine, text

from .engine import ClipScribeBaseDB

logger = logging.getLogger("clip_scribe")


class ClipScribeReaderDB(ClipScribeBaseDB):
    def __init__(self, engine: Engine):
        super().__init__(engine)
        logger.info("ClipScribeReaderDB ready.")

    def get_run(self, run_id: str) -> dict | None:
        """Fetch a specific run by its run_id."""
        with self._engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM runs WHERE run_id = :run_id"),
                {"run_id": run_id},
            )
            row = result.mappings().fetchone()
            return dict(row) if row else None

    def get_latest_run(self) -> dict | None:
        """Fetch the most recent run from the runs table."""
        with self._engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM runs ORDER BY created_at DESC LIMIT 1")
            )
            row = result.mappings().fetchone()
            return dict(row) if row else None

    def get_global_stats(self, run_id: str) -> dict | None:
        """Fetch global statistics for a specific run."""
        with self._engine.connect() as conn:
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

    def get_audio_segments(
        self, run_id: str, max_start_time: float | None = None
    ) -> list[dict]:
        """Fetch audio transcript segments for a run."""
        query = "SELECT * FROM audio_segments WHERE run_id = :run_id"
        params: dict = {"run_id": run_id}

        if max_start_time is not None:
            query += " AND start_time < :max_start_time"
            params["max_start_time"] = max_start_time

        query += " ORDER BY start_time"

        with self._engine.connect() as conn:
            result = conn.execute(text(query), params)
            return [dict(row) for row in result.mappings().fetchall()]

    def get_text_events(self, run_id: str, max_second: int | None = None) -> list[dict]:
        """Fetch OCR text events for a run."""
        query = "SELECT * FROM text_events WHERE run_id = :run_id"
        params: dict = {"run_id": run_id}

        if max_second is not None:
            query += " AND second < :max_second"
            params["max_second"] = max_second

        query += " ORDER BY second, line_index"

        with self._engine.connect() as conn:
            result = conn.execute(text(query), params)
            return [dict(row) for row in result.mappings().fetchall()]

    def get_visual_objects(
        self,
        run_id: str,
        label_contains: str | None = None,
        max_lifespan_start: float | None = None,
    ) -> list[dict]:
        """Fetch visual object occurrences for a run."""
        query = "SELECT * FROM visual_object_occurrences WHERE run_id = :run_id"
        params: dict = {"run_id": run_id}

        if label_contains is not None:
            # Use ILIKE on PostgreSQL for case-insensitive match, LIKE on SQLite
            if self._engine.dialect.name == "postgresql":
                query += " AND label ILIKE :label_pattern"
            else:
                query += " AND label LIKE :label_pattern"
            params["label_pattern"] = f"%{label_contains}%"

        if max_lifespan_start is not None:
            query += " AND lifespan_start < :max_lifespan_start"
            params["max_lifespan_start"] = max_lifespan_start

        query += " ORDER BY lifespan_start"

        with self._engine.connect() as conn:
            result = conn.execute(text(query), params)
            return [dict(row) for row in result.mappings().fetchall()]

    def get_field_descriptions(self, table_name: str | None = None) -> list[dict]:
        """Fetch field descriptions from the field_descriptions table."""
        query = "SELECT table_name, column_name, description FROM field_descriptions"
        params: dict = {}

        if table_name is not None:
            query += " WHERE table_name = :table_name"
            params["table_name"] = table_name

        query += " ORDER BY table_name, column_name"

        with self._engine.connect() as conn:
            result = conn.execute(text(query), params)
            return [dict(row) for row in result.mappings().fetchall()]

    def get_scene_descriptions(
        self,
        run_id: str,
        max_start_time: float | None = None,
        max_end_time: float | None = None,
    ) -> list[dict]:
        """Fetch scene descriptions for a run."""
        query = "SELECT * FROM scene_descriptions WHERE run_id = :run_id"
        params: dict = {"run_id": run_id}

        if max_start_time is not None:
            query += " AND start_time < :max_start_time"
            params["max_start_time"] = max_start_time

        if max_end_time is not None:
            query += " AND end_time < :max_end_time"
            params["max_end_time"] = max_end_time

        query += " ORDER BY start_time"

        with self._engine.connect() as conn:
            result = conn.execute(text(query), params)
            return [dict(row) for row in result.mappings().fetchall()]

    def list_runs(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """List runs, most recent first (jobs / run-history view)."""
        with self._engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT * FROM runs ORDER BY created_at DESC "
                    "LIMIT :limit OFFSET :offset"
                ),
                {"limit": limit, "offset": offset},
            )
            return [dict(row) for row in result.mappings().fetchall()]

    def get_shot_boundaries(self, run_id: str) -> list[dict]:
        """Fetch per-shot temporal boundaries for a run (timeline view)."""
        with self._engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT * FROM shot_boundaries WHERE run_id = :run_id "
                    "ORDER BY shot_index"
                ),
                {"run_id": run_id},
            )
            return [dict(row) for row in result.mappings().fetchall()]

    def get_frame_detections(
        self,
        run_id: str,
        from_sec: float | None = None,
        to_sec: float | None = None,
    ) -> list[dict]:
        """Fetch raw frame detections for a run, optionally within a time window.

        Backs the inspector overlay: the frontend pulls detections for a run
        (optionally a ``[from_sec, to_sec]`` playback window) and draws boxes.
        """
        query = "SELECT * FROM frame_detections WHERE run_id = :run_id"
        params: dict = {"run_id": run_id}

        if from_sec is not None:
            query += " AND timestamp_sec >= :from_sec"
            params["from_sec"] = from_sec
        if to_sec is not None:
            query += " AND timestamp_sec <= :to_sec"
            params["to_sec"] = to_sec

        query += " ORDER BY frame_idx, id"

        with self._engine.connect() as conn:
            result = conn.execute(text(query), params)
            return [dict(row) for row in result.mappings().fetchall()]

    def get_parser_results(self, run_id: str) -> list[dict]:
        """Fetch persisted per-criterion parser evaluations for a run."""
        with self._engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT * FROM parser_results WHERE run_id = :run_id "
                    "ORDER BY feature_category, feature_name"
                ),
                {"run_id": run_id},
            )
            return [dict(row) for row in result.mappings().fetchall()]

    def get_job(self, job_id: str) -> dict | None:
        """Fetch a single orchestration job by id."""
        with self._engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM jobs WHERE job_id = :job_id"),
                {"job_id": job_id},
            )
            row = result.mappings().fetchone()
            return self._decode_job(dict(row)) if row else None

    def list_jobs(
        self, status: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[dict]:
        """List jobs, most recent first, optionally filtered by status."""
        query = "SELECT * FROM jobs"
        params: dict = {"limit": limit, "offset": offset}

        if status is not None:
            query += " WHERE status = :status"
            params["status"] = status

        query += " ORDER BY created_at DESC, job_id DESC LIMIT :limit OFFSET :offset"

        with self._engine.connect() as conn:
            result = conn.execute(text(query), params)
            return [self._decode_job(dict(row)) for row in result.mappings().fetchall()]

    @staticmethod
    def _decode_job(row: dict) -> dict:
        """Decode ``params_json`` to a dict when the driver returns raw text.

        Raw ``SELECT *`` bypasses the JSON column type, so SQLite (and some
        drivers) hand back the stored JSON as a string.
        """
        raw = row.get("params_json")
        if isinstance(raw, str):
            try:
                row["params_json"] = json.loads(raw)
            except (ValueError, TypeError):
                pass
        return row
