"""Read-only database access for ClipScribe."""

import json
import logging

from sqlalchemy import Engine, text

from .engine import ClipScribeBaseDB


class ClipScribeReaderDB(ClipScribeBaseDB):
    def __init__(self, engine: Engine, logger: logging.Logger):
        super().__init__(engine, logger)
        self.logger.info("ClipScribeReaderDB ready.")

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
