import sqlite3
import json
import uuid
import logging
from pathlib import Path


class ClipScribeBaseDB:
    def __init__(self, db_path: str | Path, logger: logging.Logger):
        self.logger = logger
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def close(self) -> None:
        self._conn.close()
        self.logger.info("ClipScribeDB connection closed.")


class ClipScribeReaderDB(ClipScribeBaseDB):
    def __init__(self, db_path: str | Path, logger: logging.Logger):
        super().__init__(db_path, logger)
        self._conn.row_factory = sqlite3.Row

    def get_latest_run(self) -> dict | None:
        """Fetch the most recent run from the runs table."""
        cursor = self._conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT 1"
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_global_stats(self, run_id: str) -> dict | None:
        """Fetch global statistics for a specific run."""
        cursor = self._conn.execute(
            "SELECT * FROM global_stats WHERE run_id = ?", (run_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        stats = dict(row)
        # Parse JSON fields
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
        """
        Fetch audio transcript segments for a run.
        Optionally filter by max_start_time (e.g., first 5 seconds).
        """
        query = "SELECT * FROM audio_segments WHERE run_id = ?"
        params: list[str | float] = [run_id]

        if max_start_time is not None:
            query += " AND start_time < ?"
            params.append(max_start_time)

        query += " ORDER BY start_time"

        cursor = self._conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_text_events(self, run_id: str, max_second: int | None = None) -> list[dict]:
        """
        Fetch OCR text events for a run.
        Optionally filter by max_second (e.g., first 5 seconds).
        """
        query = "SELECT * FROM text_events WHERE run_id = ?"
        params: list[str | int] = [run_id]

        if max_second is not None:
            query += " AND second < ?"
            params.append(max_second)

        query += " ORDER BY second, line_index"

        cursor = self._conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_visual_objects(
        self,
        run_id: str,
        label_contains: str | None = None,
        max_lifespan_start: float | None = None,
    ) -> list[dict]:
        """
        Fetch visual object occurrences for a run.
        Optionally filter by label (fuzzy match) and max_lifespan_start.
        """
        query = "SELECT * FROM visual_object_occurrences WHERE run_id = ?"
        params: list[str | float] = [run_id]

        if label_contains is not None:
            query += " AND label LIKE ?"
            params.append(f"%{label_contains}%")

        if max_lifespan_start is not None:
            query += " AND lifespan_start < ?"
            params.append(max_lifespan_start)

        query += " ORDER BY lifespan_start"

        cursor = self._conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


class ClipScribeWriterDB(ClipScribeBaseDB):
    def __init__(self, db_path: str | Path, logger: logging.Logger):
        super().__init__(db_path, logger)

        self._create_tables()
        self.logger.info(f"ClipScribeDB connected at {self.db_path}")

    def _create_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id      TEXT PRIMARY KEY,
                video_name  TEXT,
                video_path  TEXT,
                video_type  TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS global_stats (
                id                              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id                          TEXT NOT NULL REFERENCES runs(run_id),
                total_shots                     INTEGER,
                video_duration                  REAL,
                avg_shot_duration               REAL,
                dynamic_start_detected          INTEGER,
                dynamic_start_first_shot_dur    REAL,
                dynamic_start_criteria          TEXT,
                qp_intro_detected               INTEGER,
                qp_intro_shot_count             INTEGER,
                qp_intro_shots                  TEXT,
                qp_intro_criteria               TEXT,
                qp_general_detected             INTEGER,
                qp_general_rapid_fire_segments  TEXT,
                qp_general_criteria             TEXT
            );

            CREATE TABLE IF NOT EXISTS visual_object_occurrences (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id              TEXT NOT NULL REFERENCES runs(run_id),
                global_id           INTEGER,
                label               TEXT,
                shot_index          INTEGER,
                lifespan_start      REAL,
                lifespan_end        REAL,
                screen_coverage     REAL,
                velocity_px_sec     REAL,
                growth_factor       REAL,
                direction           TEXT,
                centrality_score    REAL,
                screen_time_ratio   REAL,
                quadrant            TEXT
            );

            CREATE TABLE IF NOT EXISTS text_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      TEXT NOT NULL REFERENCES runs(run_id),
                second      INTEGER,
                line_index  INTEGER,
                text        TEXT
            );

            CREATE TABLE IF NOT EXISTS audio_segments (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      TEXT NOT NULL REFERENCES runs(run_id),
                start_time  REAL,
                end_time    REAL,
                text        TEXT,
                confidence  REAL
            );

            CREATE TABLE IF NOT EXISTS field_descriptions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name   TEXT NOT NULL,
                column_name  TEXT NOT NULL,
                description  TEXT NOT NULL,
                UNIQUE(table_name, column_name)
            );
            """
        )

    def save_run(
        self,
        video_name: str,
        video_path: str,
        video_type: str,
        video_metadata: dict,
    ) -> str:
        run_id = str(uuid.uuid4())

        with self._conn:
            self._insert_run(run_id, video_name, video_path, video_type)
            self._insert_global_stats(run_id, video_metadata.get("global_stats", {}))
            self._insert_visual_objects(
                run_id, video_metadata.get("visual_objects", [])
            )
            self._insert_text_events(run_id, video_metadata.get("text_events", []))
            self._insert_audio_segments(
                run_id, video_metadata.get("audio_segments", [])
            )

        self.logger.info(f"Saved run {run_id} for '{video_name}'")
        return run_id

    def _insert_run(
        self,
        run_id: str,
        video_name: str,
        video_path: str,
        video_type: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO runs (run_id, video_name, video_path, video_type)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, video_name, video_path, video_type),
        )

    def _insert_global_stats(self, run_id: str, global_stats: dict) -> None:
        ds = global_stats.get("dynamic_start", {})
        qpi = global_stats.get("quick_pacing_intro", {})
        qpg = global_stats.get("quick_pacing_general", {})

        self._conn.execute(
            """
            INSERT INTO global_stats (
                run_id,
                total_shots, video_duration, avg_shot_duration,
                dynamic_start_detected, dynamic_start_first_shot_dur, dynamic_start_criteria,
                qp_intro_detected, qp_intro_shot_count, qp_intro_shots, qp_intro_criteria,
                qp_general_detected, qp_general_rapid_fire_segments, qp_general_criteria
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                global_stats.get("total_shots"),
                global_stats.get("video_duration"),
                global_stats.get("avg_shot_duration"),
                int(ds.get("detected", False)),
                ds.get("first_shot_duration"),
                ds.get("criteria"),
                int(qpi.get("detected", False)),
                qpi.get("shot_count"),
                json.dumps(qpi.get("shots", [])),
                qpi.get("criteria"),
                int(qpg.get("detected", False)),
                json.dumps(qpg.get("rapid_fire_segments", [])),
                qpg.get("criteria"),
            ),
        )

    def _insert_visual_objects(self, run_id: str, visual_objects: list) -> None:
        rows = []
        for obj in visual_objects:
            g_id = obj.get("global_id")
            label = obj.get("label")
            for occ in obj.get("occurrences", []):
                lifespan = occ.get("lifespan", [None, None])
                rows.append(
                    (
                        run_id,
                        g_id,
                        label,
                        occ.get("shot_index"),
                        lifespan[0] if len(lifespan) > 0 else None,
                        lifespan[1] if len(lifespan) > 1 else None,
                        occ.get("screen_coverage"),
                        occ.get("velocity_px_sec"),
                        occ.get("growth_factor"),
                        occ.get("direction"),
                        occ.get("centrality_score"),
                        occ.get("screen_time_ratio"),
                        occ.get("quadrant"),
                    )
                )

        if rows:
            self._conn.executemany(
                """
                INSERT INTO visual_object_occurrences (
                    run_id, global_id, label, shot_index,
                    lifespan_start, lifespan_end,
                    screen_coverage, velocity_px_sec, growth_factor,
                    direction, centrality_score, screen_time_ratio, quadrant
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def _insert_text_events(self, run_id: str, text_events: list) -> None:
        rows = []
        for event in text_events:
            second = event.get("second")
            for line_index, line in enumerate(event.get("text", [])):
                rows.append((run_id, second, line_index, line))

        if rows:
            self._conn.executemany(
                "INSERT INTO text_events (run_id, second, line_index, text) VALUES (?, ?, ?, ?)",
                rows,
            )

    def _insert_audio_segments(self, run_id: str, audio_segments: list) -> None:
        rows = [
            (
                run_id,
                seg.get("start"),
                seg.get("end"),
                seg.get("text"),
                seg.get("confidence"),
            )
            for seg in audio_segments
        ]

        if rows:
            self._conn.executemany(
                "INSERT INTO audio_segments (run_id, start_time, end_time, text, confidence) VALUES (?, ?, ?, ?, ?)",
                rows,
            )

    def save_field_descriptions(self, descriptions: dict) -> None:
        """
        Persist field descriptions to the field_descriptions table.
        Idempotent: uses INSERT OR IGNORE to safely skip existing entries.

        Args:
            descriptions: Flat dict {table_name: {column_name: description}}
        """
        rows = []
        for table_name, columns in descriptions.items():
            for column_name, description in columns.items():
                rows.append((table_name, column_name, description))

        if rows:
            with self._conn:
                self._conn.executemany(
                    "INSERT OR IGNORE INTO field_descriptions (table_name, column_name, description) VALUES (?, ?, ?)",
                    rows,
                )
            self.logger.info(f"Saved {len(rows)} field descriptions to DB.")
