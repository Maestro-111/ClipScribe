"""Declarative schema definitions for the ClipScribe database."""

from sqlalchemy import (
    MetaData,
    Table,
    Column,
    Boolean,
    Index,
    Integer,
    JSON,
    Text,
    Float,
    UniqueConstraint,
    text,
)

metadata_obj = MetaData()

runs_table = Table(
    "runs",
    metadata_obj,
    Column("run_id", Text, primary_key=True),
    Column("video_name", Text),
    Column("video_path", Text),
    Column("video_type", Text),
    Column("created_at", Text, server_default=text("CURRENT_TIMESTAMP")),
)

global_stats_table = Table(
    "global_stats",
    metadata_obj,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", Text, nullable=False),
    Column("total_shots", Integer),
    Column("video_duration", Float),
    Column("avg_shot_duration", Float),
    Column("dynamic_start_detected", Integer),
    Column("dynamic_start_first_shot_dur", Float),
    Column("dynamic_start_criteria", Text),
    Column("qp_intro_detected", Integer),
    Column("qp_intro_shot_count", Integer),
    Column("qp_intro_shots", Text),
    Column("qp_intro_criteria", Text),
    Column("qp_general_detected", Integer),
    Column("qp_general_rapid_fire_segments", Text),
    Column("qp_general_criteria", Text),
)

visual_object_occurrences_table = Table(
    "visual_object_occurrences",
    metadata_obj,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", Text, nullable=False),
    Column("global_id", Integer),
    Column("label", Text),
    Column("shot_index", Integer),
    Column("lifespan_start", Float),
    Column("lifespan_end", Float),
    Column("screen_coverage", Float),
    Column("velocity_px_sec", Float),
    Column("growth_factor", Float),
    Column("direction", Text),
    Column("centrality_score", Float),
    Column("screen_time_ratio", Float),
    Column("quadrant", Text),
)

text_events_table = Table(
    "text_events",
    metadata_obj,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", Text, nullable=False),
    Column("second", Integer),
    Column("line_index", Integer),
    Column("text", Text),
)

audio_segments_table = Table(
    "audio_segments",
    metadata_obj,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", Text, nullable=False),
    Column("start_time", Float),
    Column("end_time", Float),
    Column("text", Text),
    Column("confidence", Float),
)

scene_descriptions_table = Table(
    "scene_descriptions",
    metadata_obj,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", Text, nullable=False),
    Column("shot_index", Integer, nullable=False),
    Column("start_time", Float),
    Column("end_time", Float),
    Column("description", Text),
)

field_descriptions_table = Table(
    "field_descriptions",
    metadata_obj,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("table_name", Text, nullable=False),
    Column("column_name", Text, nullable=False),
    Column("description", Text, nullable=False),
    UniqueConstraint("table_name", "column_name"),
)

# ---------------------------------------------------------------------------
# Web-app tables (see docs/web-app-plan.md §4). Orchestration state, raw
# per-frame detections for the UI overlay, persisted parser output, and
# per-shot temporal boundaries.
# ---------------------------------------------------------------------------

jobs_table = Table(
    "jobs",
    metadata_obj,
    # ULID, also exposed in API URLs.
    Column("job_id", Text, primary_key=True),
    # Self-FK to the parent batch job. A job is either a parent (this is NULL,
    # run_id NULL — a pure container fanned out over N videos) or a child (this
    # points at the parent, run_id set — one video, executed like a solo job).
    # Every executed run therefore has a parent (docs/deployment.md §2.1).
    Column("parent_job_id", Text),
    # Populated when the extractor writes the run; null until then (or on
    # failure before a run exists). Always NULL on a parent row.
    Column("run_id", Text),
    # queued | running | completed | failed | canceled
    Column("status", Text, nullable=False, server_default=text("'queued'")),
    Column("celery_task_id", Text),
    # full | extract | parse
    Column("mode", Text),
    Column("video_name", Text),
    Column("video_path", Text),
    Column("video_type", Text),
    Column("device", Text),
    Column("platform", Text),
    # Full request payload for reproducibility.
    Column("params_json", JSON),
    Column("error_text", Text),
    Column("created_at", Text, server_default=text("CURRENT_TIMESTAMP")),
    Column("started_at", Text),
    Column("finished_at", Text),
    # Nullable until auth lands.
    Column("created_by", Text),
    # Fetch a parent's children (batch fan-out) in one indexed lookup.
    Index("ix_jobs_parent_job_id", "parent_job_id"),
)

frame_detections_table = Table(
    "frame_detections",
    metadata_obj,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", Text, nullable=False),
    Column("shot_index", Integer),
    Column("frame_idx", Integer),
    Column("timestamp_sec", Float),
    # dino | ocr | mtcnn | sam_mask
    Column("source", Text),
    # Resolved taxonomy label, or null.
    Column("label", Text),
    # OCR text, or null.
    Column("text", Text),
    Column("box_x1", Float),
    Column("box_y1", Float),
    Column("box_x2", Float),
    Column("box_y2", Float),
    Column("confidence", Float),
    # Final global visual object id for tracked SAM detections.
    Column("object_id", Integer),
    Index("ix_frame_detections_run_frame", "run_id", "frame_idx"),
    Index("ix_frame_detections_run_object", "run_id", "object_id"),
)

parser_results_table = Table(
    "parser_results",
    metadata_obj,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", Text, nullable=False),
    Column("platform", Text),
    Column("feature_category", Text),
    Column("feature_name", Text),
    Column("feature_criteria", Text),
    Column("evaluation", Boolean),
    Column("llm_prompt", Text),
    Column("llm_explanation", Text),
    # Link to LangSmith trace.
    Column("langsmith_run_id", Text),
    Column("created_at", Text, server_default=text("CURRENT_TIMESTAMP")),
)

shot_boundaries_table = Table(
    "shot_boundaries",
    metadata_obj,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", Text, nullable=False),
    Column("shot_index", Integer),
    Column("start_sec", Float),
    Column("end_sec", Float),
    Column("duration_sec", Float),
)

# User-facing transcript for the advisory chat agent (web-app-plan §13). The
# agent fetches run data on demand via read-only tools; this table just stores
# the conversation so the UI can list sessions and replay history.
#
# A message is scoped to exactly one of two chats: the per-run inspector chat
# (``run_id`` set, ``job_id`` NULL) or the job-level chat that analyzes every run
# in a batch job (``job_id`` set, ``run_id`` NULL).
chat_messages_table = Table(
    "chat_messages",
    metadata_obj,
    Column("id", Integer, primary_key=True, autoincrement=True),
    # Set for the per-run inspector chat; NULL for job-level chat.
    Column("run_id", Text),
    # Set for the job-level chat; NULL for the per-run chat.
    Column("job_id", Text),
    # Groups messages into one conversation; == the LangGraph thread id.
    Column("session_id", Text, nullable=False),
    # user | assistant
    Column("role", Text, nullable=False),
    Column("content", Text, nullable=False),
    # Optional: tool names the assistant invoked, for UI transparency.
    Column("tool_calls_json", JSON),
    Column("created_at", Text, server_default=text("CURRENT_TIMESTAMP")),
    Index("ix_chat_messages_run_session", "run_id", "session_id"),
    Index("ix_chat_messages_job_session", "job_id", "session_id"),
)
