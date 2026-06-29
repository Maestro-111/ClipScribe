"""Declarative schema definitions for the ClipScribe database."""

from sqlalchemy import (
    MetaData,
    Table,
    Column,
    Integer,
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
