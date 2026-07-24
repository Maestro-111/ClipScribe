"""add run_id and hot-path indexes

Adds indexes that back the run-inspection reads and job-orchestration queries.
Before this, every run-keyed table except ``frame_detections`` was scanned in
full on a ``WHERE run_id = ?`` read/delete, and the ``jobs`` status filters /
guarded transitions had no supporting index. See src/db/reader.py and
src/db/writer.py::delete_run.

Revision ID: b8e4f2a1c6d3
Revises: f3a1c9d2b7e5
Create Date: 2026-07-23 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "b8e4f2a1c6d3"
down_revision: Union[str, Sequence[str], None] = "f3a1c9d2b7e5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# (table, index_name, [columns]) — single source for upgrade/downgrade.
_RUN_ID_INDEXES = [
    ("global_stats", "ix_global_stats_run_id", ["run_id"]),
    (
        "visual_object_occurrences",
        "ix_visual_object_occurrences_run_id",
        ["run_id"],
    ),
    ("text_events", "ix_text_events_run_id", ["run_id"]),
    ("audio_segments", "ix_audio_segments_run_id", ["run_id"]),
    ("scene_descriptions", "ix_scene_descriptions_run_id", ["run_id"]),
    ("shot_boundaries", "ix_shot_boundaries_run_id", ["run_id"]),
    ("parser_results", "ix_parser_results_run_id", ["run_id"]),
    # Time-windowed overlay read: WHERE run_id=? AND timestamp_sec BETWEEN ?..?
    ("frame_detections", "ix_frame_detections_run_ts", ["run_id", "timestamp_sec"]),
    # jobs: status filters/guarded transitions and the sibling-by-run lookup.
    ("jobs", "ix_jobs_status", ["status"]),
    ("jobs", "ix_jobs_run_id", ["run_id"]),
]


def upgrade() -> None:
    """Upgrade schema."""
    for table, index_name, columns in _RUN_ID_INDEXES:
        with op.batch_alter_table(table, schema=None) as batch_op:
            batch_op.create_index(index_name, columns, unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    for table, index_name, _columns in reversed(_RUN_ID_INDEXES):
        with op.batch_alter_table(table, schema=None) as batch_op:
            batch_op.drop_index(index_name)
