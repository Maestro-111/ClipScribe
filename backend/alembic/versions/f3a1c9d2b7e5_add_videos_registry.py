"""add videos registry (upload dedup + friendly-name picker)

Revision ID: f3a1c9d2b7e5
Revises: e2f9c7a1b3d4
Create Date: 2026-07-15 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f3a1c9d2b7e5"
down_revision: Union[str, Sequence[str], None] = "e2f9c7a1b3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the ``videos`` registry (schema.py::videos_table)."""
    op.create_table(
        "videos",
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("stored_key", sa.Text(), nullable=False),
        sa.Column("original_name", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.Text(), server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column(
            "last_seen_at", sa.Text(), server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        sa.UniqueConstraint("user_id", "content_hash", name="uq_videos_user_hash"),
    )
    op.create_index("ix_videos_user", "videos", ["user_id"], unique=False)


def downgrade() -> None:
    """Drop the ``videos`` registry."""
    op.drop_index("ix_videos_user", table_name="videos")
    op.drop_table("videos")
