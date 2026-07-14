"""add job_id to chat_messages (job-level advisory chat)

Revision ID: e2f9c7a1b3d4
Revises: d5e8b2c4a9f1
Create Date: 2026-07-14 00:00:00.000000

Job-level advisory chat stores its transcript in the same ``chat_messages``
table as the per-run chat, but scoped to a batch job instead of a single run.
A message belongs to exactly one scope: run chat sets ``run_id`` (``job_id``
NULL, unchanged), job chat sets ``job_id`` (``run_id`` NULL). ``run_id`` is
therefore relaxed to nullable.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e2f9c7a1b3d4"
down_revision: Union[str, Sequence[str], None] = "d5e8b2c4a9f1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Batch mode so SQLite (which can't ALTER nullability in place) recreates the
    # table; a no-op wrapper on Postgres.
    with op.batch_alter_table("chat_messages") as batch_op:
        batch_op.add_column(sa.Column("job_id", sa.Text(), nullable=True))
        batch_op.alter_column("run_id", existing_type=sa.Text(), nullable=True)
    op.create_index(
        "ix_chat_messages_job_session",
        "chat_messages",
        ["job_id", "session_id"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_chat_messages_job_session", table_name="chat_messages")
    with op.batch_alter_table("chat_messages") as batch_op:
        batch_op.alter_column("run_id", existing_type=sa.Text(), nullable=False)
        batch_op.drop_column("job_id")
