"""add parent_job_id to jobs (batch fan-out)

Revision ID: d5e8b2c4a9f1
Revises: c1a2d3e4f5a6
Create Date: 2026-07-13 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d5e8b2c4a9f1"
down_revision: Union[str, Sequence[str], None] = "c1a2d3e4f5a6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the parent/child self-reference used by batch jobs (§2.1)."""
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("parent_job_id", sa.Text(), nullable=True))
        batch_op.create_index("ix_jobs_parent_job_id", ["parent_job_id"], unique=False)


def downgrade() -> None:
    """Drop the parent/child self-reference."""
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.drop_index("ix_jobs_parent_job_id")
        batch_op.drop_column("parent_job_id")
