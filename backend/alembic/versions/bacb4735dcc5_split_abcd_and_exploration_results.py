"""split abcd and exploration results

Renames ``parser_results`` to ``abcd_results`` (data preserved) and adds
``exploration_results``, one versioned JSON report per exploratory run.

Revision ID: bacb4735dcc5
Revises: 63ce44d9dd05
Create Date: 2026-10-19 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "bacb4735dcc5"
down_revision: Union[str, Sequence[str], None] = "63ce44d9dd05"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Rename keeps existing rows; the index keeps its old name across a table
    # rename on both SQLite and PostgreSQL, so recreate it under the new name.
    op.rename_table("parser_results", "abcd_results")
    op.drop_index("ix_parser_results_run_id", table_name="abcd_results")
    op.create_index("ix_abcd_results_run_id", "abcd_results", ["run_id"])

    op.create_table(
        "exploration_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("model", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.Text(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", name="uq_exploration_results_run_id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("exploration_results")

    op.drop_index("ix_abcd_results_run_id", table_name="abcd_results")
    op.rename_table("abcd_results", "parser_results")
    op.create_index("ix_parser_results_run_id", "parser_results", ["run_id"])
