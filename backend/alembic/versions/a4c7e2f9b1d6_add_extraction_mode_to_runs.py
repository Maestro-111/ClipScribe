"""add extraction_mode to runs

Persists whether a run included object detection and tracking. Existing runs
default to ``with_objects``, which matches the behavior before extraction modes
were introduced.

Revision ID: a4c7e2f9b1d6
Revises: b8e4f2a1c6d3
Create Date: 2026-10-13 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "a4c7e2f9b1d6"
down_revision: Union[str, Sequence[str], None] = "b8e4f2a1c6d3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("runs", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "extraction_mode",
                sa.Text(),
                server_default=sa.text("'with_objects'"),
                nullable=False,
            )
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("runs", schema=None) as batch_op:
        batch_op.drop_column("extraction_mode")
