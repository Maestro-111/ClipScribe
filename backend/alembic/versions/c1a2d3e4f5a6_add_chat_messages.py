"""add chat_messages (advisory chat agent, web-app-plan §13)

Revision ID: c1a2d3e4f5a6
Revises: 5b7769f8e6fd
Create Date: 2026-07-08 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c1a2d3e4f5a6"
down_revision: Union[str, Sequence[str], None] = "5b7769f8e6fd"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("session_id", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("tool_calls_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.Text(), server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_chat_messages_run_session",
        "chat_messages",
        ["run_id", "session_id"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_chat_messages_run_session", table_name="chat_messages")
    op.drop_table("chat_messages")
