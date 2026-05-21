"""add word_count and answer_length to reviews

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-21
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("reviews", sa.Column("word_count", sa.Integer))
    op.add_column("reviews", sa.Column("answer_length", sa.Integer))


def downgrade() -> None:
    op.drop_column("reviews", "answer_length")
    op.drop_column("reviews", "word_count")
