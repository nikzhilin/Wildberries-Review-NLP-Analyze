"""Shrink embedding column from vector(768) to vector(384) for e5-small

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-21
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # All embeddings are NULL at this point — safe to drop and recreate.
    op.drop_column("review_ml", "embedding")
    op.add_column("review_ml", sa.Column("embedding", Vector(384)))


def downgrade() -> None:
    op.drop_column("review_ml", "embedding")
    op.add_column("review_ml", sa.Column("embedding", Vector(768)))
