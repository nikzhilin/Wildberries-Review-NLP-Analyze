"""ml tables: review_ml, topics, alerts

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-19
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "topics",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("product_id", sa.Integer, sa.ForeignKey("products.id"), nullable=False),
        sa.Column("polarity", sa.String(8), nullable=False),
        sa.Column("keywords", sa.Text),
        sa.Column("review_count", sa.Integer, server_default="0"),
    )

    op.create_table(
        "alerts",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("product_id", sa.Integer, sa.ForeignKey("products.id"), nullable=False),
        sa.Column("rule_name", sa.String(128), nullable=False),
        sa.Column("severity", sa.String(32), nullable=False),
        sa.Column("details", sa.JSON),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )

    op.create_table(
        "review_ml",
        sa.Column("review_id", sa.Integer, sa.ForeignKey("reviews.id"), primary_key=True),
        sa.Column("sentiment_label", sa.String(16)),
        sa.Column("sentiment_score", sa.Float),
        sa.Column("topic_id", sa.Integer, sa.ForeignKey("topics.id")),
        sa.Column("fake_score", sa.Float),
        sa.Column("predicted_rating", sa.Float),
        sa.Column("embedding", Vector(768)),
    )


def downgrade() -> None:
    op.drop_table("review_ml")
    op.drop_table("alerts")
    op.drop_table("topics")
