"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-05-18
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "products",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("marketplace", sa.String(64), nullable=False),
        sa.Column("external_id", sa.String(64), nullable=False),
        sa.Column("title", sa.String(512)),
        sa.Column("url", sa.String(1024)),
        sa.UniqueConstraint("marketplace", "external_id", name="uq_products_marketplace_external_id"),
    )
    op.create_table(
        "reviews",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("product_id", sa.Integer, sa.ForeignKey("products.id"), nullable=False),
        sa.Column("external_id", sa.String(128), nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("text_clean", sa.Text),
        sa.Column("rating", sa.Integer),
        sa.Column("color", sa.String(256)),
        sa.Column("answer", sa.Text),
        sa.Column("has_answer", sa.Boolean, server_default="false"),
        sa.Column("review_length", sa.Integer),
        sa.UniqueConstraint("external_id", name="uq_reviews_external_id"),
    )


def downgrade() -> None:
    op.drop_table("reviews")
    op.drop_table("products")
