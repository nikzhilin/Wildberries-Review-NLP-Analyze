"""Rating-gap → conversion-drop → revenue-loss simulation.

The model is linear:
    rating_gap       = real_avg_rating − predicted_avg_rating
    conversion_drop  = max(0, rating_gap × conversion_rate_per_star)
    lost_revenue     = monthly_sales × conversion_drop × avg_price

A positive gap means the displayed star rating overstates what behavioral
signals (sentiment, review length, seller responsiveness) justify.  The
conversion coefficient (default 0.15) is an industry rule-of-thumb: each star
of inflated rating reduces conversion by ~15 % once customers read the reviews
and notice the mismatch.  Adjust via the parameter to match internal data.

monthly_sales and avg_price are not stored in the database — callers supply
them as parameters or per-product overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from sqlalchemy import text


@dataclass
class ProductEffect:
    """Business impact estimate for a single product."""

    product_id: int
    review_count: int
    real_avg_rating: float
    predicted_avg_rating: float
    rating_gap: float  # real − predicted; positive = inflated
    conversion_drop_pct: float  # fractional conversion reduction [0, 1]
    lost_revenue: float  # monthly_sales × conversion_drop_pct × avg_price


@dataclass
class EffectSummary:
    """Fleet-level aggregation of business impact."""

    total_products: int
    products_with_gap: int  # products where rating_gap > 0
    avg_gap: float
    max_gap: float
    total_lost_revenue: float
    avg_lost_revenue_per_product: float
    top_products: list[ProductEffect] = field(default_factory=list)


def compute_per_product(
    session,
    *,
    min_reviews: int = 20,
    conversion_rate_per_star: float = 0.15,
    monthly_sales: float = 1_000,
    avg_price: float = 1_000,
) -> list[ProductEffect]:
    """Compute per-product rating gap and estimated revenue impact.

    Queries review_ml for predicted_rating, joins with reviews for real rating,
    and applies the linear conversion model to each product.

    Args:
        session: Open SQLAlchemy session (read-only).
        min_reviews: Products with fewer reviews are excluded to suppress noise.
        conversion_rate_per_star: Fraction of conversion lost per star of gap.
            0.15 = 15 % per star (adjustable; derive from your A/B data).
        monthly_sales: Assumed monthly unit volume per product.
        avg_price: Assumed average sale price in ₽ per product.

    Returns:
        List of ProductEffect sorted by rating_gap descending.
    """
    rows = session.execute(
        text(
            """
            SELECT
                r.product_id,
                COUNT(*) AS review_count,
                AVG(r.rating::float) AS real_avg,
                AVG(m.predicted_rating) AS pred_avg
            FROM reviews r
            JOIN review_ml m ON m.review_id = r.id
            WHERE m.predicted_rating IS NOT NULL
            GROUP BY r.product_id
            HAVING COUNT(*) >= :min_reviews
            ORDER BY (AVG(r.rating::float) - AVG(m.predicted_rating)) DESC
            """
        ),
        {"min_reviews": min_reviews},
    ).fetchall()

    effects: list[ProductEffect] = []
    for row in rows:
        real_avg = float(row.real_avg)
        pred_avg = float(row.pred_avg)
        gap = real_avg - pred_avg
        conversion_drop = max(0.0, gap * conversion_rate_per_star)
        lost_revenue = monthly_sales * conversion_drop * avg_price

        effects.append(
            ProductEffect(
                product_id=row.product_id,
                review_count=int(row.review_count),
                real_avg_rating=round(real_avg, 3),
                predicted_avg_rating=round(pred_avg, 3),
                rating_gap=round(gap, 3),
                conversion_drop_pct=round(conversion_drop, 4),
                lost_revenue=round(lost_revenue, 2),
            )
        )
    return effects


def compute_for_product(
    session,
    product_id: int,
    *,
    conversion_rate_per_star: float = 0.15,
    monthly_sales: float = 1_000,
    avg_price: float = 1_000,
) -> ProductEffect | None:
    """Compute business impact for a single product.

    Returns None if the product has no predicted_rating data.
    """
    row = session.execute(
        text(
            """
            SELECT
                COUNT(*) AS review_count,
                AVG(r.rating::float) AS real_avg,
                AVG(m.predicted_rating) AS pred_avg
            FROM reviews r
            JOIN review_ml m ON m.review_id = r.id
            WHERE r.product_id = :pid
              AND m.predicted_rating IS NOT NULL
            """
        ),
        {"pid": product_id},
    ).fetchone()

    if not row or not row.real_avg:
        return None

    real_avg = float(row.real_avg)
    pred_avg = float(row.pred_avg)
    gap = real_avg - pred_avg
    conversion_drop = max(0.0, gap * conversion_rate_per_star)

    return ProductEffect(
        product_id=product_id,
        review_count=int(row.review_count),
        real_avg_rating=round(real_avg, 3),
        predicted_avg_rating=round(pred_avg, 3),
        rating_gap=round(gap, 3),
        conversion_drop_pct=round(conversion_drop, 4),
        lost_revenue=round(monthly_sales * conversion_drop * avg_price, 2),
    )


def summarize(
    effects: Sequence[ProductEffect],
    top_n: int = 10,
) -> EffectSummary:
    """Aggregate per-product effects into a fleet-level summary.

    Args:
        effects: Output of compute_per_product().
        top_n: Number of top products (by lost_revenue) to include in the result.

    Returns:
        EffectSummary with global statistics and top-N products.
    """
    if not effects:
        return EffectSummary(
            total_products=0,
            products_with_gap=0,
            avg_gap=0.0,
            max_gap=0.0,
            total_lost_revenue=0.0,
            avg_lost_revenue_per_product=0.0,
        )

    n = len(effects)
    gaps = [e.rating_gap for e in effects]
    revenues = [e.lost_revenue for e in effects]

    top = sorted(effects, key=lambda e: e.lost_revenue, reverse=True)[:top_n]

    return EffectSummary(
        total_products=n,
        products_with_gap=sum(1 for g in gaps if g > 0),
        avg_gap=round(sum(gaps) / n, 4),
        max_gap=round(max(gaps), 4),
        total_lost_revenue=round(sum(revenues), 2),
        avg_lost_revenue_per_product=round(sum(revenues) / n, 2),
        top_products=top,
    )
