"""Threshold-based alert rules over ML outputs.

Each rule accepts an open SQLAlchemy session and returns a list of AlertRow
objects.  Rules read pre-computed columns in reviews / review_ml — no ML
inference is performed here.

Topic-based rules are not yet implemented: topic_id is populated only after
clustering completes (requires embeddings for all reviews).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)


@dataclass
class AlertRow:
    product_id: int
    rule_name: str
    severity: str  # "low" | "medium" | "high"
    details: dict[str, Any] = field(default_factory=dict)


def rule_high_negative_share(
    session,
    min_reviews: int = 10,
    medium_threshold: float = 0.30,
    high_threshold: float = 0.50,
) -> list[AlertRow]:
    """Products where negative-sentiment reviews exceed a share threshold.

    Args:
        min_reviews: Minimum review count to qualify.
        medium_threshold: Negative share that triggers a medium alert.
        high_threshold: Negative share that escalates to high.
    """
    rows = session.execute(
        text(
            """
            SELECT
                r.product_id,
                COUNT(*) AS total,
                SUM(CASE WHEN m.sentiment_label = 'negative' THEN 1 ELSE 0 END) AS neg_count
            FROM reviews r
            JOIN review_ml m ON m.review_id = r.id
            GROUP BY r.product_id
            HAVING COUNT(*) >= :min_reviews
              AND SUM(CASE WHEN m.sentiment_label = 'negative' THEN 1 ELSE 0 END)
                  * 1.0 / COUNT(*) > :medium_threshold
            """
        ),
        {"min_reviews": min_reviews, "medium_threshold": medium_threshold},
    ).fetchall()

    alerts: list[AlertRow] = []
    for row in rows:
        neg_pct = int(row.neg_count) / int(row.total)
        alerts.append(
            AlertRow(
                product_id=row.product_id,
                rule_name="high_negative_share",
                severity="high" if neg_pct >= high_threshold else "medium",
                details={
                    "neg_pct": round(neg_pct, 3),
                    "neg_count": int(row.neg_count),
                    "total_reviews": int(row.total),
                },
            )
        )
    return alerts


def rule_rating_sentiment_mismatch(
    session,
    min_reviews: int = 10,
    min_avg_rating: float = 4.0,
    medium_threshold: float = 0.20,
    high_threshold: float = 0.35,
) -> list[AlertRow]:
    """Products with a high star rating but a significant share of negative-sentiment text.

    A high rating combined with negative text suggests inflated stars or
    customers who rate generously but complain in writing.

    Args:
        min_avg_rating: Only check products whose avg rating is at least this value.
        medium_threshold: Negative text share that triggers a medium alert.
        high_threshold: Negative text share that escalates to high.
    """
    rows = session.execute(
        text(
            """
            SELECT
                r.product_id,
                COUNT(*) AS total,
                AVG(r.rating::float) AS avg_rating,
                SUM(CASE WHEN m.sentiment_label = 'negative' THEN 1 ELSE 0 END) AS neg_count
            FROM reviews r
            JOIN review_ml m ON m.review_id = r.id
            GROUP BY r.product_id
            HAVING COUNT(*) >= :min_reviews
              AND AVG(r.rating::float) >= :min_avg_rating
              AND SUM(CASE WHEN m.sentiment_label = 'negative' THEN 1 ELSE 0 END)
                  * 1.0 / COUNT(*) > :medium_threshold
            """
        ),
        {
            "min_reviews": min_reviews,
            "min_avg_rating": min_avg_rating,
            "medium_threshold": medium_threshold,
        },
    ).fetchall()

    alerts: list[AlertRow] = []
    for row in rows:
        neg_pct = int(row.neg_count) / int(row.total)
        alerts.append(
            AlertRow(
                product_id=row.product_id,
                rule_name="rating_sentiment_mismatch",
                severity="high" if neg_pct >= high_threshold else "medium",
                details={
                    "avg_rating": round(float(row.avg_rating), 2),
                    "neg_pct": round(neg_pct, 3),
                    "neg_count": int(row.neg_count),
                    "total_reviews": int(row.total),
                },
            )
        )
    return alerts


def rule_predicted_rating_gap(
    session,
    min_reviews: int = 20,
    medium_threshold: float = 0.5,
    high_threshold: float = 1.0,
) -> list[AlertRow]:
    """Products where the avg real rating substantially exceeds the model's prediction.

    The rating predictor is trained on behavioral signals (text sentiment,
    review length, seller responsiveness).  A large positive gap means the
    star rating is higher than those signals justify.

    Args:
        min_reviews: Minimum reviews required to suppress single-review noise.
        medium_threshold: Gap in stars that triggers a medium alert.
        high_threshold: Gap that escalates to high.
    """
    rows = session.execute(
        text(
            """
            SELECT
                r.product_id,
                COUNT(*) AS total,
                AVG(r.rating::float) AS real_avg,
                AVG(m.predicted_rating) AS pred_avg
            FROM reviews r
            JOIN review_ml m ON m.review_id = r.id
            WHERE m.predicted_rating IS NOT NULL
            GROUP BY r.product_id
            HAVING COUNT(*) >= :min_reviews
              AND AVG(r.rating::float) - AVG(m.predicted_rating) > :medium_threshold
            """
        ),
        {"min_reviews": min_reviews, "medium_threshold": medium_threshold},
    ).fetchall()

    alerts: list[AlertRow] = []
    for row in rows:
        gap = float(row.real_avg) - float(row.pred_avg)
        alerts.append(
            AlertRow(
                product_id=row.product_id,
                rule_name="predicted_rating_gap",
                severity="high" if gap >= high_threshold else "medium",
                details={
                    "real_avg": round(float(row.real_avg), 3),
                    "predicted_avg": round(float(row.pred_avg), 3),
                    "gap": round(gap, 3),
                    "total_reviews": int(row.total),
                },
            )
        )
    return alerts


def rule_low_overall_rating(
    session,
    min_reviews: int = 5,
    medium_threshold: float = 3.0,
    high_threshold: float = 2.0,
) -> list[AlertRow]:
    """Products with a consistently low star rating.

    Args:
        min_reviews: Minimum reviews required to avoid alerting on sparse data.
        medium_threshold: Avg rating below this triggers a medium alert.
        high_threshold: Avg rating below this escalates to high.
    """
    rows = session.execute(
        text(
            """
            SELECT r.product_id, COUNT(*) AS total, AVG(r.rating::float) AS avg_rating
            FROM reviews r
            GROUP BY r.product_id
            HAVING COUNT(*) >= :min_reviews
              AND AVG(r.rating::float) < :medium_threshold
            """
        ),
        {"min_reviews": min_reviews, "medium_threshold": medium_threshold},
    ).fetchall()

    alerts: list[AlertRow] = []
    for row in rows:
        avg = float(row.avg_rating)
        alerts.append(
            AlertRow(
                product_id=row.product_id,
                rule_name="low_overall_rating",
                severity="high" if avg < high_threshold else "medium",
                details={
                    "avg_rating": round(avg, 3),
                    "total_reviews": int(row.total),
                },
            )
        )
    return alerts


_RULES = [
    rule_high_negative_share,
    rule_rating_sentiment_mismatch,
    rule_predicted_rating_gap,
    rule_low_overall_rating,
]


def evaluate_all(session) -> list[AlertRow]:
    """Run all rules and return the combined alert list."""
    results: list[AlertRow] = []
    for rule in _RULES:
        fired = rule(session)
        logger.info("%-35s → %d alerts", rule.__name__, len(fired))
        results.extend(fired)
    return results
