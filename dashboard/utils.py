"""Shared DB helpers and cached query functions for all dashboard pages."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when running via `streamlit run dashboard/app.py`
_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import streamlit as st
from sqlalchemy import text


@st.cache_resource
def _get_session_factory():
    from db import SessionLocal
    return SessionLocal


def _session():
    return _get_session_factory()()


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

@st.cache_data(ttl=120)
def get_overview_stats() -> dict:
    with _session() as s:
        def q(sql: str):
            return s.execute(text(sql)).scalar() or 0

        return {
            "total_reviews":    q("SELECT COUNT(*) FROM reviews"),
            "total_products":   q("SELECT COUNT(*) FROM products"),
            "with_sentiment":   q("SELECT COUNT(*) FROM review_ml WHERE sentiment_label IS NOT NULL"),
            "with_embedding":   q("SELECT COUNT(*) FROM review_ml WHERE embedding IS NOT NULL"),
            "with_predicted":   q("SELECT COUNT(*) FROM review_ml WHERE predicted_rating IS NOT NULL"),
            "total_topics":     q("SELECT COUNT(*) FROM topics"),
            "total_alerts":     q("SELECT COUNT(*) FROM alerts"),
            "alerts_high":      q("SELECT COUNT(*) FROM alerts WHERE severity = 'high'"),
        }


@st.cache_data(ttl=120)
def get_sentiment_distribution() -> pd.DataFrame:
    with _session() as s:
        rows = s.execute(text(
            "SELECT sentiment_label, COUNT(*) AS cnt FROM review_ml "
            "WHERE sentiment_label IS NOT NULL GROUP BY sentiment_label"
        )).fetchall()
    return pd.DataFrame(rows, columns=["label", "count"])


@st.cache_data(ttl=120)
def get_top_products_by_reviews(limit: int = 50) -> pd.DataFrame:
    with _session() as s:
        rows = s.execute(text(
            "SELECT product_id, COUNT(*) AS review_count FROM reviews "
            "GROUP BY product_id ORDER BY review_count DESC LIMIT :lim"
        ), {"lim": limit}).fetchall()
    return pd.DataFrame(rows, columns=["product_id", "review_count"])


# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------

@st.cache_data(ttl=120)
def get_product_sentiment(product_id: int) -> pd.DataFrame:
    with _session() as s:
        rows = s.execute(text(
            "SELECT m.sentiment_label, m.sentiment_score, r.rating, r.text "
            "FROM review_ml m JOIN reviews r ON r.id = m.review_id "
            "WHERE r.product_id = :pid AND m.sentiment_label IS NOT NULL "
            "ORDER BY m.sentiment_score DESC"
        ), {"pid": product_id}).fetchall()
    return pd.DataFrame(rows, columns=["sentiment_label", "sentiment_score", "rating", "text"])


@st.cache_data(ttl=300)
def get_rating_sentiment_matrix() -> pd.DataFrame:
    with _session() as s:
        rows = s.execute(text(
            "SELECT r.rating, m.sentiment_label, COUNT(*) AS cnt "
            "FROM reviews r JOIN review_ml m ON m.review_id = r.id "
            "WHERE r.rating IS NOT NULL AND m.sentiment_label IS NOT NULL "
            "GROUP BY r.rating, m.sentiment_label"
        )).fetchall()
    return pd.DataFrame(rows, columns=["rating", "sentiment_label", "count"])


# ---------------------------------------------------------------------------
# Topics
# ---------------------------------------------------------------------------

@st.cache_data(ttl=120)
def get_products_with_topics() -> pd.DataFrame:
    with _session() as s:
        rows = s.execute(text(
            "SELECT DISTINCT t.product_id, COUNT(t.id) AS topic_count, "
            "SUM(t.review_count) AS review_count "
            "FROM topics t GROUP BY t.product_id ORDER BY review_count DESC LIMIT 100"
        )).fetchall()
    return pd.DataFrame(rows, columns=["product_id", "topic_count", "review_count"])


@st.cache_data(ttl=120)
def get_product_topics(product_id: int) -> pd.DataFrame:
    with _session() as s:
        rows = s.execute(text(
            "SELECT id, polarity, keywords, review_count "
            "FROM topics WHERE product_id = :pid ORDER BY polarity, review_count DESC"
        ), {"pid": product_id}).fetchall()
    return pd.DataFrame(rows, columns=["topic_id", "polarity", "keywords", "review_count"])


@st.cache_data(ttl=120)
def get_topic_reviews(topic_id: int, limit: int = 20) -> pd.DataFrame:
    with _session() as s:
        rows = s.execute(text(
            "SELECT r.text, r.rating, m.sentiment_label, m.sentiment_score "
            "FROM review_ml m JOIN reviews r ON r.id = m.review_id "
            "WHERE m.topic_id = :tid ORDER BY m.sentiment_score DESC LIMIT :lim"
        ), {"tid": topic_id, "lim": limit}).fetchall()
    return pd.DataFrame(rows, columns=["text", "rating", "sentiment_label", "sentiment_score"])


# ---------------------------------------------------------------------------
# Rating predictor
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def get_rating_gap_per_product(min_reviews: int = 20) -> pd.DataFrame:
    with _session() as s:
        rows = s.execute(text(
            "SELECT r.product_id, COUNT(*) AS review_count, "
            "AVG(r.rating::float) AS real_avg, AVG(m.predicted_rating) AS pred_avg, "
            "AVG(r.rating::float) - AVG(m.predicted_rating) AS gap "
            "FROM reviews r JOIN review_ml m ON m.review_id = r.id "
            "WHERE m.predicted_rating IS NOT NULL "
            "GROUP BY r.product_id HAVING COUNT(*) >= :min "
            "ORDER BY gap DESC"
        ), {"min": min_reviews}).fetchall()
    df = pd.DataFrame(rows, columns=["product_id", "review_count", "real_avg", "pred_avg", "gap"])
    return df.astype({"real_avg": float, "pred_avg": float, "gap": float})


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def get_alerts() -> pd.DataFrame:
    with _session() as s:
        rows = s.execute(text(
            "SELECT a.id, a.product_id, a.rule_name, a.severity, a.details, a.created_at "
            "FROM alerts a ORDER BY a.severity DESC, a.created_at DESC"
        )).fetchall()
    if not rows:
        return pd.DataFrame(columns=["id", "product_id", "rule_name", "severity", "details", "created_at"])
    return pd.DataFrame(rows, columns=["id", "product_id", "rule_name", "severity", "details", "created_at"])
