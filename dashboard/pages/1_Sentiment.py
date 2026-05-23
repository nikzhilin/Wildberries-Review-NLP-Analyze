"""Sentiment analysis — per-product breakdown and rating×sentiment matrix."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import plotly.express as px
import streamlit as st

from dashboard.utils import (
    get_product_sentiment,
    get_rating_sentiment_matrix,
    get_top_products_by_reviews,
)

st.set_page_config(page_title="Sentiment · WB Analytics", layout="wide")
st.title("💬 Sentiment Analysis")

# ── Product selector ────────────────────────────────────────────────────────
top_df = get_top_products_by_reviews(limit=100)
product_ids = top_df["product_id"].tolist()
selected = st.selectbox(
    "Select product (top 100 by review count)",
    product_ids,
    format_func=lambda pid: f"Product {pid}  ({top_df.loc[top_df.product_id == pid, 'review_count'].iloc[0]:,} reviews)",
)

df = get_product_sentiment(selected)
if df.empty:
    st.warning("No sentiment data for this product.")
    st.stop()

# ── Per-product metrics ─────────────────────────────────────────────────────
counts = df["sentiment_label"].value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total reviews", f"{len(df):,}")
c2.metric("Positive", f"{counts.get('positive', 0):,}")
c3.metric("Neutral",  f"{counts.get('neutral', 0):,}")
c4.metric("Negative", f"{counts.get('negative', 0):,}")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Sentiment distribution")
    bar_df = counts.reset_index()
    bar_df.columns = ["label", "count"]
    fig = px.bar(
        bar_df,
        x="label",
        y="count",
        color="label",
        color_discrete_map={
            "positive": "#2ecc71",
            "neutral":  "#95a5a6",
            "negative": "#e74c3c",
        },
        labels={"label": "Sentiment", "count": "Reviews"},
    )
    fig.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Rating × sentiment score")
    fig2 = px.scatter(
        df.sample(min(len(df), 500)),
        x="rating",
        y="sentiment_score",
        color="sentiment_label",
        color_discrete_map={
            "positive": "#2ecc71",
            "neutral":  "#95a5a6",
            "negative": "#e74c3c",
        },
        opacity=0.6,
        labels={"rating": "Star rating", "sentiment_score": "Confidence"},
    )
    fig2.update_layout(height=300)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Global rating × sentiment heatmap ──────────────────────────────────────
st.subheader("Rating × sentiment — all products")
matrix_df = get_rating_sentiment_matrix()
if not matrix_df.empty:
    pivot = matrix_df.pivot(index="sentiment_label", columns="rating", values="count").fillna(0)
    fig3 = px.imshow(
        pivot,
        text_auto=True,
        color_continuous_scale="Blues",
        labels={"x": "Star rating", "y": "Sentiment", "color": "Reviews"},
        aspect="auto",
    )
    fig3.update_layout(height=280)
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── Negative reviews table ──────────────────────────────────────────────────
st.subheader("Most negative reviews")
neg = df[df["sentiment_label"] == "negative"].sort_values("sentiment_score").head(15)
if neg.empty:
    st.info("No negative reviews for this product.")
else:
    st.dataframe(
        neg[["rating", "sentiment_score", "text"]].rename(columns={
            "rating": "Stars", "sentiment_score": "Confidence", "text": "Review"
        }),
        use_container_width=True,
        hide_index=True,
        column_config={"Review": st.column_config.TextColumn(width="large")},
    )
