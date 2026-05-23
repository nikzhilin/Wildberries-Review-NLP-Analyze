"""Topics — BERTopic results per product."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import plotly.express as px
import streamlit as st

from dashboard.utils import (
    get_product_topics,
    get_products_with_topics,
    get_topic_reviews,
)

st.set_page_config(page_title="Topics · WB Analytics", layout="wide")
st.title("🗂 Topics")

topics_index = get_products_with_topics()

if topics_index.empty:
    st.warning("No topics in the database yet. Clustering may still be running.")
    st.stop()

st.caption(f"{len(topics_index):,} products with topics · {topics_index['review_count'].sum():,} reviews assigned")

# ── Product selector ────────────────────────────────────────────────────────
selected_pid = st.selectbox(
    "Select product",
    topics_index["product_id"].tolist(),
    format_func=lambda pid: (
        f"Product {pid}  "
        f"({topics_index.loc[topics_index.product_id == pid, 'topic_count'].iloc[0]} topics, "
        f"{topics_index.loc[topics_index.product_id == pid, 'review_count'].iloc[0]:,} reviews)"
    ),
)

topics_df = get_product_topics(selected_pid)
if topics_df.empty:
    st.info("No topics for this product.")
    st.stop()

# ── Polarity tabs ───────────────────────────────────────────────────────────
polarities = topics_df["polarity"].unique().tolist()
tabs = st.tabs([p.upper() for p in polarities])

for tab, polarity in zip(tabs, polarities):
    with tab:
        pol_df = topics_df[topics_df["polarity"] == polarity].copy()

        left, right = st.columns([2, 1])

        with left:
            st.subheader(f"{polarity.upper()} topics")
            st.dataframe(
                pol_df[["keywords", "review_count"]].rename(columns={
                    "keywords": "Keywords", "review_count": "Reviews"
                }),
                use_container_width=True,
                hide_index=True,
                column_config={"Keywords": st.column_config.TextColumn(width="large")},
            )

        with right:
            fig = px.bar(
                pol_df.sort_values("review_count", ascending=True),
                x="review_count",
                y="keywords",
                orientation="h",
                labels={"review_count": "Reviews", "keywords": ""},
            )
            fig.update_layout(height=max(200, len(pol_df) * 40), margin=dict(l=0))
            st.plotly_chart(fig, use_container_width=True)

        # ── Reviews for selected topic ──────────────────────────────────────
        st.divider()
        topic_options = pol_df.apply(
            lambda r: f"[{r.topic_id}] {r.keywords[:50]}…  ({r.review_count} reviews)", axis=1
        ).tolist()
        topic_ids = pol_df["topic_id"].tolist()

        chosen_label = st.selectbox(
            "Show reviews for topic", topic_options, key=f"topic_sel_{polarity}"
        )
        chosen_id = topic_ids[topic_options.index(chosen_label)]

        reviews_df = get_topic_reviews(chosen_id, limit=20)
        if not reviews_df.empty:
            st.dataframe(
                reviews_df[["rating", "sentiment_label", "text"]].rename(columns={
                    "rating": "Stars", "sentiment_label": "Sentiment", "text": "Review"
                }),
                use_container_width=True,
                hide_index=True,
                column_config={"Review": st.column_config.TextColumn(width="large")},
            )
        else:
            st.info("No reviews found for this topic.")
