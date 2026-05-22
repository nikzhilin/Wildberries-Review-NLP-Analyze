"""Overview dashboard — key pipeline metrics at a glance."""
import plotly.express as px
import streamlit as st

from utils import (
    get_alerts,
    get_overview_stats,
    get_sentiment_distribution,
    get_top_products_by_reviews,
)

st.set_page_config(
    page_title="WB Review Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("📊 WB Review Analytics")
st.caption("Wildberries reviews — NLP pipeline overview")

stats = get_overview_stats()

# ── KPI row ────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Reviews", f"{stats['total_reviews']:,}")
c2.metric("Products", f"{stats['total_products']:,}")
c3.metric("Topics", f"{stats['total_topics']:,}")
c4.metric("Alerts", f"{stats['total_alerts']:,}")
c5.metric("High alerts", f"{stats['alerts_high']:,}", delta_color="inverse")

st.divider()

# ── Pipeline completeness ───────────────────────────────────────────────────
st.subheader("Pipeline completeness")
total = stats["total_reviews"]
stages = {
    "Sentiment":  stats["with_sentiment"],
    "Embeddings": stats["with_embedding"],
    "Predicted":  stats["with_predicted"],
}
prog_cols = st.columns(len(stages))
for col, (label, done) in zip(prog_cols, stages.items()):
    pct = done / total if total else 0
    col.metric(label, f"{done:,} / {total:,}")
    col.progress(pct)

st.divider()

# ── Sentiment + top products ────────────────────────────────────────────────
left, right = st.columns([1, 2])

with left:
    st.subheader("Sentiment distribution")
    sent_df = get_sentiment_distribution()
    if not sent_df.empty:
        fig = px.pie(
            sent_df,
            names="label",
            values="count",
            color="label",
            color_discrete_map={
                "positive": "#2ecc71",
                "neutral":  "#95a5a6",
                "negative": "#e74c3c",
            },
            hole=0.4,
        )
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=280)
        st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Top 20 products by review count")
    top_df = get_top_products_by_reviews(limit=20)
    if not top_df.empty:
        fig2 = px.bar(
            top_df,
            x="product_id",
            y="review_count",
            labels={"product_id": "Product ID", "review_count": "Reviews"},
        )
        fig2.update_layout(margin=dict(t=0, b=0), height=280)
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Recent high-severity alerts ─────────────────────────────────────────────
st.subheader("Recent high-severity alerts")
alerts_df = get_alerts()
high = alerts_df[alerts_df["severity"] == "high"] if not alerts_df.empty else alerts_df
if high.empty:
    st.info("No high-severity alerts.")
else:
    st.dataframe(
        high[["product_id", "rule_name", "severity", "details"]].iloc[:20],
        use_container_width=True,
        hide_index=True,
    )
