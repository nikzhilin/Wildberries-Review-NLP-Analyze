"""Rating predictor — real vs predicted rating analysis."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import plotly.express as px
import streamlit as st

from dashboard.utils import get_rating_gap_per_product

st.set_page_config(page_title="Rating · WB Analytics", layout="wide")
st.title("⭐ Rating Predictor")

min_reviews = st.slider("Minimum reviews per product", 5, 100, 20, step=5)
df = get_rating_gap_per_product(min_reviews)

if df.empty:
    st.warning("No predicted rating data available.")
    st.stop()

# ── Summary metrics ─────────────────────────────────────────────────────────
with_gap = (df["gap"] > 0).sum()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Products analyzed", f"{len(df):,}")
c2.metric("With positive gap", f"{with_gap:,}", help="real > predicted")
c3.metric("Max gap", f"{df['gap'].max():.3f} ★")
c4.metric("Avg gap", f"{df['gap'].mean():.3f} ★")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Real vs predicted rating")
    st.caption("Each point = one product. Size = review count.")
    fig = px.scatter(
        df,
        x="pred_avg",
        y="real_avg",
        size="review_count",
        color="gap",
        color_continuous_scale="RdYlGn_r",
        hover_data={"product_id": True, "review_count": True, "gap": ":.3f"},
        labels={
            "pred_avg": "Predicted avg rating",
            "real_avg": "Real avg rating",
            "gap": "Gap",
        },
        opacity=0.7,
    )
    # Diagonal reference line: real == predicted
    mn = min(df["pred_avg"].min(), df["real_avg"].min()) - 0.1
    mx = max(df["pred_avg"].max(), df["real_avg"].max()) + 0.1
    fig.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                  line=dict(dash="dash", color="gray", width=1))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Gap distribution")
    fig2 = px.histogram(
        df,
        x="gap",
        nbins=60,
        color_discrete_sequence=["#3498db"],
        labels={"gap": "Rating gap (real − predicted)"},
    )
    fig2.add_vline(x=0, line_dash="dash", line_color="gray")
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Top products by gap ─────────────────────────────────────────────────────
st.subheader("Top 30 products by rating gap")
top = df.head(30).copy()
top["gap"] = top["gap"].round(3)
top["real_avg"] = top["real_avg"].round(3)
top["pred_avg"] = top["pred_avg"].round(3)
st.dataframe(
    top[["product_id", "review_count", "real_avg", "pred_avg", "gap"]].rename(columns={
        "product_id": "Product", "review_count": "Reviews",
        "real_avg": "Real avg ★", "pred_avg": "Predicted ★", "gap": "Gap ★",
    }),
    use_container_width=True,
    hide_index=True,
)
