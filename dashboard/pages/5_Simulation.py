"""Business impact simulation — rating gap → conversion drop → revenue loss."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import plotly.express as px
import streamlit as st

from dashboard.utils import _session

st.set_page_config(page_title="Simulation · WB Analytics", layout="wide")
st.title("💰 Business Impact Simulation")
st.caption("Estimates potential revenue loss from inflated star ratings")

# ── Assumption sliders ───────────────────────────────────────────────────────
with st.expander("⚙ Adjust assumptions", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    monthly_sales         = c1.number_input("Monthly sales per product (units)", 100, 100_000, 1_000, step=100)
    avg_price             = c2.number_input("Avg product price (₽)", 100, 50_000, 1_000, step=100)
    conversion_rate       = c3.slider("Conversion penalty per ★ gap", 0.01, 0.30, 0.15, step=0.01,
                                      help="0.15 = 15 % conversion drop per star of inflated rating")
    min_reviews           = c4.slider("Min reviews per product", 5, 100, 20, step=5)

# ── Compute ─────────────────────────────────────────────────────────────────
from simulation.business_effect import compute_per_product, summarize  # type: ignore[import-untyped]

with _session() as s:
    effects = compute_per_product(
        s,
        min_reviews=min_reviews,
        conversion_rate_per_star=conversion_rate,
        monthly_sales=float(monthly_sales),
        avg_price=float(avg_price),
    )

summary = summarize(effects, top_n=20)

# ── KPI row ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Products analyzed",    f"{summary.total_products:,}")
k2.metric("Products with gap > 0", f"{summary.products_with_gap:,}")
k3.metric("Avg gap",              f"{summary.avg_gap:+.3f} ★")
k4.metric("Total lost revenue",   f"{summary.total_lost_revenue:,.0f} ₽/мес",
          help="Assuming the same monthly_sales and avg_price for all products")

st.divider()

# ── Top products chart ───────────────────────────────────────────────────────
import pandas as pd

top_df = pd.DataFrame([
    {
        "Product":       f"#{e.product_id}",
        "Reviews":       e.review_count,
        "Gap ★":         e.rating_gap,
        "Conv. drop %":  round(e.conversion_drop_pct * 100, 1),
        "Lost revenue":  e.lost_revenue,
    }
    for e in summary.top_products
])

left, right = st.columns([2, 1])

with left:
    st.subheader("Top 20 products by estimated revenue loss")
    if not top_df.empty:
        fig = px.bar(
            top_df,
            x="Lost revenue",
            y="Product",
            orientation="h",
            color="Gap ★",
            color_continuous_scale="Reds",
            hover_data={"Reviews": True, "Conv. drop %": True, "Gap ★": ":.3f"},
            labels={"Lost revenue": "Est. lost revenue (₽/мес)", "Product": ""},
        )
        fig.update_layout(height=500, coloraxis_colorbar_title="Gap ★")
        st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Details")
    if not top_df.empty:
        st.dataframe(
            top_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Lost revenue": st.column_config.NumberColumn(format="₽ %,.0f"),
                "Gap ★":        st.column_config.NumberColumn(format="%.3f"),
            },
        )

st.divider()

# ── Gap distribution across all products ────────────────────────────────────
st.subheader("Rating gap distribution — all products")
all_df = pd.DataFrame([
    {"product_id": e.product_id, "gap": e.rating_gap, "reviews": e.review_count}
    for e in effects
])
if not all_df.empty:
    fig2 = px.histogram(
        all_df,
        x="gap",
        nbins=80,
        color_discrete_sequence=["#3498db"],
        labels={"gap": "Rating gap (real − predicted)"},
    )
    fig2.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="no gap")
    fig2.update_layout(height=300)
    st.plotly_chart(fig2, use_container_width=True)
