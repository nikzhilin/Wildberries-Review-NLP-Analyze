"""Alerts — threshold-rule violations with filters."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import plotly.express as px
import streamlit as st

from dashboard.utils import get_alerts

st.set_page_config(page_title="Alerts · WB Analytics", layout="wide")
st.title("🚨 Alerts")

df = get_alerts()

if df.empty:
    st.info("No alerts in the database. Run `python main.py` to generate them.")
    st.stop()

# ── Summary metrics ─────────────────────────────────────────────────────────
sev_counts = df["severity"].value_counts()
c1, c2, c3 = st.columns(3)
c1.metric("Total alerts",  f"{len(df):,}")
c2.metric("High",   f"{sev_counts.get('high',   0):,}", delta_color="inverse")
c3.metric("Medium", f"{sev_counts.get('medium', 0):,}")

st.divider()

# ── Charts ──────────────────────────────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("Alerts by rule")
    rule_counts = df["rule_name"].value_counts().reset_index()
    rule_counts.columns = ["rule", "count"]
    fig = px.bar(
        rule_counts,
        x="count",
        y="rule",
        orientation="h",
        color="count",
        color_continuous_scale="Reds",
        labels={"count": "Alerts", "rule": "Rule"},
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False, height=300)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Alerts by severity")
    fig2 = px.pie(
        sev_counts.reset_index(),
        names="severity",
        values="count",
        color="severity",
        color_discrete_map={"high": "#e74c3c", "medium": "#f39c12", "low": "#95a5a6"},
        hole=0.4,
    )
    fig2.update_layout(height=300)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Filters ─────────────────────────────────────────────────────────────────
st.subheader("Alert table")
col1, col2 = st.columns(2)
with col1:
    rule_filter = st.multiselect(
        "Filter by rule", df["rule_name"].unique().tolist(),
        default=df["rule_name"].unique().tolist(),
    )
with col2:
    sev_filter = st.multiselect(
        "Filter by severity", ["high", "medium", "low"],
        default=["high", "medium"],
    )

filtered = df[df["rule_name"].isin(rule_filter) & df["severity"].isin(sev_filter)]

# Flatten details dict into readable string
filtered = filtered.copy()
filtered["details_str"] = filtered["details"].apply(
    lambda d: "  |  ".join(f"{k}: {v}" for k, v in d.items()) if isinstance(d, dict) else str(d)
)

st.dataframe(
    filtered[["product_id", "rule_name", "severity", "details_str"]].rename(columns={
        "product_id": "Product", "rule_name": "Rule",
        "severity": "Severity", "details_str": "Details",
    }),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Severity": st.column_config.TextColumn(width="small"),
        "Details":  st.column_config.TextColumn(width="large"),
    },
)
st.caption(f"{len(filtered):,} alerts shown")
