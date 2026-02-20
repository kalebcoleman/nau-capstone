"""Player Archetypes — clustering visualization and archetype details."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    ACCENT,
    ANALYSIS_DATA_DIR,
    APP_BG,
    FIGURES_DIR,
    MUTED_TEXT,
    PANEL_BG,
    TEXT_COLOR,
    apply_theme,
)

st.set_page_config(page_title="Player Archetypes", layout="wide")
apply_theme()
st.title("🧬 Player Archetypes")


# ---------------------------------------------------------------------------
# Load cluster data
# ---------------------------------------------------------------------------

CLUSTERS_PATH = ANALYSIS_DATA_DIR / "player_clusters.csv"


@st.cache_data(show_spinner=False)
def load_clusters() -> pd.DataFrame:
    if not CLUSTERS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CLUSTERS_PATH)


clusters_df = load_clusters()

if clusters_df.empty:
    st.warning(f"Cluster data not found at `{CLUSTERS_PATH}`. Run `advanced_analytics.py` first.")
    st.stop()

st.markdown(
    f'<div style="color: {MUTED_TEXT}; margin-bottom: 16px;">'
    f"Player archetypes derived from shot profile clustering (GMM). "
    f"Each player is assigned a spatial-descriptive archetype based on "
    f"their shooting tendencies.</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Archetype Filters")
    archetypes = sorted(clusters_df["archetype"].dropna().unique().tolist())
    selected_archetypes = st.multiselect("Archetypes", archetypes, default=archetypes, key="arch_sel")
    min_att = st.slider("Min Attempts", 50, 1000, 100, 25, key="arch_min_att")
    search = st.text_input("Search Player", key="arch_search")

filtered = clusters_df.copy()
if selected_archetypes:
    filtered = filtered[filtered["archetype"].isin(selected_archetypes)]
if "total_attempts" in filtered.columns:
    filtered = filtered[filtered["total_attempts"] >= min_att]
elif "attempts_per_game" in filtered.columns and "games_played" in filtered.columns:
    filtered["approx_total"] = filtered["attempts_per_game"] * filtered["games_played"]
    filtered = filtered[filtered["approx_total"] >= min_att]
if search:
    filtered = filtered[filtered["player"].str.contains(search, case=False, na=False)]


# ---------------------------------------------------------------------------
# PCA Scatter
# ---------------------------------------------------------------------------

st.subheader("Archetype Scatter (PCA-like Projection)")

if len(filtered) < 3:
    st.info("Not enough players for scatter. Adjust filters.")
else:
    feature_cols = [c for c in [
        "avg_distance", "avg_sdi", "pullup_rate", "pct_rim",
        "pct_midrange", "pct_3pt", "pct_corner_3", "distance_entropy",
    ] if c in filtered.columns]

    if len(feature_cols) >= 2:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        X = filtered[feature_cols].fillna(0).values
        X_scaled = StandardScaler().fit_transform(X)
        pcs = PCA(n_components=2).fit_transform(X_scaled)
        filtered = filtered.copy()
        filtered["PC1"] = pcs[:, 0]
        filtered["PC2"] = pcs[:, 1]

        archetype_colors = [
            "#F5C84C", "#FF6B6B", "#4ECDC4", "#A78BFA",
            "#60A5FA", "#F97316", "#34D399", "#FB7185",
            "#FBBF24", "#818CF8", "#94A3B8",
        ]

        fig = px.scatter(
            filtered, x="PC1", y="PC2",
            color="archetype",
            color_discrete_sequence=archetype_colors,
            hover_name="player",
            hover_data={
                "archetype": True,
                "avg_sdi": ":.3f" if "avg_sdi" in filtered.columns else False,
                "actual_fg_pct": ":.1%" if "actual_fg_pct" in filtered.columns else False,
                "PC1": False, "PC2": False,
            },
            size_max=14,
        )
        fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color="black")))
        fig.update_layout(
            paper_bgcolor=APP_BG, plot_bgcolor=PANEL_BG,
            font=dict(color=TEXT_COLOR),
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            legend=dict(font=dict(size=11, color=TEXT_COLOR)),
            margin=dict(l=50, r=20, t=30, b=50),
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough feature columns for PCA projection.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Archetype summary table
# ---------------------------------------------------------------------------

st.subheader("Archetype Summary")

if "archetype" in filtered.columns:
    agg_cols = {}
    for col in ["avg_sdi", "actual_fg_pct", "pullup_rate", "pct_rim", "pct_3pt", "pct_midrange",
                 "avg_distance", "usage_pct"]:
        if col in filtered.columns:
            agg_cols[col] = "mean"
    if "player" in filtered.columns:
        agg_cols["player"] = "count"

    if agg_cols:
        summary = filtered.groupby("archetype").agg(agg_cols).reset_index()
        rename = {"player": "Count"}
        for c in summary.columns:
            if c == "archetype":
                rename[c] = "Archetype"
            elif c == "avg_sdi":
                rename[c] = "Avg SDI"
            elif c == "actual_fg_pct":
                rename[c] = "FG%"
            elif c == "pullup_rate":
                rename[c] = "Pull-Up Rate"
            elif c == "pct_rim":
                rename[c] = "Rim %"
            elif c == "pct_3pt":
                rename[c] = "3PT %"
            elif c == "pct_midrange":
                rename[c] = "Mid-Range %"
            elif c == "avg_distance":
                rename[c] = "Avg Dist"
            elif c == "usage_pct":
                rename[c] = "Usage %"
        summary = summary.rename(columns=rename)

        # Format percentages
        for col in summary.columns:
            if "%" in col or col in ["FG%", "Pull-Up Rate"]:
                summary[col] = (summary[col] * 100).round(1)
            elif col in ["Avg SDI", "Avg Dist"]:
                summary[col] = summary[col].round(2)

        st.dataframe(summary, use_container_width=True, hide_index=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# All players table
# ---------------------------------------------------------------------------

st.subheader("Player Details")

player_display_cols = [c for c in [
    "player", "archetype", "role", "avg_sdi", "actual_fg_pct",
    "pullup_rate", "pct_rim", "pct_3pt", "usage_pct", "total_attempts",
] if c in filtered.columns]

if player_display_cols:
    show = filtered[player_display_cols].copy()
    for col in show.columns:
        if col in ["actual_fg_pct", "pullup_rate", "pct_rim", "pct_3pt", "usage_pct"]:
            show[col] = (show[col] * 100).round(1)
        elif col in ["avg_sdi"]:
            show[col] = show[col].round(3)

    col_rename = {
        "player": "Player", "archetype": "Archetype", "role": "Role",
        "avg_sdi": "SDI", "actual_fg_pct": "FG%", "pullup_rate": "Pull-Up%",
        "pct_rim": "Rim%", "pct_3pt": "3PT%", "usage_pct": "USG%",
        "total_attempts": "Attempts",
    }
    show = show.rename(columns=col_rename).sort_values("SDI", ascending=False)
    st.dataframe(show, use_container_width=True, height=400, hide_index=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Pre-generated figures
# ---------------------------------------------------------------------------

st.subheader("Clustering Analysis Figures")

figure_files = {
    "Player Archetypes Scatter": FIGURES_DIR / "player_archetypes_scatter.png",
    "GMM BIC by Role": FIGURES_DIR / "gmm_bic_by_role.png",
    "Player Metric Correlations": FIGURES_DIR / "player_metric_correlations.png",
}

cols = st.columns(len(figure_files))
for i, (label, path) in enumerate(figure_files.items()):
    with cols[i]:
        if path.exists():
            st.markdown(f"**{label}**")
            st.image(str(path), use_container_width=True)
        else:
            st.caption(f"{label}: not found")
