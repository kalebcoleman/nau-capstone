"""NHL Research Figures — dynamic replication of RMarkdown analysis figures."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import APP_BG, PANEL_BG, MUTED_TEXT, TEXT_COLOR, ACCENT, apply_theme
from nhl_utils import load_nhl_shots_data, NHL_SHOTS_DATA_PATH

st.set_page_config(page_title="NHL Research Figures", layout="wide")
apply_theme()
st.title("🏒 NHL Research Figures")

if not NHL_SHOTS_DATA_PATH.exists():
    st.error(f"Sample data not found: {NHL_SHOTS_DATA_PATH}. Have you run the data preprocessor?")
    st.stop()

st.markdown(
    f'<div style="color: {MUTED_TEXT}; margin-bottom: 24px;">'
    f"Dynamic gallery of NHL analysis figures ported from the RMarkdown capstone pipeline."
    f"</div>",
    unsafe_allow_html=True,
)

# Load data
@st.cache_data(show_spinner="Loading data for figures...")
def get_figure_data():
    df = load_nhl_shots_data()
    df["shotDistance"] = pd.to_numeric(df["shotDistance"], errors="coerce")
    df["xGoal"] = pd.to_numeric(df["xGoal"], errors="coerce")
    df["goal"] = pd.to_numeric(df["goal"], errors="coerce")
    return df

df = get_figure_data()

if df.empty:
    st.warning("Data is empty.")
    st.stop()

st.header("1. Scored vs Expected Goals by Distance")
with st.container():
    df_dist = df.dropna(subset=["shotDistance", "goal", "xGoal"]).copy()
    df_dist["distance_group"] = (df_dist["shotDistance"] // 10) * 10
    dist_agg = df_dist.groupby("distance_group").agg(
        Actual=("goal", "mean"),
        Expected=("xGoal", "mean")
    ).reset_index()
    dist_agg = dist_agg[dist_agg["distance_group"] <= 100]  # Reasonable hockey distance

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        y=dist_agg["distance_group"], x=dist_agg["Actual"],
        name="Actual Goals", orientation='h', marker_color="steelblue"
    ))
    fig1.add_trace(go.Bar(
        y=dist_agg["distance_group"], x=dist_agg["Expected"],
        name="Expected Goals", orientation='h', marker_color="darkred"
    ))
    fig1.update_layout(
        barmode='group',
        yaxis_title="Distance from Goal (feet)",
        xaxis_title="Goal Probability",
        paper_bgcolor=APP_BG, plot_bgcolor=PANEL_BG, font=dict(color=TEXT_COLOR)
    )
    st.plotly_chart(fig1, use_container_width=True)

st.divider()

# Compute player stats for the rest of the figures
@st.cache_data(show_spinner="Computing player statistics...")
def get_player_stats(df):
    df_play = df.dropna(subset=["shooterName"]).copy()
    df_play['residual'] = df_play['goal'] - df_play['xGoal']
    
    stats = df_play.groupby("shooterName").agg(
        shots=("goal", "count"),
        goal_mean=("goal", "mean"),
        xGoal_mean=("xGoal", "mean"),
        residual_mean=("residual", "mean"),
        SDI_mean=("SDI", "mean"),
        goals=("goal", "sum")
    ).reset_index()
    return stats

player_stats = get_player_stats(df)

st.header("2. Top 10 Overperformers & Underperformers vs xG")
with st.container():
    min_shots = 100
    res_stats = player_stats[player_stats["shots"] >= min_shots].copy()
    res_stats = res_stats.sort_values("residual_mean", ascending=False)
    
    if not res_stats.empty:
        top_10 = res_stats.head(10)
        bot_10 = res_stats.tail(10)
        plot_data = pd.concat([top_10, bot_10])
        plot_data["residual_pct"] = plot_data["residual_mean"] * 100
        plot_data["color"] = np.where(plot_data["residual_pct"] > 0, "steelblue", "darkred")
        
        fig2 = px.bar(
            plot_data, x="residual_pct", y="shooterName", orientation='h',
            labels={"residual_pct": "Residual (Actual - Expected Goal %)", "shooterName": "Player"},
            color="color", color_discrete_map="identity"
        )
        fig2.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor=APP_BG, plot_bgcolor=PANEL_BG, font=dict(color=TEXT_COLOR))
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

st.header("3. Shot Difficulty vs Actual Goal Percentage")
with st.container():
    min_shots = 100
    sdi_stats = player_stats[player_stats["shots"] >= min_shots].copy()
    
    if not sdi_stats.empty:
        fig3 = px.scatter(
            sdi_stats, x="SDI_mean", y="goal_mean", size="shots", color="residual_mean",
            color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
            hover_name="shooterName",
            labels={"SDI_mean": "Shot Difficulty Index (SDI)", "goal_mean": "Actual Goal %", "residual_mean": "FG% Residual"},
        )
        # Add trendline approx using numpy polyfit
        z = np.polyfit(sdi_stats["SDI_mean"], sdi_stats["goal_mean"], 1)
        p = np.poly1d(z)
        fig3.add_trace(go.Scatter(
            x=sdi_stats["SDI_mean"], y=p(sdi_stats["SDI_mean"]),
            mode="lines", name="Trend", line=dict(color="blue", dash="dash")
        ))
        
        fig3.update_layout(paper_bgcolor=APP_BG, plot_bgcolor=PANEL_BG, font=dict(color=TEXT_COLOR))
        st.plotly_chart(fig3, use_container_width=True)
        
st.divider()

st.header("4. Shot Volume vs Goal Percentage")
with st.container():
    min_shots = 50
    vol_stats = player_stats[player_stats["shots"] >= min_shots].copy()
    vol_stats["goal_pct"] = vol_stats["goal_mean"] * 100
    
    if not vol_stats.empty:
        fig4 = px.scatter(
            vol_stats, x="shots", y="goal_pct", hover_name="shooterName",
            labels={"shots": "Number of Shots", "goal_pct": "Goal Percentage (%)"},
        )
        fig4.update_traces(marker=dict(color="steelblue", size=8, opacity=0.6))
        
        z2 = np.polyfit(vol_stats["shots"], vol_stats["goal_pct"], 1)
        p2 = np.poly1d(z2)
        fig4.add_trace(go.Scatter(
            x=vol_stats["shots"], y=p2(vol_stats["shots"]),
            mode="lines", name="Trend", line=dict(color="darkred", dash="dash")
        ))
        fig4.update_layout(paper_bgcolor=APP_BG, plot_bgcolor=PANEL_BG, font=dict(color=TEXT_COLOR))
        st.plotly_chart(fig4, use_container_width=True)

st.divider()

st.header("5. Goal Percentage: Players with Most vs Least Shot Volume")
with st.container():
    min_shots = 100
    vbar_stats = player_stats[player_stats["shots"] >= min_shots].copy()
    vbar_stats["goal_pct"] = vbar_stats["goal_mean"] * 100
    vbar_stats = vbar_stats.sort_values("shots", ascending=False)
    
    if not vbar_stats.empty:
        most_shots = vbar_stats.head(10).copy()
        least_shots = vbar_stats.tail(10).copy()
        most_shots["category"] = "Most Shots"
        least_shots["category"] = "Least Shots"
        plot_vol = pd.concat([most_shots, least_shots])
        
        fig5 = px.bar(
            plot_vol, x="goal_pct", y="shooterName", color="category", orientation='h',
            labels={"goal_pct": "Goal Percentage (%)", "shooterName": "Player"},
            color_discrete_map={"Most Shots": "darkblue", "Least Shots": "darkred"},
            text="shots"
        )
        fig5.update_traces(texttemplate='%{text} shots', textposition='outside')
        fig5.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor=APP_BG, plot_bgcolor=PANEL_BG, font=dict(color=TEXT_COLOR))
        st.plotly_chart(fig5, use_container_width=True)

st.divider()

st.header("6. Goal Percentage: Players with Most vs Least Goals Scored")
with st.container():
    min_goals = 10
    gbar_stats = player_stats[(player_stats["goals"] >= min_goals) & (player_stats["shots"] >= 50)].copy()
    gbar_stats["goal_pct"] = gbar_stats["goal_mean"] * 100
    gbar_stats = gbar_stats.sort_values("goals", ascending=False)
    
    if not gbar_stats.empty:
        most_goals = gbar_stats.head(10).copy()
        least_goals = gbar_stats.tail(10).copy()
        most_goals["category"] = "Most Goals"
        least_goals["category"] = "Least Goals"
        plot_goal = pd.concat([most_goals, least_goals])
        
        fig6 = px.bar(
            plot_goal, x="goal_pct", y="shooterName", color="category", orientation='h',
            labels={"goal_pct": "Goal Percentage (%)", "shooterName": "Player"},
            color_discrete_map={"Most Goals": "darkblue", "Least Goals": "darkred"},
            text="goals"
        )
        fig6.update_traces(texttemplate='%{text} goals', textposition='outside')
        fig6.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor=APP_BG, plot_bgcolor=PANEL_BG, font=dict(color=TEXT_COLOR))
        st.plotly_chart(fig6, use_container_width=True)

