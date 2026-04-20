"""Minimal shared utilities for the poster-driven Streamlit demo."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

from path_setup import ensure_project_paths

ensure_project_paths()

from feature_spec import poster_model_snapshot_frame


APP_BG = "#0B0F1A"
PANEL_BG = "#121826"
TEXT_COLOR = "#E6E8EE"
MUTED_TEXT = "#98A1B3"
ACCENT = "#F5C84C"


def _style_axis_text(
    ax,
    *,
    title: str,
    x_label: str,
    y_label: str,
    title_size: float = 15,
    label_size: float = 12,
    tick_size: float = 11,
) -> None:
    ax.set_title(title, fontsize=title_size, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=label_size, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=label_size, fontweight="bold")
    ax.tick_params(axis="both", labelsize=tick_size, width=1.2)
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def _style_legend(legend) -> None:
    if legend is None:
        return
    legend.get_frame().set_linewidth(1.1)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    legend_title = legend.get_title()
    if legend_title is not None:
        legend_title.set_fontweight("bold")


def apply_theme() -> None:
    """Inject the shared poster-demo theme."""
    st.markdown(
        f"""
        <style>
        :root {{
            --app-bg: {APP_BG};
            --panel-bg: {PANEL_BG};
            --text-main: {TEXT_COLOR};
            --text-muted: {MUTED_TEXT};
            --accent: {ACCENT};
        }}
        html, body, [class*="css"] {{
            background: {APP_BG};
            color: {TEXT_COLOR};
            font-family: "Space Grotesk", "IBM Plex Sans", "SF Pro Display",
                         "Segoe UI", sans-serif;
        }}
        .stApp {{
            background: {APP_BG};
        }}
        [data-testid="stSidebar"] {{
            background: {PANEL_BG};
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }}
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {{
            color: {TEXT_COLOR};
        }}
        h1, h2, h3, h4 {{
            color: {TEXT_COLOR};
            letter-spacing: 0.02em;
        }}
        p, span, label {{
            color: {TEXT_COLOR};
        }}
        .stMetric {{
            background: {PANEL_BG};
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 12px;
        }}
        .stMetric div {{
            overflow: visible !important;
            text-overflow: unset !important;
            white-space: nowrap !important;
        }}
        .stMetric [data-testid="stMetricValue"] {{
            font-size: clamp(1.2rem, 2.5vw, 2.2rem) !important;
            white-space: nowrap !important;
        }}
        .stMetric label {{
            color: {MUTED_TEXT};
            font-size: 0.85rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        .hero-shell {{
            background:
                radial-gradient(circle at top right, rgba(245, 200, 76, 0.18), transparent 32%),
                linear-gradient(135deg, rgba(18, 24, 38, 0.98), rgba(11, 15, 26, 0.98));
            border: 1px solid rgba(245, 200, 76, 0.18);
            border-radius: 24px;
            padding: 1.4rem 1.4rem 1.2rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 16px 42px rgba(0, 0, 0, 0.26);
        }}
        .hero-kicker {{
            color: {ACCENT};
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin-bottom: 0.65rem;
        }}
        .hero-title {{
            font-size: clamp(1.8rem, 4vw, 3.2rem);
            font-weight: 700;
            line-height: 1.04;
            margin: 0 0 0.6rem 0;
        }}
        .hero-copy {{
            color: {MUTED_TEXT};
            font-size: 1rem;
            line-height: 1.6;
            max-width: 48rem;
            margin: 0;
        }}
        .hero-chip-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 1rem;
        }}
        .hero-chip {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 999px;
            color: {TEXT_COLOR};
            font-size: 0.88rem;
            padding: 0.5rem 0.82rem;
        }}
        .panel-card {{
            background: {PANEL_BG};
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 1rem 1rem 0.9rem;
            height: 100%;
        }}
        .panel-title {{
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }}
        .panel-copy {{
            color: {MUTED_TEXT};
            font-size: 0.92rem;
            line-height: 1.5;
            margin-bottom: 0.85rem;
        }}
        .snapshot-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.92rem;
        }}
        .snapshot-table th {{
            color: {ACCENT};
            text-align: left;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 0 0 0.6rem 0;
        }}
        .snapshot-table td {{
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            padding: 0.72rem 0;
            vertical-align: top;
        }}
        .share-url {{
            display: block;
            word-break: break-word;
            color: {TEXT_COLOR};
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 0.85rem;
            margin: 0.75rem 0;
            text-decoration: none;
        }}
        @media (max-width: 900px) {{
            .block-container {{
                padding-left: 1rem;
                padding-right: 1rem;
                padding-top: 1rem;
            }}
            .hero-shell {{
                padding: 1.1rem 1rem 1rem;
                border-radius: 18px;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_nhl_distance_comparison_figure(
    *,
    all_shots_data_path: Path,
    non_empty_net_data_path: Path,
    title: str,
    markers: list[dict[str, object]] | None = None,
) -> Figure:
    all_df = pd.read_csv(all_shots_data_path)
    non_empty_df = pd.read_csv(non_empty_net_data_path)

    fig, ax = plt.subplots(figsize=(11, 6.8))
    series_specs = (
        (all_df, "All shots", "#C06C2B"),
        (non_empty_df, "Non-empty-net shots", "#2A6F97"),
    )

    max_x = 0.0
    for frame, label, color in series_specs:
        x_values = pd.to_numeric(frame["x_value"], errors="coerce")
        effects = pd.to_numeric(frame["fitted_effect"], errors="coerce")
        lower_ci = pd.to_numeric(frame["lower_ci"], errors="coerce")
        upper_ci = pd.to_numeric(frame["upper_ci"], errors="coerce")

        ax.plot(x_values, effects, color=color, linewidth=2.5, label=label)
        ax.fill_between(x_values, lower_ci, upper_ci, color=color, alpha=0.18)

        baseline_col = "baseline_value" if "baseline_value" in frame.columns else "baseline_distance"
        baseline_value = float(pd.to_numeric(frame[baseline_col], errors="coerce").iloc[0])
        if pd.notna(baseline_value):
            ax.axvline(
                baseline_value,
                color=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.55,
                label=f"{label} median baseline",
            )

        max_x = max(max_x, float(x_values.max()))

    ax.axhline(0, color="#7A7A7A", linestyle="--", alpha=0.5, label="Baseline")
    for marker in markers or []:
        marker_value = float(marker["value"])
        if marker_value > max_x:
            continue
        ax.axvline(
            marker_value,
            color=str(marker["color"]),
            linestyle=str(marker.get("linestyle", ":")),
            linewidth=1.7,
            alpha=0.9,
            label=str(marker["label"]),
        )

    ax.set_xlim(0, max_x)
    _style_axis_text(
        ax,
        title=title,
        x_label="Shot Distance (feet)",
        y_label="Marginal log-odds contribution",
    )
    ax.grid(alpha=0.2)
    legend = ax.legend(loc="upper right", fontsize=8.5, frameon=True)
    _style_legend(legend)
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def load_poster_snapshot_data() -> pd.DataFrame:
    return poster_model_snapshot_frame()
