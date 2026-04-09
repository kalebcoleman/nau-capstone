"""Minimal shared utilities for the poster-driven Streamlit demo."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from path_setup import ensure_project_paths

ensure_project_paths()

from feature_spec import poster_model_snapshot_frame


APP_BG = "#0B0F1A"
PANEL_BG = "#121826"
TEXT_COLOR = "#E6E8EE"
MUTED_TEXT = "#98A1B3"
ACCENT = "#F5C84C"


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


@st.cache_data(show_spinner=False)
def load_poster_snapshot_data() -> pd.DataFrame:
    return poster_model_snapshot_frame()
