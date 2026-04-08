"""Poster-oriented overview page for the Streamlit demo."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from demo_content import (
    ABSTRACT_TEXT,
    APP_SUBTITLE,
    APP_TITLE,
    MAIN_RESULT_BULLETS,
    MODEL_MEASURE_BULLETS,
    SDI_SUMMARY_PATHS,
)
from app_utils import MUTED_TEXT, apply_theme, load_poster_snapshot_data


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_share_url(base_url: str) -> str:
    if not base_url:
        return ""
    parts = urlsplit(base_url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(query, doseq=True),
            parts.fragment,
        )
    )


def get_share_base_url() -> str:
    qp_share = st.query_params.get("share_url")
    if isinstance(qp_share, list):
        qp_share = qp_share[0] if qp_share else ""
    if qp_share:
        return str(qp_share).strip()
    return os.getenv("NBA_DASHBOARD_SHARE_URL", "").strip()


def render_qr_code(share_url: str) -> None:
    qr_html = f"""
    <div style="display:flex; justify-content:center; align-items:center; min-height:220px;">
      <div id="poster-qr"></div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js"></script>
    <script>
      const target = document.getElementById("poster-qr");
      const url = {json.dumps(share_url)};
      if (window.QRCode) {{
        new QRCode(target, {{
          text: url,
          width: 200,
          height: 200,
          colorDark: "#F5C84C",
          colorLight: "#121826",
          correctLevel: QRCode.CorrectLevel.M
        }});
      }} else {{
        target.innerHTML = '<div style="color:#98A1B3;font-family:sans-serif;">QR preview unavailable.</div>';
      }}
    </script>
    """
    components.html(qr_html, height=240)


def render_snapshot_table(snapshot_df: pd.DataFrame) -> None:
    rows = []
    for row in snapshot_df.itertuples(index=False):
        rows.append(
            f"""
            <tr>
              <td><strong>{row.concept}</strong></td>
              <td>{row.nba_variables}</td>
              <td>{row.nhl_variables}</td>
              <td>{row.why_it_matters}</td>
            </tr>
            """
        )

    st.markdown(
        f"""
        <section class="panel-card">
          <div class="panel-title">Data Snapshot</div>
          <div class="panel-copy">
            The poster only needs the main cross-sport variables. This table keeps the model story compact and judge-friendly.
          </div>
          <table class="snapshot-table">
            <thead>
              <tr>
                <th>Concept</th>
                <th>NBA</th>
                <th>NHL</th>
                <th>Why it matters</th>
              </tr>
            </thead>
            <tbody>
              {"".join(rows)}
            </tbody>
          </table>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_navigation_links() -> None:
    st.markdown(
        """
        <section class="panel-card">
          <div class="panel-title">Explore The Demo</div>
          <div class="panel-copy">
            Start with the poster summary here, then jump into the interactive SDI view or the curated GAM page.
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.page_link(
            Path(__file__).resolve().parent / "sdi_explorer_page.py",
            label="Open SDI Explorer",
            icon="📊",
            use_container_width=True,
        )
    with col2:
        st.page_link(
            Path(__file__).resolve().parent / "gam_explorer_page.py",
            label="Open GAM Explorer",
            icon="📈",
            use_container_width=True,
        )


def render_share_panel(share_url: str) -> None:
    st.markdown(
        """
        <section class="panel-card">
          <div class="panel-title">Open On Phone</div>
          <div class="panel-copy">
            Use this QR block on the poster so viewers can open the demo on their phones.
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if share_url:
        render_qr_code(share_url)
        st.markdown(
            f'<a class="share-url" href="{share_url}" target="_blank">{share_url}</a>',
            unsafe_allow_html=True,
        )
        st.caption("Use this exact QR block for the printed poster once the app has a public URL.")
    else:
        st.info(
            "Set `NBA_DASHBOARD_SHARE_URL` to the deployed app URL, or open the app with "
            "`?share_url=https://...` to render the poster QR code."
        )


def main() -> None:
    apply_theme()

    nba_summary = load_summary(SDI_SUMMARY_PATHS["NBA"])
    nhl_summary = load_summary(SDI_SUMMARY_PATHS["NHL"])
    snapshot_df = load_poster_snapshot_data()
    share_url = build_share_url(get_share_base_url())

    st.markdown(
        f"""
        <section class="hero-shell">
          <div class="hero-kicker">NAU Capstone Demo</div>
          <div class="hero-title">{APP_TITLE}</div>
          <p class="hero-copy">{APP_SUBTITLE}</p>
          <div class="hero-chip-row">
            <span class="hero-chip">Matched 2014–2024 comparison</span>
            <span class="hero-chip">Poster-driven demo</span>
            <span class="hero-chip">NBA + NHL</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("NBA Players", f"{len(nba_summary):,}" if not nba_summary.empty else "—")
    metric_cols[1].metric("NHL Players", f"{len(nhl_summary):,}" if not nhl_summary.empty else "—")
    metric_cols[2].metric(
        "NBA Attempts",
        f"{int(nba_summary['attempts'].sum()):,}" if not nba_summary.empty else "—",
    )
    metric_cols[3].metric(
        "NHL Shots",
        f"{int(nhl_summary['attempts'].sum()):,}" if not nhl_summary.empty else "—",
    )

    intro_col, share_col = st.columns([1.6, 1], gap="large")
    with intro_col:
        st.markdown("### Abstract")
        st.markdown(
            f'<div class="panel-copy" style="font-size:1rem;color:{MUTED_TEXT};">{ABSTRACT_TEXT}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("### What The Models Measure")
        for bullet in MODEL_MEASURE_BULLETS:
            st.markdown(f"- {bullet}")
        st.markdown("### Main Results")
        for bullet in MAIN_RESULT_BULLETS:
            st.markdown(f"- {bullet}")
    with share_col:
        render_share_panel(share_url)

    render_snapshot_table(snapshot_df)
    render_navigation_links()


main()
