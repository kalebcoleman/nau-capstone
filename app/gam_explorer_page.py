"""Curated GAM explorer for the poster demo."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from demo_content import GAM_CORE_FIGURES, GAM_EXTRA_GROUPS
from app_utils import MUTED_TEXT, apply_theme


def render_figure_card(title: str, path: Path, caption: str) -> None:
    if not path.exists():
        st.warning(f"Missing figure: {path.name}")
        return
    st.image(str(path), use_container_width=True)
    st.caption(caption)


def render_extra_group(group: dict[str, object]) -> None:
    with st.expander(group["group_title"], expanded=False):
        st.markdown(
            f'<div class="panel-copy" style="margin-bottom:1rem;color:{MUTED_TEXT};">{group["blurb"]}</div>',
            unsafe_allow_html=True,
        )
        figures = [fig for fig in group["figures"] if Path(fig["path"]).exists()]
        if not figures:
            st.info("No figures available in this group.")
            return

        for start in range(0, len(figures), 2):
            cols = st.columns(min(2, len(figures) - start))
            for col, fig in zip(cols, figures[start : start + 2]):
                with col:
                    render_figure_card(fig["title"], Path(fig["path"]), fig["caption"])


def main() -> None:
    apply_theme()
    st.title("📈 GAM Explorer")
    st.markdown(
        f'<div class="panel-copy" style="font-size:1rem;color:{MUTED_TEXT};">'
        f'This page shows the centralized expected FG/xG GAM outputs used for the demo, with matched naming and styling across both sports.'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Core Distance Effects")
    st.markdown(
        "- These are the two headline partial dependence plots for the cross-sport demo.\n"
        "- Both are full-window expected FG/xG GAM effects with the same visual language and 95% confidence intervals."
    )

    core_cols = st.columns(2, gap="large")
    for col, figure in zip(core_cols, GAM_CORE_FIGURES):
        with col:
            render_figure_card(figure["title"], Path(figure["path"]), figure["caption"])

    st.divider()
    nba_tab, nhl_tab = st.tabs(["NBA Extras", "NHL Extras"])
    with nba_tab:
        st.markdown(
            f'<div class="panel-copy" style="color:{MUTED_TEXT};">'
            f'Full-window NBA expected FG GAM outputs covering continuous effects, discrete controls, and the spatial surface.'
            f"</div>",
            unsafe_allow_html=True,
        )
        for group in GAM_EXTRA_GROUPS["NBA"]:
            render_extra_group(group)

    with nhl_tab:
        st.markdown(
            f'<div class="panel-copy" style="color:{MUTED_TEXT};">'
            f'Full-window NHL expected goal GAM outputs using all shots, including the added period-time effect and the key discrete controls.'
            f"</div>",
            unsafe_allow_html=True,
        )
        for group in GAM_EXTRA_GROUPS["NHL"]:
            render_extra_group(group)


main()
