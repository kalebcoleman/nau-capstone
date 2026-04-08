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
        f'This page starts with the poster GAM story and then opens up the strongest extra semester outputs without turning the app into a raw figure dump.'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Poster Core: Distance GAMs")
    st.markdown(
        "- Both sports show a strong negative distance effect, but the hockey decline is steeper near the goal.\n"
        "- These are the two GAM results that anchor the cross-sport poster story."
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
            f'Additional NBA GAM outputs from the semester. These help explain which contexts mattered beyond the poster distance curve.'
            f"</div>",
            unsafe_allow_html=True,
        )
        for group in GAM_EXTRA_GROUPS["NBA"]:
            render_extra_group(group)

    with nhl_tab:
        st.markdown(
            f'<div class="panel-copy" style="color:{MUTED_TEXT};">'
            f'Additional NHL GAM outputs and model-comparison views that support the poster results without crowding the main narrative.'
            f"</div>",
            unsafe_allow_html=True,
        )
        for group in GAM_EXTRA_GROUPS["NHL"]:
            render_extra_group(group)


main()
