"""Curated GAM explorer for the poster demo."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from demo_content import GAM_CORE_FIGURES, GAM_EXTRA_GROUPS
from app_utils import MUTED_TEXT, apply_theme, build_nhl_distance_comparison_figure


def figure_is_available(figure: dict[str, object]) -> bool:
    if figure.get("renderer") == "nhl_distance_comparison":
        data_paths = figure.get("data_paths", {})
        return all(Path(path).exists() for path in data_paths.values())
    path = figure.get("path")
    return bool(path) and Path(path).exists()


def render_figure_card(figure: dict[str, object]) -> None:
    if figure.get("renderer") == "nhl_distance_comparison":
        data_paths = figure["data_paths"]
        comparison_fig = build_nhl_distance_comparison_figure(
            all_shots_data_path=Path(data_paths["all_shots"]),
            non_empty_net_data_path=Path(data_paths["non_empty_net"]),
            title=str(figure["title"]),
            markers=list(figure.get("markers", [])),
        )
        st.pyplot(comparison_fig, use_container_width=True)
        plt.close(comparison_fig)
        st.caption(str(figure["caption"]))
        return

    path = Path(figure["path"])
    if not path.exists():
        st.warning(f"Missing figure: {path.name}")
        return
    st.image(str(path), use_container_width=True)
    st.caption(str(figure["caption"]))


def render_extra_group(group: dict[str, object]) -> None:
    with st.expander(group["group_title"], expanded=False):
        st.markdown(
            f'<div class="panel-copy" style="margin-bottom:1rem;color:{MUTED_TEXT};">{group["blurb"]}</div>',
            unsafe_allow_html=True,
        )
        figures = [fig for fig in group["figures"] if figure_is_available(fig)]
        if not figures:
            st.info("No figures available in this group.")
            return

        for start in range(0, len(figures), 2):
            cols = st.columns(min(2, len(figures) - start))
            for col, fig in zip(cols, figures[start : start + 2]):
                with col:
                    render_figure_card(fig)


def main() -> None:
    apply_theme()
    st.title("📈 GAM Explorer")
    st.markdown(
        f'<div class="panel-copy" style="font-size:1rem;color:{MUTED_TEXT};">'
        f'This page is now GAM-only: curated continuous-effect panels and distance comparisons, with discrete summaries and spatial surfaces removed.'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Core Distance Effects")
    st.markdown(
        "- These are the two headline distance figures used in the final poster story.\n"
        "- NBA keeps the spline distance fit as the core panel, while NHL uses the current distance GAM and the support views show how the tail changes once empty-net attempts are removed."
    )

    core_cols = st.columns(2, gap="large")
    for col, figure in zip(core_cols, GAM_CORE_FIGURES):
        with col:
            render_figure_card(figure)

    st.divider()
    nba_tab, nhl_tab = st.tabs(["NBA Extras", "NHL Extras"])
    with nba_tab:
        st.markdown(
            f'<div class="panel-copy" style="color:{MUTED_TEXT};">'
            f'NBA support panels are limited to continuous GAM views only: alternate distance, angle, clock, and period.'
            f"</div>",
            unsafe_allow_html=True,
        )
        for group in GAM_EXTRA_GROUPS["NBA"]:
            render_extra_group(group)

    with nhl_tab:
        st.markdown(
            f'<div class="panel-copy" style="color:{MUTED_TEXT};">'
            f'NHL support panels stay GAM-only as well: a combined all-shots vs non-empty-net distance overlay, the standalone non-empty-net refit, the NHL spline view, and the angle, clock, and period effects.'
            f"</div>",
            unsafe_allow_html=True,
        )
        for group in GAM_EXTRA_GROUPS["NHL"]:
            render_extra_group(group)


main()
