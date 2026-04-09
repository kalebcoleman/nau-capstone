"""Entry point for the repo-front Streamlit demo."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from path_setup import ensure_project_paths

ensure_project_paths()

from demo_content import NAVIGATION_PAGES
from app_utils import apply_theme


def main() -> None:
    st.set_page_config(
        page_title="Cross-Sport Shot Analysis",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()

    pages = [
        st.Page(
            APP_DIR / page["path"],
            title=page["title"],
            icon=page["icon"],
            default=page["default"],
        )
        for page in NAVIGATION_PAGES
    ]

    navigation = st.navigation(pages, position="sidebar", expanded=True)
    navigation.run()


main()
