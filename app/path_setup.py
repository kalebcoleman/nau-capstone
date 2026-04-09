"""Shared path bootstrapping for the repo-front Streamlit app."""

from __future__ import annotations

import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
ANALYSIS_DIR = REPO_ROOT / "analysis"
NBA_ANALYSIS_DIR = ANALYSIS_DIR / "nba"
NHL_ANALYSIS_DIR = ANALYSIS_DIR / "nhl"


def ensure_project_paths() -> None:
    """Make app-local and analysis-level imports work under Streamlit page execution."""
    for path in (REPO_ROOT, APP_DIR, ANALYSIS_DIR, NBA_ANALYSIS_DIR, NHL_ANALYSIS_DIR):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
