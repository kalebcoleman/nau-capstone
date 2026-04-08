"""Shared path bootstrapping for the poster demo app."""

from __future__ import annotations

import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
NBA_ANALYSIS_DIR = APP_DIR.parent
ANALYSIS_DIR = NBA_ANALYSIS_DIR.parent
REPO_ROOT = ANALYSIS_DIR.parent
NHL_ANALYSIS_DIR = ANALYSIS_DIR / "nhl"


def ensure_project_paths() -> None:
    """Make app-local and analysis-level imports work under Streamlit page execution."""
    for path in (APP_DIR, NBA_ANALYSIS_DIR):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
