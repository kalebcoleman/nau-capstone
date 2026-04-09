"""Compatibility launcher for the repo-front Streamlit app."""

from __future__ import annotations

import runpy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TARGET_APP = REPO_ROOT / "app" / "streamlit_app.py"

if __name__ == "__main__":
    runpy.run_path(str(TARGET_APP), run_name="__main__")
