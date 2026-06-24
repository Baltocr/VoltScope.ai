"""Portfolio entrypoint for the VoltScope AI Streamlit app.

The historical Streamlit module is kept in ``MVP 1/`` to avoid breaking existing
saved workspaces while the project is being organized into a cleaner package
layout. Run this file with:

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.legacy_app import load_legacy_app


def running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    logger = logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context")
    previous_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        return get_script_run_ctx() is not None
    finally:
        logger.setLevel(previous_level)


def main() -> None:
    app = load_legacy_app()
    app.main()


if __name__ == "__main__":
    if running_under_streamlit():
        main()
    else:
        print("Run VoltScope AI with: streamlit run app/streamlit_app.py")
