"""Compatibility loader for the original VoltScope Streamlit app module."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_APP_PATH = REPO_ROOT / "MVP 1" / "voltscope_ai_mvp.py"
MODULE_NAME = "voltscope_ai_mvp"


def load_legacy_app() -> ModuleType:
    """Load the original app module without requiring ``MVP 1`` on sys.path."""
    if MODULE_NAME in sys.modules:
        return sys.modules[MODULE_NAME]

    spec = importlib.util.spec_from_file_location(MODULE_NAME, LEGACY_APP_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load VoltScope app from {LEGACY_APP_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module
