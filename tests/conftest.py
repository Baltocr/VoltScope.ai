"""Pytest configuration for local and CI test discovery.

Some pytest entrypoints do not automatically put the repository root on
``sys.path`` before importing test modules. Add it explicitly so tests can
import the portfolio package layout under ``src/``.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
