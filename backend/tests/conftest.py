"""Pytest wiring so `tests/` can import names from `backend/main.py`.

Without this, pytest run from `backend/` can't resolve `from main import ...`
because the cwd isn't automatically on sys.path for test discovery.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
