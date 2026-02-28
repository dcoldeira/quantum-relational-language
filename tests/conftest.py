"""pytest configuration for the QRL test suite."""

import os
import sys

# Ensure the repo root is on sys.path so `qai` package is importable
# (alongside PYTHONPATH=src which covers the `qrl` package)
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
