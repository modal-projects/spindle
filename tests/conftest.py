"""Repository imports and the Miles runtime commit for CPU tests."""

import os
from pathlib import Path
import sys

os.environ.setdefault("SPINDLE_MILES_COMMIT", "a" * 40)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
