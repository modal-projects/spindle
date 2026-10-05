"""Make the repository's executable scripts importable under the pytest entrypoint."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
