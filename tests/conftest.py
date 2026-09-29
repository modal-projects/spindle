"""The Miles runtime reads this commit from the environment."""

import os

os.environ.setdefault("SPINDLE_MILES_COMMIT", "a" * 40)
