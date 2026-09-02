"""Test configuration for the standalone herding coreset selector."""

import sys
from pathlib import Path


TOOL_ROOT = (
    Path(__file__).resolve().parents[4] / "tools" / "herding_coreset_selector"
)

if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))
