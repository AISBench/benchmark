"""Normalize terminal decoration in the single benchmark log after child exit."""
import re
from pathlib import Path


def normalize_log(filename):
    path = Path(filename)
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(╛)(?=[^\n])", r"\1\n", text)
    path.write_text(text.rstrip("\n") + "\n", encoding="utf-8")
