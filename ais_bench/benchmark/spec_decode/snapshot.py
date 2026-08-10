import contextlib
from dataclasses import dataclass, field


@dataclass
class SpecDecodeSnapshot:
    """A single snapshot of spec decode counters from GET /metrics.

    All counters are monotonically increasing, process-lifetime cumulative values.
    Use before/after delta to isolate activity during a benchmark window.
    """

    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    accepted_per_pos: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON output."""
        return {
            "num_drafts": self.num_drafts,
            "num_draft_tokens": self.num_draft_tokens,
            "num_accepted_tokens": self.num_accepted_tokens,
            "accepted_per_pos": dict(sorted(self.accepted_per_pos.items())),
        }


def parse_spec_decode_metrics(text: str) -> "SpecDecodeSnapshot | None":
    """Parse Prometheus text format, extracting spec decode counters.

    Args:
        text: Raw response body from GET /metrics.

    Returns:
        A SpecDecodeSnapshot if any spec decode metrics were found, or None
        if the server does not have speculative decoding enabled.
    """
    snapshot = SpecDecodeSnapshot()
    found_any = False

    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("vllm:spec_decode"):
            continue

        parts = line.split(None, 1)
        if len(parts) < 2:
            continue

        metric_name = parts[0].split("{")[0]
        if not metric_name.endswith("_total"):
            continue

        with contextlib.suppress(ValueError):
            val = int(float(parts[-1]))
            found_any = True

            if "num_drafts" in metric_name and "num_draft_tokens" not in metric_name:
                snapshot.num_drafts += val
            elif "num_draft_tokens" in metric_name:
                snapshot.num_draft_tokens += val
            elif "num_accepted_tokens_per_pos" in metric_name:
                pos = _extract_position(line)
                if pos is not None:
                    snapshot.accepted_per_pos[pos] = (
                        snapshot.accepted_per_pos.get(pos, 0) + val
                    )
            elif "num_accepted_tokens" in metric_name:
                snapshot.num_accepted_tokens += val

    return snapshot if found_any else None


def _extract_position(line: str) -> int | None:
    """Extract position=N from a Prometheus metric label string.

    Example:
        vllm:spec_decode_num_accepted_tokens_per_pos_total{...,position="3"} 7710.0
        → returns 3
    """
    marker = 'position="'
    if marker not in line:
        return None
    start = line.index(marker) + len(marker)
    end = line.index('"', start)
    return int(line[start:end])
