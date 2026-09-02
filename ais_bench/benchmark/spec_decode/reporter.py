import json
import os
import os.path as osp
import re

from ais_bench.benchmark.utils.logging.logger import AISLogger

logger = AISLogger()

_COL_WIDTH = 45


def _make_title(url: str = "") -> str:
    """Build the console title line with optional URL label."""
    base = "========== Speculative Decoding Metrics"
    label = f"  [{_url_label(url)}]" if url else ""
    return f"{base}{label} =========="


def format_spec_decode_console(stats: dict, url: str = "") -> str:
    """Format spec decode statistics as a human-readable table."""
    title = _make_title(url)
    sep = "=" * (len(title) - 2)
    lines = [
        "",
        sep,
        title,
        sep,
        _fmt_line("Acceptance rate (%)", f"{stats['acceptance_rate']:.2f}"),
        _fmt_line("Acceptance length", f"{stats['acceptance_length']:.2f}"),
        _fmt_line("Drafts", str(stats["num_drafts"])),
        _fmt_line("Draft tokens", str(stats["draft_tokens"])),
        _fmt_line("Accepted tokens", str(stats["accepted_tokens"])),
        _fmt_line(
            "Per-position acceptance rates",
            _format_per_pos_rates(stats.get("per_position_acceptance_rates", {})),
        ),
    ]
    return "\n".join(lines)


def format_spec_decode_na(url: str = "", error_message: str | None = None) -> str:
    """Format a N/A spec decode block with optional error reason."""
    title = _make_title(url)
    sep = "=" * (len(title) - 2)
    lines = [
        "",
        sep,
        title,
        sep,
        _fmt_line("Status", "N/A"),
    ]
    if error_message:
        lines.append(_fmt_line("Reason", error_message))
    return "\n".join(lines)


def _fmt_line(label: str, value: str) -> str:
    """Format a single label-value line with consistent alignment."""
    return f"{label:<{_COL_WIDTH}} {value}"


def _format_per_pos_rates(rates: dict) -> str:
    """Format per-position acceptance rates dict."""
    formatted = [f"{pos}: {rate:.4f}" for pos, rate in sorted(rates.items())]
    return "{" + ", ".join(formatted) + "}"


def _url_label(url: str) -> str:
    """Extract a human-readable label from a metrics URL.

    "http://10.0.0.1:8080/metrics" → "10.0.0.1:8080"
    """
    return re.sub(r"^https?://", "", url).rstrip("/").replace("/metrics", "")


def _url_to_key(url: str) -> str:
    """Convert a metrics URL to a filesystem-safe key.

    "http://10.0.0.1:8080/metrics" → "10.0.0.1_8080"
    """
    label = _url_label(url)
    return re.sub(r"[^a-zA-Z0-9._-]", "_", label)


def save_spec_decode_result(
    spec_stats: dict | None,
    spec_error: str | None,
    work_dir: str,
    url: str,
    before_snapshot=None,
    after_snapshot=None,
) -> None:
    """Save per-URL spec decode results to a JSON file.

    Creates performances/spec_decode_{url_key}.json so that each server
    gets its own independent spec decode statistics file.  Raw
    before/after Prometheus counter snapshots are included alongside the
    computed derived metrics for debugging and traceability.

    Args:
        spec_stats: The computed spec decode statistics dict, or None.
        spec_error: Error description if collection failed, or None.
        work_dir: The benchmark work directory.
        url: The metrics URL this data belongs to.
        before_snapshot: Raw SpecDecodeSnapshot from before the benchmark.
        after_snapshot: Raw SpecDecodeSnapshot from after the benchmark.
    """
    output_dir = osp.join(work_dir, "performances")
    os.makedirs(output_dir, exist_ok=True)

    url_key = _url_to_key(url)
    result = {
        "status": "ok" if spec_stats is not None else "na",
        "url": url,
        "error": spec_error,
        "data": spec_stats,
        "raw": {
            "before": before_snapshot.to_dict() if before_snapshot else None,
            "after": after_snapshot.to_dict() if after_snapshot else None,
        },
    }

    output_path = osp.join(output_dir, f"spec_decode_{url_key}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logger.debug(
        "Spec decode result saved to %s (url=%s, status=%s)",
        output_path, url, result["status"],
    )
