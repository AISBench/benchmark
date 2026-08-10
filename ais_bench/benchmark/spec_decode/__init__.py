# Speculative Decoding Metrics Collection
#
# This module provides client-side collection of speculative decoding performance
# metrics from vLLM-compatible inference servers via the Prometheus /metrics endpoint.
#
# Architecture:
#   snapshot.py  - Data model + Prometheus text format parsing
#   fetcher.py   - HTTP fetching of /metrics
#   calculator.py - Before/after delta computation + derived metrics
#   reporter.py  - Console and file output formatting

from ais_bench.benchmark.spec_decode.snapshot import SpecDecodeSnapshot, parse_spec_decode_metrics
from ais_bench.benchmark.spec_decode.fetcher import (
    fetch_spec_decode_metrics_with_error,
)
from ais_bench.benchmark.spec_decode.calculator import compute_spec_decode_stats
from ais_bench.benchmark.spec_decode.reporter import (
    format_spec_decode_console,
    format_spec_decode_na,
    save_spec_decode_result,
)

__all__ = [
    "SpecDecodeSnapshot",
    "parse_spec_decode_metrics",
    "fetch_spec_decode_metrics_with_error",
    "compute_spec_decode_stats",
    "format_spec_decode_console",
    "format_spec_decode_na",
    "save_spec_decode_result",
]
