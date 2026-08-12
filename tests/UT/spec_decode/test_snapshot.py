"""Unit tests for spec_decode.snapshot — Prometheus text format parser."""

import pytest
from ais_bench.benchmark.spec_decode.snapshot import (
    SpecDecodeSnapshot,
    parse_spec_decode_metrics,
)


# ---------------------------------------------------------------------------
#  Standard Prometheus text fixture containing spec decode counters.
#  Covers: HELP/TYPE lines (should be skipped), scientific notation (1.542e4),
#  multiple "position" labels, and a non-spec-decode metric (should be
#  ignored).
# ---------------------------------------------------------------------------
_VALID_METRICS_TEXT = """\
# HELP vllm:spec_decode_num_drafts_total Number of spec decoding drafts.
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{engine="0"} 15420.0

# HELP vllm:spec_decode_num_draft_tokens_total Number of draft tokens.
# TYPE vllm:spec_decode_num_draft_tokens_total counter
vllm:spec_decode_num_draft_tokens_total{engine="0"} 7.71e4

# HELP vllm:spec_decode_num_accepted_tokens_total Number of accepted tokens.
# TYPE vllm:spec_decode_num_accepted_tokens_total counter
vllm:spec_decode_num_accepted_tokens_total{engine="0"} 50115.0

# HELP vllm:spec_decode_num_accepted_tokens_per_pos_total Accepted tokens per draft position.
# TYPE vllm:spec_decode_num_accepted_tokens_per_pos_total counter
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="0"} 15420.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="1"} 13878.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="2"} 1.1565e4

# Some other metric that should be ignored
vllm:request_success_total{engine="0"} 9999.0
"""

# Metrics text that contains NO spec decode lines —
# simulating a server where speculative decoding is not enabled.
_NO_SPEC_DECODE_TEXT = """\
# HELP vllm:request_success_total Number of successful requests.
# TYPE vllm:request_success_total counter
vllm:request_success_total{engine="0"} 500.0

# HELP vllm:num_requests_running Number of requests running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0"} 3.0
"""


class TestParseSpecDecodeMetrics:
    """Tests for parse_spec_decode_metrics – the core Prometheus parser."""

    def test_parse_valid_metrics(self):
        """Parse standard Prometheus text → correct SpecDecodeSnapshot."""
        snapshot = parse_spec_decode_metrics(_VALID_METRICS_TEXT)

        assert snapshot is not None, "Should return a snapshot when metrics are present"

        # Basic counters
        assert snapshot.num_drafts == 15420
        assert snapshot.num_draft_tokens == 77100  # 7.71e4
        assert snapshot.num_accepted_tokens == 50115

        # Per-position acceptance (3 positions in fixture)
        assert len(snapshot.accepted_per_pos) == 3
        assert snapshot.accepted_per_pos[0] == 15420
        assert snapshot.accepted_per_pos[1] == 13878
        assert snapshot.accepted_per_pos[2] == 11565  # 1.1565e4

    def test_parse_no_spec_decode(self):
        """Metrics text without spec decode lines → returns None."""
        snapshot = parse_spec_decode_metrics(_NO_SPEC_DECODE_TEXT)
        assert snapshot is None


class TestSpecDecodeSnapshot:
    """Tests for the SpecDecodeSnapshot dataclass defaults."""

    def test_defaults(self):
        """All fields should default to zero / empty dict."""
        s = SpecDecodeSnapshot()
        assert s.num_drafts == 0
        assert s.num_draft_tokens == 0
        assert s.num_accepted_tokens == 0
        assert s.accepted_per_pos == {}
