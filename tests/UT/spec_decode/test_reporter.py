"""Unit tests for spec_decode.reporter — console and file output formatting."""

import json
import os
import tempfile
import pytest
from ais_bench.benchmark.spec_decode.reporter import (
    format_spec_decode_console,
    format_spec_decode_na,
    save_spec_decode_result,
)
from ais_bench.benchmark.spec_decode.snapshot import SpecDecodeSnapshot


# Sample stats dict (as returned by compute_spec_decode_stats)
_SAMPLE_STATS = {
    "num_drafts": 15420,
    "draft_tokens": 77100,
    "accepted_tokens": 50115,
    "acceptance_rate": 65.0,
    "acceptance_length": 4.25,
    "per_position_acceptance_rates": {0: 1.0, 1: 0.9, 2: 0.75, 3: 0.5, 4: 0.3},
}

_SAMPLE_URL = "http://10.0.0.1:8080/metrics"


class TestFormatSpecDecodeConsole:
    """Tests for format_spec_decode_console."""

    def test_format_with_url(self):
        """Output includes the URL label and all metric values."""
        output = format_spec_decode_console(_SAMPLE_STATS, _SAMPLE_URL)

        # URL label should appear as [host:port]
        assert "[10.0.0.1:8080]" in output

        # Key metric values should be present
        assert "Acceptance rate (%)" in output
        assert "65.00" in output
        assert "Acceptance length" in output
        assert "4.25" in output
        assert "Drafts" in output
        assert "15420" in output
        assert "Draft tokens" in output
        assert "77100" in output
        assert "Accepted tokens" in output
        assert "50115" in output
        assert "Per-position acceptance rates" in output
        assert "{0: 1.0000, 1: 0.9000, 2: 0.7500, 3: 0.5000, 4: 0.3000}" in output

    def test_format_without_url(self):
        """Output without URL should NOT contain the URL bracket label."""
        output = format_spec_decode_console(_SAMPLE_STATS)
        # The title line should not have [host:port]
        assert "Metrics  [" not in output


class TestFormatSpecDecodeNA:
    """Tests for format_spec_decode_na."""

    def test_format_na_with_url_and_error(self):
        """N/A output includes URL label, N/A status, and reason."""
        output = format_spec_decode_na(_SAMPLE_URL, "Connection refused")

        assert "[10.0.0.1:8080]" in output
        assert "N/A" in output
        assert "Reason" in output
        assert "Connection refused" in output

    def test_format_na_without_error(self):
        """N/A output without error shows only status."""
        output = format_spec_decode_na(_SAMPLE_URL)
        assert "N/A" in output
        assert "Reason" not in output


class TestSaveSpecDecodeResult:
    """Tests for save_spec_decode_result – file I/O."""

    def test_save_ok_result(self):
        """Save a successful result with raw snapshots → file content check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            before = SpecDecodeSnapshot(
                num_drafts=10, num_draft_tokens=50, num_accepted_tokens=30,
                accepted_per_pos={0: 10, 1: 9},
            )
            after = SpecDecodeSnapshot(
                num_drafts=15430, num_draft_tokens=77150, num_accepted_tokens=50145,
                accepted_per_pos={0: 15430, 1: 13887},
            )
            save_spec_decode_result(
                _SAMPLE_STATS, None, tmpdir, _SAMPLE_URL,
                before_snapshot=before, after_snapshot=after,
            )

            # Verify file exists with correct naming
            files = os.listdir(os.path.join(tmpdir, "performances"))
            assert len(files) == 1
            assert files[0].startswith("spec_decode_")
            assert files[0].endswith(".json")

            # Verify content
            filepath = os.path.join(tmpdir, "performances", files[0])
            with open(filepath, "r", encoding="utf-8") as f:
                result = json.load(f)

            assert result["status"] == "ok"
            assert result["url"] == _SAMPLE_URL
            assert result["error"] is None
            # Note: dict with int keys becomes str keys after JSON round-trip
            assert result["data"]["num_drafts"] == _SAMPLE_STATS["num_drafts"]
            assert result["data"]["acceptance_rate"] == _SAMPLE_STATS["acceptance_rate"]
            assert result["data"]["acceptance_length"] == _SAMPLE_STATS["acceptance_length"]

            # Raw snapshots
            raw = result["raw"]
            assert raw["before"]["num_drafts"] == 10
            assert raw["before"]["num_draft_tokens"] == 50
            assert raw["after"]["num_drafts"] == 15430
            assert raw["after"]["num_draft_tokens"] == 77150
            assert raw["before"]["accepted_per_pos"] == {"0": 10, "1": 9}

    def test_save_na_result(self):
        """Save a failed result → file content shows N/A."""
        with tempfile.TemporaryDirectory() as tmpdir:
            error_msg = "No spec decode metrics found on server"
            save_spec_decode_result(None, error_msg, tmpdir, _SAMPLE_URL)

            files = os.listdir(os.path.join(tmpdir, "performances"))
            filepath = os.path.join(tmpdir, "performances", files[0])
            with open(filepath, "r", encoding="utf-8") as f:
                result = json.load(f)

            assert result["status"] == "na"
            assert result["url"] == _SAMPLE_URL
            assert result["error"] == error_msg
            assert result["data"] is None
