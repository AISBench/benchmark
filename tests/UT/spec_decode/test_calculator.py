"""Unit tests for spec_decode.calculator — delta and derived metrics."""

import math
import pytest
from ais_bench.benchmark.spec_decode.snapshot import SpecDecodeSnapshot
from ais_bench.benchmark.spec_decode.calculator import compute_spec_decode_stats


def _snap(num_drafts=0, num_draft_tokens=0, num_accepted=0, per_pos=None):
    """Shorthand factory for SpecDecodeSnapshot in tests."""
    return SpecDecodeSnapshot(
        num_drafts=num_drafts,
        num_draft_tokens=num_draft_tokens,
        num_accepted_tokens=num_accepted,
        accepted_per_pos=per_pos or {},
    )


class TestComputeSpecDecodeStats:
    """Tests for compute_spec_decode_stats – the core delta computation."""

    # ------------------------------------------------------------------
    #  Normal case
    # ------------------------------------------------------------------
    def test_compute_normal(self):
        """Standard before/after pair → correct derived metrics."""
        before = _snap(
            num_drafts=100,
            num_draft_tokens=500,
            num_accepted=300,
            per_pos={0: 100, 1: 90, 2: 75, 3: 50, 4: 30},
        )
        after = _snap(
            num_drafts=15520,   # delta = 15420
            num_draft_tokens=77600,  # delta = 77100
            num_accepted=50415,      # delta = 50115
            per_pos={0: 15520, 1: 13968, 2: 11640, 3: 7760, 4: 4656},
        )

        stats = compute_spec_decode_stats(before, after)
        assert stats is not None

        # Raw deltas
        assert stats["num_drafts"] == 15420
        assert stats["draft_tokens"] == 77100
        assert stats["accepted_tokens"] == 50115

        # Derived: acceptance_rate = (50115 / 77100) * 100
        expected_rate = (50115 / 77100) * 100
        assert math.isclose(stats["acceptance_rate"], expected_rate, rel_tol=1e-9)

        # Derived: acceptance_length = 1 + (50115 / 15420)
        expected_length = 1 + (50115 / 15420)
        assert math.isclose(stats["acceptance_length"], expected_length, rel_tol=1e-9)

        # Per-position rates: delta_pos / delta_drafts
        per_pos = stats["per_position_acceptance_rates"]
        assert len(per_pos) == 5
        assert math.isclose(per_pos[0], 15420 / 15420, rel_tol=1e-9)
        assert math.isclose(per_pos[1], (13968 - 90) / 15420, rel_tol=1e-9)
        assert math.isclose(per_pos[4], (4656 - 30) / 15420, rel_tol=1e-9)

    # ------------------------------------------------------------------
    #  No activity
    # ------------------------------------------------------------------
    def test_compute_no_activity(self):
        """Delta draft tokens = 0 → returns None (no spec decode activity)."""
        before = _snap(num_drafts=10, num_draft_tokens=50, num_accepted=30)
        after = _snap(num_drafts=10, num_draft_tokens=50, num_accepted=30)
        assert compute_spec_decode_stats(before, after) is None

    # ------------------------------------------------------------------
    #  First-time enable: before has no per_pos data
    # ------------------------------------------------------------------
    def test_compute_empty_positions_in_before(self):
        """Before has no per_pos data → positions are built from scratch."""
        before = _snap(num_drafts=0, num_draft_tokens=0, num_accepted=0)
        after = _snap(
            num_drafts=100,
            num_draft_tokens=500,
            num_accepted=300,
            per_pos={0: 100, 1: 90},
        )

        stats = compute_spec_decode_stats(before, after)
        assert stats is not None

        per_pos = stats["per_position_acceptance_rates"]
        assert len(per_pos) == 2
        assert math.isclose(per_pos[0], 1.0, rel_tol=1e-9)
        assert math.isclose(per_pos[1], 0.9, rel_tol=1e-9)
