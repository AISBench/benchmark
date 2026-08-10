from ais_bench.benchmark.spec_decode.snapshot import SpecDecodeSnapshot


def compute_spec_decode_stats(
    before: SpecDecodeSnapshot,
    after: SpecDecodeSnapshot,
) -> dict | None:
    """Compute derived spec decode metrics from before/after snapshots.

    All metrics are based on the delta between the two snapshots, isolating
    only the activity that occurred during the benchmark window.

    Args:
        before: Snapshot taken before the benchmark inference.
        after: Snapshot taken after the benchmark inference.

    Returns:
        A dict of derived metrics:
        {
            "num_drafts": int,                          # draft-and-verify cycles
            "draft_tokens": int,                        # total candidate tokens proposed
            "accepted_tokens": int,                     # total tokens accepted
            "acceptance_rate": float,                   # percentage (0-100)
            "acceptance_length": float,                 # avg tokens per forward pass
            "per_position_acceptance_rates": {int: float},  # position → acceptance rate
        }
        Returns None if delta_draft_tokens <= 0 (no spec decode activity).
    """
    delta_drafts = after.num_drafts - before.num_drafts
    delta_draft_tokens = after.num_draft_tokens - before.num_draft_tokens
    delta_accepted = after.num_accepted_tokens - before.num_accepted_tokens

    if delta_draft_tokens <= 0:
        return None

    per_pos_rates: dict[int, float] = {}
    if delta_drafts > 0:
        all_positions = sorted(
            set(before.accepted_per_pos.keys()) | set(after.accepted_per_pos.keys())
        )
        for pos in all_positions:
            before_val = before.accepted_per_pos.get(pos, 0)
            after_val = after.accepted_per_pos.get(pos, before_val)
            delta_pos = max(after_val - before_val, 0)
            per_pos_rates[pos] = delta_pos / delta_drafts

    acceptance_rate = (delta_accepted / delta_draft_tokens) * 100

    acceptance_length = (
        1 + (delta_accepted / delta_drafts) if delta_drafts > 0 else 0.0
    )

    return {
        "num_drafts": delta_drafts,
        "draft_tokens": delta_draft_tokens,
        "accepted_tokens": delta_accepted,
        "acceptance_rate": acceptance_rate,
        "acceptance_length": acceptance_length,
        "per_position_acceptance_rates": per_pos_rates,
    }
