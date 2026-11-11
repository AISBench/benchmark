"""Shared per-request latency derivation for performance reports.

All returned values are in seconds and unscaled; callers apply their own unit
conversion (e.g. perf_batch multiplies by 1000 for milliseconds) so that the
numbers and units rendered in reports stay unchanged.
"""

import numpy as np


def derive_request_metrics(time_points, output_tokens) -> dict:
    """Derive latency/TTFT/TPOT/ITL from a request's time points (seconds).

    - latency: total request duration
    - ttft: time to first token
    - tpot: per-output-token time after the first token; None when unavailable
    - itl: array of inter-token gaps, empty when there are fewer than 3 points

    The input dtype is not coerced, so integer time points keep their original
    numeric type just like the callers did before.
    """
    latency = time_points[-1] - time_points[0]
    ttft = time_points[1] - time_points[0]
    tpot = None
    if (output_tokens is not None
            and not isinstance(output_tokens, bool)
            and output_tokens > 1):
        tpot = (latency - ttft) / (output_tokens - 1)
    itl = np.diff(time_points[1:]) if len(time_points) > 2 else np.array([])
    return {"latency": latency, "ttft": ttft, "tpot": tpot, "itl": itl}
