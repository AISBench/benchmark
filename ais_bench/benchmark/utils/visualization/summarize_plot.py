import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ais_bench.benchmark.utils.logging import AISLogger
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import time

# ================== Constants ==================
WEBGL_CONFIG = {
    "scrollZoom": True,
    "plotGlPixelRatio": 1,
    "showLink": False,
    "displaylogo": False,
    "queueLength": 10,
}

AXIS_CONFIG = dict(
    showline=True,
    showgrid=True,
    showticklabels=True,
    gridwidth=0.5,
    gridcolor="rgba(211,211,211,0.5)",
    linecolor="black",
)

# ================== Chunk rendering configuration ==================
MAX_POINTS_PER_TRACE = 10000  # maximum number of points per trace
TIMELINE_POINTS_PER_REQUEST = 3  # each request takes 3 points in the timeline chart (start, end, break)


logger = AISLogger()

# ================== Helper functions ==================
def validate_input_data(
    start_time_list: List[float],
    prefill_latency_list: List[float],
    end_time_list: List[float],
) -> bool:
    """Validate input data"""
    n_requests = len(start_time_list)
    if n_requests == 0:
        logger.warning("No requests to plot!")
        return False

    if n_requests != len(prefill_latency_list) or n_requests != len(end_time_list):
        logger.warning(
            f"Input list lengths mismatch! start_list:{n_requests}, "
            f"prefill_latency_list:{len(prefill_latency_list)}, end_time_list:{len(end_time_list)}"
        )
        return False

    return True


def is_non_streaming_scenario(
    prefill_latency_list: List[float],
) -> bool:
    """Check if it is a non-streaming scenario"""
    return all(p == 0.0 for p in prefill_latency_list)


def preprocess_data(
    start_times: np.ndarray,
    prefill_latencies: np.ndarray,
    end_times: np.ndarray,
) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, bool]:
    """
    Data preprocessing
    返回: (first_token_times, adjusted_starts, adjusted_ends, is_non_streaming)
    """
    # Check if it is a non-streaming scenario
    is_non_streaming = is_non_streaming_scenario(prefill_latencies)

    # Calculate the first token time
    first_token_times = (start_times + prefill_latencies) if not is_non_streaming else None

    # For each request, check if it contains non-first token delay. If so, update the end_time of the request.
    # Because the end_time_list has errors due to the timing of the points, we need to use the value of first_token_time_list to correct it.
    # Only correct the end time in non-streaming scenarios
    if not is_non_streaming:
        no_decode_indices = np.where(np.abs(end_times - first_token_times) < 0.001)[0]
        if no_decode_indices.any():
            end_times[no_decode_indices] = first_token_times[no_decode_indices]
            logger.debug(
                f"Adjusted {len(no_decode_indices)} requests with no decode tokens"
            )
            del no_decode_indices

    # Calculate the global minimum time
    global_x_min = np.min(start_times) if len(start_times) > 0 else 0.0

    # Calculate the relative time
    adjusted_starts = start_times - global_x_min
    adjusted_first_tokens = (
        (first_token_times - global_x_min) if not is_non_streaming else None
    )
    adjusted_ends = end_times - global_x_min

    return adjusted_first_tokens, adjusted_starts, adjusted_ends, is_non_streaming


def generate_timeline_traces(
    adjusted_starts: np.ndarray,
    adjusted_ends: np.ndarray,
    adjusted_first_tokens: np.ndarray,
    multiturn_group_id_list: list,
    unit: str,
) -> List[go.Scattergl]:
    """Generate the trajectory of the request timeline chart"""
    n_requests = len(adjusted_starts)
    if n_requests == 0:
        return []
    unique_ids = set(multiturn_group_id_list)
    # if has repeated ids, it is a multi-turn conversation
    is_multiturn = len(unique_ids) < len(multiturn_group_id_list)
    if is_multiturn:
        logger.info("Visualization in multi-turn conversations...")
    # Pre-allocate memory
    red_x = np.full(TIMELINE_POINTS_PER_REQUEST * n_requests, np.nan, dtype=np.float32)
    red_y = np.full_like(red_x, np.nan)
    blue_x = np.full_like(red_x, np.nan)
    blue_y = np.full_like(red_x, np.nan)
    hover_text = np.full(TIMELINE_POINTS_PER_REQUEST * n_requests, None, dtype=object)
    sorted_indices = np.argsort(adjusted_starts)
    group_id_to_y = {}
    index = 0
    for sorted_pos, orig_idx in enumerate(sorted_indices):
        # Get the key time points of the current request
        start_t = adjusted_starts[orig_idx]
        first_token_t = adjusted_first_tokens[orig_idx]
        end_t = adjusted_ends[orig_idx]
        # Get y value based on the group_id of the current request
        group_id = multiturn_group_id_list[orig_idx]
        if group_id not in group_id_to_y:
            index += 1
            group_id_to_y[group_id] = index
        y = group_id_to_y[group_id] if is_multiturn else sorted_pos + 1

        # Calculate the position in the array
        arr_idx = sorted_pos * 3

        # Red line (TTFT): from start to the first token
        red_x[arr_idx] = start_t
        red_x[arr_idx + 1] = first_token_t
        red_y[arr_idx : arr_idx + 2] = y

        blue_content_data = "NaN"

        # Blue line (Decode): from the first token to the end
        if end_t > first_token_t:
            blue_x[arr_idx] = first_token_t
            blue_x[arr_idx + 1] = end_t
            blue_y[arr_idx : arr_idx + 2] = y
            decode_time = end_t - first_token_t
            blue_content_data = f"{first_token_t:.2f}→{end_t:.2f}={decode_time:.2f}"

        # Hover text, the trigger point is on the start of the red line
        ttft = first_token_t - start_t
        e2e = end_t - start_t

        red_content = f"<span style='color:red'>TTFT({unit}): {start_t:.2f}→{first_token_t:.2f}={ttft:.2f}</span><br>"
        blue_content = (
            f"<span style='color:blue'>Decode({unit}): {blue_content_data}</span><br>"
        )
        e2e_content = f"E2E({unit}): {start_t:.2f}→{end_t:.2f}={e2e:.2f}"
        hover_text[arr_idx] = red_content + blue_content + e2e_content

    # Generate traces in chunks
    traces = []
    n_points = len(red_x)
    chunk_size = min(n_points, MAX_POINTS_PER_TRACE)
    n_chunks = (n_points + chunk_size - 1) // chunk_size

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_points)
        chunk = slice(start_idx, end_idx)

        # Red line
        if np.any(~np.isnan(red_x[chunk])):
            traces.append(
                go.Scattergl(
                    x=red_x[chunk],
                    y=red_y[chunk],
                    mode="lines",
                    line=dict(color="red", width=1, shape="hv"),
                    hoverinfo="text",
                    hovertext=hover_text[chunk],
                    showlegend=False,
                    connectgaps=False,
                )
            )

        # Blue line
        if np.any(~np.isnan(blue_x[chunk])):
            traces.append(
                go.Scattergl(
                    x=blue_x[chunk],
                    y=blue_y[chunk],
                    mode="lines",
                    line=dict(color="blue", width=1, shape="hv"),
                    hoverinfo="none",
                    showlegend=False,
                    connectgaps=False,
                )
            )

    del red_x, red_y, blue_x, blue_y, hover_text
    return traces


def generate_concurrency_traces(
    adjusted_starts: np.ndarray, adjusted_ends: np.ndarray, unit: str
) -> List[go.Scattergl]:
    """Generate the trajectory of the concurrency chart"""
    # Filter zero-length requests
    valid_mask = adjusted_starts < adjusted_ends
    if not np.any(valid_mask):
        logger.warning("No valid requests for concurrency plot!")
        return []

    valid_starts = adjusted_starts[valid_mask]
    valid_ends = adjusted_ends[valid_mask]
    n_events = len(valid_starts) * 2

    # Generate the event array
    events = np.empty((n_events, 2), dtype=np.float32)
    events[: len(valid_starts), 0] = valid_starts
    events[: len(valid_starts), 1] = 1  # Start event
    events[len(valid_starts) :, 0] = valid_ends
    events[len(valid_starts) :, 1] = -1  # End event

    # Stable sorting (start event priority if time is the same)
    sort_indices = np.lexsort((events[:, 1], events[:, 0]))
    events = events[sort_indices]

    # Calculate the concurrency
    unique_times, inverse_indices = np.unique(events[:, 0], return_inverse=True)
    delta_per_time = np.bincount(inverse_indices, weights=events[:, 1])
    cumulative = np.cumsum(delta_per_time)

    conc_times = unique_times
    conc_counts = cumulative

    # Create hover text
    conc_hover_text = [
        f"Time: {t:.4f}{unit}<br>Concurrency: {c:.0f}"
        for t, c in zip(conc_times, conc_counts)
    ]

    # Render in chunks
    traces = []
    n_points = len(conc_times)
    chunk_size = min(n_points, MAX_POINTS_PER_TRACE)
    n_chunks = (n_points + chunk_size - 1) // chunk_size

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_points)

        if i > 0:
            start_idx = max(0, start_idx - 1)  # Ensure continuous

        chunk = slice(start_idx, end_idx)

        traces.append(
            go.Scattergl(
                x=conc_times[chunk],
                y=conc_counts[chunk],
                mode="lines",
                line=dict(color="#4CAF50", width=1, shape="hv"),
                fill="tozeroy",
                fillcolor="rgba(76,175,80,0.1)",
                hoverinfo="text",
                hovertext=conc_hover_text[chunk],
                showlegend=False,
                connectgaps=True,
            )
        )

    # Clean up large arrays and release memory
    del events, sort_indices, unique_times, inverse_indices, delta_per_time, cumulative
    del conc_times, conc_counts, conc_hover_text
    return traces


def create_plot_layout(
    max_time: float, unit: str, has_timeline: bool
) -> Dict[str, Any]:
    """Create the layout configuration of the chart"""
    xaxis_config = dict(
        **AXIS_CONFIG,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikecolor="#666",
        spikedash="dot",
        title=f"Relative Time ({unit})",
        range=[0, max_time],
    )

    yaxis_config = dict(
        **AXIS_CONFIG,
        rangemode="nonnegative",
        tickmode="auto",
        nticks=10,
    )

    if has_timeline:
        # Double chart mode
        return dict(
            height=1200,
            plot_bgcolor="white",
            xaxis1=dict(
                **xaxis_config,
                matches="x2",
            ),
            yaxis1=dict(
                **yaxis_config,
                title="Request Index",
            ),
            xaxis2=dict(
                **xaxis_config,
            ),
            yaxis2=dict(
                **yaxis_config,
                title="Request Concurrency Count",
            ),
            hoverlabel=dict(
                bgcolor="rgba(255,255,255,0.9)", font_size=12, align="left"
            ),
            hovermode="closest",
        )
    else:
        # Single chart mode (only concurrency chart)
        return dict(
            height=600,
            plot_bgcolor="white",
            xaxis=dict(**xaxis_config),
            yaxis=dict(**yaxis_config, title="Request Concurrency Count"),
            hoverlabel=dict(
                bgcolor="rgba(255,255,255,0.9)", font_size=12, align="left"
            ),
            hovermode="closest",
        )


# ================== Main function for external use ==================
def plot_sorted_request_timelines(
    start_times: np.ndarray,
    end_times: np.ndarray,
    prefill_latencies: np.ndarray,
    multiturn_group_id_list: List[str],
    output_file: str = "timeline.html",
    unit: str = "s",
) -> None:
    """Plot the request timeline and concurrency chart"""
    # ===== 1. Data validation and preprocessing =====
    logger.debug("Starting request timeline processing...")

    # Validate input data
    if not validate_input_data(start_times, prefill_latencies, end_times):
        return False

    # Data preprocessing
    preprocess_start = time.perf_counter()
    adjusted_first_token_times, adjusted_starts, adjusted_ends, is_non_streaming = (
        preprocess_data(
            start_times,
            prefill_latencies,
            end_times
        )
    )

    if is_non_streaming:
        logger.warning(
            "[Non-streaming scenario] The plot will only show the request concurrency chart!"
        )

    n_requests = len(start_times)
    has_timeline = (
        not is_non_streaming
        and adjusted_first_token_times is not None
        and n_requests > 0
    )
    max_time = np.max(adjusted_ends) if n_requests > 0 else 1.0

    logger.debug(
        f"Data preprocessing completed in {time.perf_counter() - preprocess_start:.4f}s"
    )

    # ===== 2. Generate timeline chart trajectory (only in streaming scenario) =====
    timeline_traces = []
    if has_timeline:
        logger.debug(f"Generating timeline traces for {n_requests} requests...")
        timeline_traces = generate_timeline_traces(
            adjusted_starts,
            adjusted_ends,
            adjusted_first_token_times,
            multiturn_group_id_list,
            unit,
        )

    # ===== 3. Generate concurrency chart trajectory =====
    logger.debug("Generating concurrency traces...")
    concurrency_traces = generate_concurrency_traces(
        adjusted_starts, adjusted_ends, unit
    )

    # ===== 4. Create chart =====
    logger.debug("Creating figure layout...")

    # Create layout configuration
    layout = create_plot_layout(max_time, unit, has_timeline)

    # Create chart object
    if has_timeline:
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.1, shared_xaxes=True)
        for trace in timeline_traces:
            fig.add_trace(trace, row=1, col=1)
        for trace in concurrency_traces:
            fig.add_trace(trace, row=2, col=1)
    else:
        fig = go.Figure()
        for trace in concurrency_traces:
            fig.add_trace(trace)

    # Apply layout configuration
    fig.update_layout(layout)

    # ===== 5. Output HTML =====

    fig.write_html(
        output_file,
        include_plotlyjs="cdn",
        config=WEBGL_CONFIG,
        auto_open=False,
        full_html=True,
    )

    logger.info(f"Request timeline and concurrency chart saved to {output_file}")
    return True
