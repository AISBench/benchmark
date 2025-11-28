import os
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Tuple, Union, Callable
from tabulate import tabulate

from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.utils.logging.exceptions import AISBenchValueError
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES

logger = AISLogger()

_NORMAL_INDICES_MASK: np.ndarray = np.array([], dtype=int)
_ADAPTIVE_WINDOW_SIZE: int = 0


# RPS-distribution-visualized functions
def plot_rps_distribution(
        cumulative_delays: np.ndarray,  # Changed to ndarray for performance
        timing_anomaly_indices: np.ndarray, # show in p1/p2/p3 (red)
        burstiness_anomaly_indices: np.ndarray, # show in p1 (yellow)
        request_rate: float,
        burstiness: float,
        ramp_up_strategy: Optional[str],
        ramp_up_start_rps: Optional[float],
        ramp_up_end_rps: Optional[float],
        output_path: str
) -> None:
    """
    Main function: RPS distribution analysis with three charts
    """
    # 1. Prepare time-RPS data
    time_points, rps_values = _prepare_time_rps_data(cumulative_delays)

    # 2. Separate points into three categories
    # Create masks for different anomaly types
    timing_mask = np.zeros(len(time_points), dtype=bool)
    if timing_anomaly_indices.size > 0:
        timing_mask[timing_anomaly_indices] = True
    burstiness_mask = np.zeros(len(time_points), dtype=bool)
    if burstiness_anomaly_indices.size > 0:
        burstiness_mask[burstiness_anomaly_indices] = True
    # Create all mask
    valid_mask = ~np.isinf(rps_values)
    invalid_mask = np.isinf(rps_values)
    total_invalid = np.sum(invalid_mask)
    # Apply masks
    normal_mask = valid_mask & ~timing_mask & ~burstiness_mask
    timing_anomaly_mask = valid_mask & timing_mask
    burstiness_anomaly_mask = valid_mask & burstiness_mask
    global _NORMAL_INDICES_MASK
    _NORMAL_INDICES_MASK = normal_mask

    # Extract points for each category
    normal_time_points = time_points[normal_mask]
    normal_rps_values = rps_values[normal_mask]
    timing_anomaly_time_points = time_points[timing_anomaly_mask]
    timing_anomaly_rps_values = rps_values[timing_anomaly_mask]
    burstiness_anomaly_time_points = time_points[burstiness_anomaly_mask]
    burstiness_anomaly_rps_values = rps_values[burstiness_anomaly_mask]

    # 3. Calculate bins for classic RPS distribution
    # For classic RPS distribution, burstiness anomalies are treated as normal
    combined_normal_rps = np.concatenate([normal_rps_values, burstiness_anomaly_rps_values])
    display_min, display_max, num_bins = _calculate_rps_bins(
        combined_normal_rps, ramp_up_start_rps, ramp_up_end_rps, request_rate
    )

    # 4. Calculate max y-value for normal values
    max_normal_y = _calculate_max_normal_y(combined_normal_rps, display_min, display_max, num_bins)
    anomaly_y = max_normal_y * 10 if max_normal_y > 0 else 1

    # 5. Prepare request interval data
    intervals = _prepare_interval_data(cumulative_delays)
    if timing_anomaly_indices.size > 0:
        normal_intervals, timing_anomaly_intervals = _separate_normal_anomaly_intervals(
            intervals, timing_anomaly_indices
        )
    else:
        # Handle case with no anomalies
        normal_intervals = intervals
        timing_anomaly_intervals = np.array([])

    # 6. Calculate bins for interval distribution
    display_min_interval, display_max_interval, num_bins_interval = _calculate_interval_bins(
        normal_intervals
    )
    # 7. Calculate max y-value for normal intervals
    max_normal_y_interval = _calculate_max_normal_y_interval(
        normal_intervals, display_min_interval, display_max_interval, num_bins_interval
    )
    anomaly_y_interval = max_normal_y_interval * 10 if max_normal_y_interval > 0 else 1
    # 8. Create combined title
    target_rate = ramp_up_end_rps if (
        ramp_up_strategy is not None and ramp_up_start_rps is not None and ramp_up_end_rps is not None
        ) else request_rate
    combined_title = _create_combined_title(target_rate, ramp_up_strategy, ramp_up_start_rps, ramp_up_end_rps)

    # 9. Create chart and add traces
    fig = _create_chart_figure(
        normal_time_points, normal_rps_values,
        timing_anomaly_time_points, timing_anomaly_rps_values,
        burstiness_anomaly_time_points, burstiness_anomaly_rps_values,
        combined_normal_rps,
        anomaly_y,
        normal_intervals,
        timing_anomaly_intervals,
        anomaly_y_interval,
        request_rate,
        burstiness,
        ramp_up_strategy, ramp_up_start_rps, ramp_up_end_rps,
        display_min, display_max, num_bins,
        display_min_interval, display_max_interval, num_bins_interval,
        time_points,
        combined_title
    )
    # 10. Save chart
    _export_to_html(fig, output_path)

    # 11. Log statistics
    _log_statistics(
        cumulative_delays,
        normal_rps_values,
        timing_anomaly_rps_values,
        burstiness_anomaly_rps_values,
        time_points,
        intervals,
        normal_intervals,
        timing_anomaly_intervals,
        target_rate,
        burstiness,
        total_invalid
    )


def _prepare_time_rps_data(cumulative_delays: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare time-RPS distribution data
    Uses vectorized operations for better performance
    """
    intervals = np.diff(cumulative_delays, prepend=0.0)
    rps_values = np.divide(1.0, intervals, where=intervals > 1e-6, out=np.full_like(intervals, np.inf))
    time_points = np.round(cumulative_delays, 3)
    rps_values = np.round(rps_values, 3)
    return time_points, rps_values


def _calculate_rps_bins(
    finite_normal_rps: np.ndarray,
    ramp_up_start_rps: Optional[float],
    ramp_up_end_rps: Optional[float],
    request_rate: float
) -> Tuple[float, float, int]:
    """
    Calculate bins for classic RPS distribution
    Uses efficient numpy operations for min/max calculations
    """
    EMPTY_ARRAY_RETURN = (0.0, 1.0, 1)
    DEFAULT_MIN_MULTIPLIER = 0.9
    DEFAULT_MAX_MULTIPLIER = 1.1
    RAMP_UP_END_MULTIPLIER = 1.5
    SINGLE_VALUE_ADJUSTMENT = 1.0
    STD_DEVIATION_THRESHOLD = 1e-6
    SCOTT_FACTOR = 3.5  # Scott's rule factor for bin width
    MIN_BIN_WIDTH = 1e-6
    MIN_BINS = 10
    MAX_BINS = 100

    if finite_normal_rps.size == 0:
        return EMPTY_ARRAY_RETURN

    data_min = np.min(finite_normal_rps)
    data_max = np.max(finite_normal_rps)

    ref_points = []
    if ramp_up_start_rps is not None:
        ref_points.append(ramp_up_start_rps)
    if ramp_up_end_rps is not None:
        ref_points.append(ramp_up_end_rps)
    if request_rate is not None:
        ref_points.append(request_rate)

    if ref_points:
        ref_min = min(ref_points)
        ref_max = max(ref_points)
        display_min = min(data_min, ref_min) * DEFAULT_MIN_MULTIPLIER
        display_max = max(data_max, ref_max) * DEFAULT_MAX_MULTIPLIER
    else:
        display_min = data_min * DEFAULT_MIN_MULTIPLIER
        display_max = data_max * DEFAULT_MAX_MULTIPLIER

    if ramp_up_end_rps is not None:
        display_max = min(display_max, ramp_up_end_rps * RAMP_UP_END_MULTIPLIER)

    if finite_normal_rps.size == 1 or np.isclose(data_min, data_max):
        display_min = max(0, data_min - SINGLE_VALUE_ADJUSTMENT)
        display_max = data_max + SINGLE_VALUE_ADJUSTMENT
        return float(display_min), float(display_max), 1

    std_dev = np.std(finite_normal_rps)

    if std_dev < STD_DEVIATION_THRESHOLD:
        data_range = data_max - data_min
        h = SCOTT_FACTOR * data_range / (finite_normal_rps.size ** (1/3))
    else:
        h = SCOTT_FACTOR * std_dev / (finite_normal_rps.size ** (1/3))

    bin_width = max(MIN_BIN_WIDTH, h)

    num_bins = int((display_max - display_min) / bin_width)
    num_bins = max(MIN_BINS, min(MAX_BINS, num_bins))

    return float(display_min), float(display_max), num_bins


def _calculate_max_normal_y(
    finite_normal_rps: np.ndarray,
    display_min: float,
    display_max: float,
    num_bins: int
) -> float:
    """
    Calculate max y-value for normal values
    Uses numpy histogram for efficient bin calculation
    """
    if finite_normal_rps.size == 0:
        return 1.0
    # Calculate histogram
    hist, _ = np.histogram(
        finite_normal_rps,
        bins=num_bins,
        range=(display_min, display_max)
    )
    return float(hist.max()) if hist.size > 0 else 1.0


def _calculate_interval_bins(
    normal_intervals: np.ndarray
) -> Tuple[float, float, int]:
    """
    Calculate bins for interval distribution
    Efficiently calculates bin parameters using numpy
    """
    EMPTY_ARRAY_RETURN = (0.0, 1.0, 1)
    MIN_DISPLAY_MULTIPLIER = 0.9
    MAX_DISPLAY_MULTIPLIER = 1.1
    SINGLE_VALUE_ADJUSTMENT = 0.001
    STD_DEVIATION_THRESHOLD = 1e-6
    SCOTT_FACTOR = 3.5  # Scott's rule factor for bin width
    MIN_BIN_WIDTH = 1e-6
    MIN_BINS = 10
    MAX_BINS = 100

    if normal_intervals.size == 0:
        return EMPTY_ARRAY_RETURN

    data_min = np.min(normal_intervals)
    data_max = np.max(normal_intervals)

    display_min_interval = max(0, data_min * MIN_DISPLAY_MULTIPLIER)
    display_max_interval = data_max * MAX_DISPLAY_MULTIPLIER

    if normal_intervals.size == 1 or np.isclose(data_min, data_max):
        display_min_interval = max(0, data_min - SINGLE_VALUE_ADJUSTMENT)
        display_max_interval = data_max + SINGLE_VALUE_ADJUSTMENT
        return float(display_min_interval), float(display_max_interval), 1

    std_dev = np.std(normal_intervals)

    if std_dev < STD_DEVIATION_THRESHOLD:
        data_range = data_max - data_min
        h = SCOTT_FACTOR * data_range / (normal_intervals.size ** (1/3))
    else:
        h = SCOTT_FACTOR * std_dev / (normal_intervals.size ** (1/3))

    bin_width = max(MIN_BIN_WIDTH, h)
    num_bins_interval = int((display_max_interval - display_min_interval) / bin_width)
    num_bins_interval = max(MIN_BINS, min(MAX_BINS, num_bins_interval))

    return float(display_min_interval), float(display_max_interval), num_bins_interval


def _calculate_max_normal_y_interval(
    normal_intervals: np.ndarray,
    display_min_interval: float,
    display_max_interval: float,
    num_bins_interval: int
) -> float:
    """
    Calculate max y-value for normal intervals
    Efficiently calculates max bin height using numpy histogram
    """
    if normal_intervals.size == 0:
        return 1.0

    hist, _ = np.histogram(
        normal_intervals,
        bins=num_bins_interval,
        range=(display_min_interval, display_max_interval)
    )
    return float(hist.max()) if hist.size > 0 else 1.0


def _calculate_theoretical_ramp(
    total_requests: int,
    ramp_up_strategy: Optional[str],
    ramp_up_start_rps: Optional[float],
    ramp_up_end_rps: Optional[float],
    request_rate: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Computing the theoretical climb path, correctly accounting for climb strategy."""

    request_indices = np.arange(total_requests)
    progress = request_indices / max(total_requests - 1, 1)

    if ramp_up_strategy == "linear" and ramp_up_start_rps is not None and ramp_up_end_rps is not None:
        theoretical_rates = ramp_up_start_rps + (ramp_up_end_rps - ramp_up_start_rps) * progress
    elif ramp_up_strategy == "exponential" and ramp_up_start_rps is not None and ramp_up_end_rps is not None:
        ratio = ramp_up_end_rps / ramp_up_start_rps
        theoretical_rates = ramp_up_start_rps * (ratio ** progress)
    else:
        theoretical_rates = np.full(total_requests, request_rate)

    theoretical_intervals = np.zeros(total_requests)
    non_zero_mask = theoretical_rates > 0
    theoretical_intervals[non_zero_mask] = 1.0 / theoretical_rates[non_zero_mask]

    cumulative_theoretical_times = np.cumsum(theoretical_intervals)
    return cumulative_theoretical_times, theoretical_rates


def _create_combined_title(
    target_rate: float,
    ramp_up_strategy: Optional[str],
    ramp_up_start_rps: Optional[float],
    ramp_up_end_rps: Optional[float]
) -> str:
    """Create combined title for the chart"""
    title = f"Request Per Second(RPS) Distribution Analysis | Target Rate: {target_rate:.2f}"
    if ramp_up_strategy:
        title += f" | {ramp_up_strategy.capitalize()} Ramp-up: {ramp_up_start_rps or 0:.1f}→{ramp_up_end_rps or 0:.1f}"
    return title


def _prepare_interval_data(cumulative_delays: np.ndarray) -> np.ndarray:
    """Prepare interval data """
    if cumulative_delays.size == 0:
        return np.array([])
    # Calculate intervals using vectorized diff
    return np.diff(cumulative_delays, prepend=0.0)


def _separate_normal_anomaly_intervals(
    intervals: np.ndarray,
    timing_anomaly_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Separate normal and anomaly intervals """
    if intervals.size == 0:
        return np.array([]), np.array([])
    # Create anomaly mask
    anomaly_mask = np.zeros(intervals.size, dtype=bool)
    if timing_anomaly_indices.size > 0:
        # Convert indices to mask for efficient indexing
        anomaly_mask[timing_anomaly_indices] = True
    return intervals[~anomaly_mask], intervals[anomaly_mask]


def _density_based_sampling(
    time_points: np.ndarray,
    values: np.ndarray,
    max_samples: int = 1000,
    num_strata: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified Sampling Method Based on Cumulative Distribution Function (CDF)
    Improving Efficiency While Preserving Temporal Density Distribution

    Parameters:
        time_points: Array of time points
        values: Corresponding values array
        max_samples: Maximum number of sampling points
        num_strata: Number of strata

    Returns:
        Sampled time points and values
    """
    n = len(time_points)
    if n <= max_samples:
        return time_points, values

    time_indices = np.argsort(time_points)
    sorted_times = time_points[time_indices]

    strata_bounds = np.quantile(sorted_times, np.linspace(0, 1, num_strata + 1))

    strata_idx = np.digitize(sorted_times, strata_bounds) - 1
    strata_idx = np.clip(strata_idx, 0, num_strata - 1)

    strata_counts = np.bincount(strata_idx, minlength=num_strata)
    sample_counts = (strata_counts * max_samples / n).astype(int)

    total_samples = sample_counts.sum()
    if total_samples < max_samples:
        extra = max_samples - total_samples
        extra_per_stratum = (strata_counts / strata_counts.sum() * extra).astype(int)
        sample_counts += extra_per_stratum
        remainder = max_samples - sample_counts.sum()
        if remainder > 0:
            largest_strata = np.argsort(strata_counts)[-remainder:]
            sample_counts[largest_strata] += 1
    elif total_samples > max_samples:
        reduction = max_samples / total_samples
        sample_counts = (sample_counts * reduction).astype(int)
        remainder = max_samples - sample_counts.sum()
        if remainder > 0:
            largest_strata = np.argsort(strata_counts)[-remainder:]
            sample_counts[largest_strata] += 1

    sampled_indices = []
    for i in range(num_strata):
        if sample_counts[i] == 0:
            continue

        stratum_mask = (strata_idx == i)
        stratum_global_indices = time_indices[stratum_mask]

        if len(stratum_global_indices) <= sample_counts[i]:
            sampled_indices.append(stratum_global_indices)
        else:
            selected = np.random.choice(
                stratum_global_indices,
                size=sample_counts[i],
                replace=False
            )
            sampled_indices.append(np.sort(selected))

    sampled_indices = np.concatenate(sampled_indices)
    sampled_indices.sort()

    return time_points[sampled_indices], values[sampled_indices]


def _calculate_adaptive_window(data_size):
    """Adjust window size dynamically based on data scale"""
    return next((
        window for threshold, window in sorted({
            1000: 20,
            10000: 50,
            100000: 100
        }.items())
        if data_size <= threshold
    ), 200)


def _exponential_moving_average(data, window_size, alpha=None):
    """Calculate exponential weighted moving average"""
    if alpha is None:
        alpha = 2 / (window_size + 1)

    weights = np.exp(np.linspace(-alpha, 0, window_size))
    weights /= weights.sum()

    return np.convolve(data, weights, mode='valid')


def _create_chart_figure(
    normal_time_points: np.ndarray,
    normal_rps_values: np.ndarray,
    timing_anomaly_time_points: np.ndarray,
    timing_anomaly_rps_values: np.ndarray,
    burstiness_anomaly_time_points: np.ndarray,
    burstiness_anomaly_rps_values: np.ndarray,
    finite_normal_rps: np.ndarray,
    anomaly_y: float,
    normal_intervals: np.ndarray,
    timing_anomaly_intervals: np.ndarray,
    anomaly_y_interval: float,
    request_rate: float,
    burstiness: float,
    ramp_up_strategy: Optional[str],
    ramp_up_start_rps: Optional[float],
    ramp_up_end_rps: Optional[float],
    display_min: float,
    display_max: float,
    num_bins: int,
    display_min_interval: float,
    display_max_interval: float,
    num_bins_interval: int,
    time_points: np.ndarray,
    combined_title: str
) -> go.Figure:
    """
    Create chart figure with traces
    Uses efficient Plotly methods and avoids unnecessary data copies
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Time vs RPS - Distribution',
            'Expected: RPS vs Request Count - Distribution',
            f'Expected: Gamma Distribution (burstiness: {burstiness})',
            'Legend Explanation'),
        specs=[
            [{"type": "scatter", "rowspan": 1}, {"type": "xy", "rowspan": 1}],
            [{"type": "xy", "rowspan": 1}, {"type": "table", "rowspan": 1}]
        ],
        row_heights=[0.5, 0.5],
        column_widths=[0.5, 0.5],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    legend_trace_name_prefix_list = [
        "Time - RPS: ",
        "RPS - Request Count: ",
        "Gamma Dist: ",
    ]

    traces_names_dict = dict()

    curr_chart_name = legend_trace_name_prefix_list[0]
    traces_names_dict[curr_chart_name]=_add_time_rps_traces(fig,
                                            normal_time_points, normal_rps_values,
                                            timing_anomaly_time_points, timing_anomaly_rps_values,
                                            burstiness_anomaly_time_points, burstiness_anomaly_rps_values,
                                            request_rate, ramp_up_strategy, ramp_up_start_rps, ramp_up_end_rps,
                                            curr_chart_name, row=1, col=1)

    curr_chart_name = legend_trace_name_prefix_list[1]
    traces_names_dict[curr_chart_name]=_add_classic_rps_traces(fig, finite_normal_rps, timing_anomaly_rps_values,
                                            display_min, display_max, num_bins,
                                            curr_chart_name, row=1, col=2)

    curr_chart_name = legend_trace_name_prefix_list[2]
    traces_names_dict[curr_chart_name]=_add_interval_traces(fig, normal_intervals, timing_anomaly_intervals,
                                            display_min_interval, display_max_interval, num_bins_interval,
                                            curr_chart_name, row=2, col=1)

    _add_legend_explanation_table(fig, traces_names_dict, row=2, col=2)

    fig.update_traces(
        legendgroup="time_rps",
        selector=lambda trace: trace.name is not None and trace.name.startswith(legend_trace_name_prefix_list[0])
    )
    fig.update_traces(
        legendgroup="rps_dist",
        selector=lambda trace: trace.name is not None and trace.name.startswith(legend_trace_name_prefix_list[1])
    )
    fig.update_traces(
        legendgroup="interval_dist",
        selector=lambda trace: trace.name is not None and trace.name.startswith(legend_trace_name_prefix_list[2])
    )
    fig.update_layout(
        title=combined_title,
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=12),
        title_font=dict(size=16),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1,
            font=dict(size=10),
            groupclick="toggleitem",
            tracegroupgap=30
        ),
        margin=dict(l=50, r=50, t=150, b=200),
        hovermode="closest",
        plot_bgcolor="rgba(240,240,240,0.9)",
        paper_bgcolor="rgba(255,255,255,1)",
        autosize=True
    )

    return fig


def _add_time_rps_traces(
    fig: go.Figure,
    normal_time_points: np.ndarray,
    normal_rps_values: np.ndarray,
    timing_anomaly_time_points: np.ndarray,
    timing_anomaly_rps_values: np.ndarray,
    burstiness_anomaly_time_points: np.ndarray,
    burstiness_anomaly_rps_values: np.ndarray,
    request_rate: float,
    ramp_up_strategy: Optional[str],
    ramp_up_start_rps: Optional[float],
    ramp_up_end_rps: Optional[float],
    legend_trace_name_prefix: Optional[str],
    row: int = 1,
    col: int = 1
) -> List[str]:
    """Add traces for time-RPS distribution with separate anomaly types"""
    scatter_class = go.Scattergl if len(normal_time_points) > 10000 else go.Scatter
    traces_names = []
    curr_trace_name = f'{legend_trace_name_prefix}Normal RPS'
    traces_names.append(curr_trace_name)

    fig.add_trace(
        scatter_class(
            x=normal_time_points,
            y=normal_rps_values,
            mode='lines+markers',
            marker=dict(size=3, color='rgba(31, 119, 180, 0.3)', opacity=0.3),
            line=dict(color='rgba(31, 119, 180, 0.3)'),
            name=curr_trace_name,
            showlegend=True
        ),
        row=row, col=col
    )

    if timing_anomaly_time_points.size > 0:
        sampled_time, sampled_rps = _density_based_sampling(
            timing_anomaly_time_points,
            timing_anomaly_rps_values,
            max_samples=1000
        )
        curr_trace_name = f'{legend_trace_name_prefix}Time Interval Caused Anomaly'
        traces_names.append(curr_trace_name)
        fig.add_trace(
            go.Scatter(
                x=sampled_time,
                y=sampled_rps,
                mode='markers',
                marker=dict(size=8, color='red', opacity=1.0, symbol='triangle-up'),
                name=curr_trace_name,
                showlegend=True,
                visible='legendonly'
            ),
            row=row, col=col
        )

    if burstiness_anomaly_time_points.size > 0:
        sampled_time, sampled_rps = _density_based_sampling(
            burstiness_anomaly_time_points,
            burstiness_anomaly_rps_values,
            max_samples=1000
        )
        curr_trace_name = f'{legend_trace_name_prefix}Burstiness Caused Anomaly'
        traces_names.append(curr_trace_name)
        fig.add_trace(
            go.Scatter(
                x=sampled_time,
                y=sampled_rps,
                mode='markers',
                marker=dict(size=8, color='yellow', opacity=1.0, symbol='square'),
                name=curr_trace_name,
                showlegend=True,
                visible='legendonly'
            ),
            row=row, col=col
        )

    if normal_rps_values.size > 0:
        finite_mask = np.isfinite(normal_rps_values)
        finite_rps = normal_rps_values[finite_mask]
        finite_time = normal_time_points[finite_mask]

        adaptive_window = _calculate_adaptive_window(len(finite_rps))

        if len(finite_rps) >= adaptive_window:
            global _ADAPTIVE_WINDOW_SIZE
            _ADAPTIVE_WINDOW_SIZE = adaptive_window
            moving_avg = _exponential_moving_average(finite_rps, adaptive_window)
            moving_avg_time = finite_time[adaptive_window-1:]

            curr_trace_name = f'{legend_trace_name_prefix}{adaptive_window}-point EWMA'
            traces_names.append(curr_trace_name)
            fig.add_trace(
                go.Scatter(
                    x=moving_avg_time,
                    y=moving_avg,
                    mode='lines',
                    line=dict(color='purple', width=2),
                    name=curr_trace_name,
                    showlegend=True,
                    visible='legendonly'
                ),
                row=row, col=col
            )

    if ramp_up_strategy in ("linear", "exponential") and ramp_up_start_rps is not None and ramp_up_end_rps is not None:
        cumulative_theoretical_times, theoretical_rates = _calculate_theoretical_ramp(
            total_requests=len(normal_time_points) + len(timing_anomaly_time_points) + len(burstiness_anomaly_time_points),
            ramp_up_strategy=ramp_up_strategy,
            ramp_up_start_rps=ramp_up_start_rps,
            ramp_up_end_rps=ramp_up_end_rps,
            request_rate=request_rate
        )
        valid_mask = np.isfinite(theoretical_rates) & np.isfinite(cumulative_theoretical_times)
        if np.any(valid_mask):
            curr_trace_name = f'{legend_trace_name_prefix}Theoretical Ramp-up'
            traces_names.append(curr_trace_name)
            fig.add_trace(
                go.Scatter(
                    x=cumulative_theoretical_times[valid_mask],
                    y=theoretical_rates[valid_mask],
                    mode='lines',
                    line=dict(color='green', width=2, dash='dash'),
                    name=curr_trace_name,
                    showlegend=True,
                    visible='legendonly'
                ),
                row=row, col=col
            )

    fig.update_xaxes(title_text="Time (seconds)", row=row, col=col)
    fig.update_yaxes(title_text="RPS", row=row, col=col)
    return traces_names


def _add_classic_rps_traces(
    fig: go.Figure,
    finite_normal_rps: np.ndarray,
    timing_anomaly_rps_values: np.ndarray,
    display_min: float,
    display_max: float,
    num_bins: int,
    legend_trace_name_prefix: Optional[str],
    row: int = 1,
    col: int = 2
) -> List[str]:
    """Add traces for classic RPS distribution (only timing anomalies)"""
    traces_names = []
    curr_trace_name = f'{legend_trace_name_prefix}Normal Request Count'
    traces_names.append(curr_trace_name)

    fig.add_trace(
        go.Histogram(
            x=finite_normal_rps,
            xbins=dict(start=display_min, end=display_max, size=(display_max - display_min) / num_bins),
            marker_color='#2ca02c',
            opacity=0.6,
            name=curr_trace_name,
            showlegend=True
        ),
        row=row, col=col
    )

    if timing_anomaly_rps_values.size > 0:
        sampled_rps, _ = _density_based_sampling(
            timing_anomaly_rps_values,
            timing_anomaly_rps_values,
            max_samples=1000
        )
        curr_trace_name = f'{legend_trace_name_prefix}Time Interval Caused Anomaly'
        traces_names.append(curr_trace_name)
        fig.add_trace(
            go.Scatter(
                x=sampled_rps,
                y=sampled_rps,
                mode='markers',
                marker=dict(size=8, color='red', symbol='triangle-up'),
                name=curr_trace_name,
                showlegend=True,
                visible='legendonly'
            ),
            row=row, col=col
        )

    fig.update_xaxes(title_text="RPS", row=row, col=col)
    fig.update_yaxes(title_text="Request Count", row=row, col=col)
    return traces_names


def _add_interval_traces(
    fig: go.Figure,
    normal_intervals: np.ndarray,
    timing_anomaly_intervals: np.ndarray,
    display_min_interval: float,
    display_max_interval: float,
    num_bins_interval: int,
    legend_trace_name_prefix: Optional[str],
    row: int = 2,
    col: int = 1
) -> List[str]:
    """Add traces for interval distribution (only timing anomalies)"""
    traces_names = []
    curr_trace_name = f'{legend_trace_name_prefix}Normal Intervals'
    traces_names.append(curr_trace_name)

    fig.add_trace(
        go.Histogram(
            x=normal_intervals,
            xbins=dict(
                start=display_min_interval,
                end=display_max_interval,
                size=(display_max_interval - display_min_interval) / num_bins_interval
            ),
            marker_color='#9467bd',
            opacity=0.6,
            name=curr_trace_name,
            showlegend=True
        ),
        row=row, col=col
    )

    if timing_anomaly_intervals.size > 0:
        sampled_intervals, _ = _density_based_sampling(
            timing_anomaly_intervals,
            timing_anomaly_intervals,
            max_samples=1000
        )
        curr_trace_name = f'{legend_trace_name_prefix}Time Interval Caused Anomaly'
        traces_names.append(curr_trace_name)
        fig.add_trace(
            go.Scatter(
                x=sampled_intervals,
                y=sampled_intervals,
                mode='markers',
                marker=dict(size=8, color='red', symbol='triangle-up'),
                name=curr_trace_name,
                showlegend=True,
                visible='legendonly'
            ),
            row=row, col=col
        )
    # Set axis labels
    fig.update_xaxes(title_text="Interval Time (seconds)", row=row, col=col)
    fig.update_yaxes(title_text="Request Count", row=row, col=col)
    return traces_names


def _add_legend_explanation_table(
    fig: go.Figure,
    traces_names_dict: dict,
    row: int = 2,
    col: int = 2
) -> None:
    """Add expanded legend explanation table to the chart with grouped display"""
    prefix_list = list(traces_names_dict.keys())
    global _ADAPTIVE_WINDOW_SIZE
    base_descriptions = {
        f"{prefix_list[0]}Normal RPS": ("期望请求率(排除异常值)", "实际间隔时间 ≥ 1ms<br>且与期望间隔的偏差 ≤ 50%", "期望间隔 = 1 / 当前请求率", "蓝色连线+点状标记"),
        f"{prefix_list[0]}Time Interval Caused Anomaly": ("请求间隔时间异常点", "系统无法可靠处理低于1ms的时间间隔", "实际间隔时间 < 1ms", "红色三角形标记<br>(密度采样最多1000个点)"),
        f"{prefix_list[0]}Burstiness Caused Anomaly": ("突发性异常点", "Gamma分布生成的间隔时间显著偏离期望值", "|实际间隔 - 期望间隔| / 期望间隔 > 50%", "黄色方形标记<br>(密度采样最多1000个点)"),
        f"{prefix_list[0]}{_ADAPTIVE_WINDOW_SIZE}-point EWMA": ("指数加权移动平均线", "EWMA_t = α * RPS_t + (1-α) * EWMA_{t-1}", "对近期RPS值赋予更高权重，平滑序列", "紫色实线"),
        f"{prefix_list[0]}Theoretical Ramp-up": ("理论爬升线", "线性: RPS = start + (end - start)*progress<br>指数: RPS = start * (ratio^progress)", "根据配置的爬升策略计算期望RPS值", "绿色虚线"),
        f"{prefix_list[1]}Normal Request Count": ("请求个数(排除异常值)", "实际间隔时间 ≥ 1ms<br>且与期望间隔的偏差 ≤ 50%", "落于该横坐标区间内的请求频数", "绿色直方图"),
        f"{prefix_list[1]}Time Interval Caused Anomaly": ("请求间隔时间异常点", "系统无法可靠处理低于1ms的时间间隔", "实际间隔时间 < 1ms", "红色三角形标记<br>(密度采样最多1000个点)"),
        f"{prefix_list[2]}Normal Intervals": ("请求间隔时间分布(排除异常值)", "自适应计算最优分桶范围", "统计正常间隔时间的分布频率", "紫色直方图"),
        f"{prefix_list[2]}Time Interval Caused Anomaly": ("请求间隔时间异常点", "仅包含时间间隔异常点", "实际间隔时间 < 1ms", "红色三角形标记<br>(密度采样最多1000个点)")
    }

    group_names = []
    trace_names = []
    meanings = []
    calculations = []
    criteria = []
    visualizations = []

    for group_prefix, trace_list in traces_names_dict.items():
        if not trace_list:
            continue
        group_names.append(group_prefix)
        trace_names.append("")
        meanings.append("")
        calculations.append("")
        criteria.append("")
        visualizations.append("")
        for trace_name in trace_list:
            base_name = trace_name.replace(group_prefix, "").strip()
            description = base_descriptions.get(trace_name, ("", "", "", ""))
            group_names.append("")
            trace_names.append(base_name)
            meanings.append(description[0])
            calculations.append(description[1])
            criteria.append(description[2])
            visualizations.append(description[3])

    total_rows = 0
    for group_prefix, trace_list in traces_names_dict.items():
        if trace_list:
            total_rows += len(trace_list) + 1  # +1 for group header

    base_row_height = 25
    max_table_height = 400
    row_height = min(base_row_height, max_table_height / max(total_rows, 1))
    base_font_size = 14
    font_size = max(10, base_font_size - max(0, total_rows - 10) // 2)

    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>组名</b>', '<b>图例项</b>', '<b>含义</b>', '<b>计算原理</b>', '<b>判断方式</b>', '<b>可视化表现</b>'],
                font=dict(size=font_size * 1.2, color='white'),
                fill_color='#4a5568',
                align='left',
                height=row_height * 1.5,
            ),
            cells=dict(
                values=[group_names, trace_names, meanings, calculations, criteria, visualizations],
                font=dict(size=font_size),
                align='left',
                fill_color='rgba(247, 250, 252, 0.9)',
                height=row_height,
                line=dict(color='rgba(0,0,0,0.1)', width=1)
            ),
            columnwidth=[100, 150, 120, 180, 180, 150]
        ),
        row=row, col=col
    )

    fig.update_xaxes(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        row=row, col=col,
        domain=[0, 1]
    )
    fig.update_yaxes(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        row=row, col=col,
        domain=[0, 1]
    )
    fig.update_layout(
        margin=dict(l=50, r=50, t=100, b=50),
        autosize=True
    )


def _log_statistics(
    cumulative_delays: np.ndarray,
    finite_normal_rps: np.ndarray,
    timing_anomaly_rps_values: np.ndarray,
    burstiness_anomaly_rps_values: np.ndarray,
    time_points: np.ndarray,
    intervals: np.ndarray,
    normal_intervals: np.ndarray,
    timing_anomaly_intervals: np.ndarray,
    target_rate: float,
    burstiness: float,
    total_invalid: int
) -> None:
    """Log essential statistical information in a compact tabular format"""
    total_normal = finite_normal_rps.size
    total_timing_anomaly = timing_anomaly_rps_values.size
    total_burstiness_anomaly = burstiness_anomaly_rps_values.size

    total_requests = cumulative_delays.size
    calculated_total = total_normal + total_timing_anomaly + total_burstiness_anomaly  + total_invalid
    if total_requests != calculated_total:
        logger.warning(f"Request count mismatch! Total: {total_requests}, "
                      f"Calculated: {calculated_total}. Adjusting counts.")
        scale_factor = total_requests / calculated_total
        total_normal = int(total_normal * scale_factor)
        total_timing_anomaly = int(total_timing_anomaly * scale_factor)
        total_burstiness_anomaly = int(total_burstiness_anomaly * scale_factor)
        total_invalid = total_requests - total_normal - total_timing_anomaly - total_burstiness_anomaly

    core_stats = [
        ("Total Requests", total_requests),
        ("Request Classification",
         f"Normal: {total_normal} | "
         f"Timing Anomaly: {total_timing_anomaly} | "
         f"Burstiness Anomaly: {total_burstiness_anomaly} | "
         f"Infinite RPS Anomaly: {total_invalid}"),
        ("Target Rate", f"{target_rate:.2f} RPS"),
        ("Burstiness", f"{burstiness:.3f}")
    ]

    rps_stats = []
    if finite_normal_rps.size > 0:
        rps_stats.extend([
            ("Normal RPS", f"{finite_normal_rps.mean():.2f} ± {finite_normal_rps.std():.2f}"),
            ("Normal RPS Range", f"{finite_normal_rps.min():.2f}-{finite_normal_rps.max():.2f}")
        ])

    interval_stats = [
        ("Interval Stats",
         f"Avg: {intervals.mean():.3f}s | "
         f"Min: {intervals.min():.3f}s | "
         f"Max: {intervals.max():.3f}s"),
        ("Interval Classification",
         f"Normal (Normal + Burstiness Anomaly): {normal_intervals.size} | "
         f"Anomaly (Timing Anomaly + Infinite RPS Anomaly): {timing_anomaly_intervals.size}")
    ]

    stats = core_stats + rps_stats + interval_stats
    table = tabulate(
        stats,
        headers=["Metric", "Value"],
        tablefmt="simple",
        stralign="left",
        numalign="right"
    )

    title = "Request Per Second (RPS) Distribution Summary".center(len(table.split('\n')[0]))
    logger.info(f"\n{title}\n{table}\n")


# post time: Time - RPS functions
def add_actual_rps_to_chart(
    base_chart: Union[str, go.Figure, dict],
    post_time_list: List[float],
    output_name: Optional[str] = None,
    trace_name: str = "Actual RPS: After Excluding Anomalies",
    color: str = '#FFA500',
    opacity: float = 0.5,
    mode: str = "lines+markers",
    marker_size: int = 4,
    line_width: int = 1
) -> None:
    """
    Add actual request posting times to an existing RPS distribution chart

    Parameters:
        base_chart: Existing chart (json file path, Figure object, or dict)
        post_time_list: List of actual request posting times
        output_name: Optional output html file name (without path)
        trace_name: Name for the actual RPS trace
        color: Trace color
        opacity: Trace opacity
        mode: Plot mode
        marker_size: Marker size
        line_width: Line width
    """
    # construct a valid output path
    output_path = _determine_output_path(base_chart, output_name)
    if not output_path:
        logger.warning(f"Invalid output path or output path not provided: \n"
                       f"got base chart: {base_chart}\n"
                       f"got output name: {output_name}")
        logger.warning("Chart will not be saved.")
        return

    # Create actual RPS trace
    actual_rps_trace = _create_time_rps_trace(
        post_time_list,
        trace_name=trace_name,
        color=color,
        opacity=opacity,
        mode=mode,
        marker_size=marker_size,
        line_width=line_width
    )

    def update_legend_callback(fig: go.Figure, src_dict: dict):
        description = (
            "实际请求率(排除异常值)",
            "基于实际请求发送时间计算",
            "采用期望请求率下的索引，<br>排除了期望请求率计算时的两种异常点索引",
            "橙色实线 + 点状标记"
        )
        _update_legend_explanation(fig, trace_name, description)

    def define_layout_rollback(fig: go.Figure):
        fig.update_xaxes(title_text="Time (seconds)", showgrid=True, gridwidth=1,
                         gridcolor='rgba(200, 200, 200, 0.5)')
        fig.update_yaxes(title_text="RPS", showgrid=True, gridwidth=1,
                         gridcolor='rgba(200, 200, 200, 0.5)')

        fig.update_layout(
            title="Time - Actual RPS distribution",
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=16),
            title_font=dict(size=24),
            margin=dict(l=50, r=50, t=150, b=200),
            hovermode="closest",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            autosize=True
        )

    # Add trace to the base chart at position (1,1)
    merged_fig = _merge_into_subplot(
        dst=base_chart,
        src=actual_rps_trace,
        row=1,
        col=1,
        visible=True,
        callback=update_legend_callback,
        rollback=define_layout_rollback,
    )

    # Export the updated chart
    if merged_fig is not None:
        _export_to_html(merged_fig, output_path, save_json=False)
        logger.info(f"Updated chart with actual RPS saved to {output_path}")


def _is_valid_chart_html_file(file_path: Optional[str]) -> bool:
    if not file_path:
        return False
    if not file_path.lower().endswith('.html'):
        return False
    if not os.path.exists(file_path):
        return False
    return True


def _determine_output_path(
    base_chart: Union[str, go.Figure, dict],
    output_name: Optional[str]
) -> str:
    """
    Intelligently determine the output path for saving chart files

    Rules:
    1. If output_name is provided:
        - Automatically appends '.html' if output_name lacks it
        - If base_chart is a valid HTML file path:
            Output to base_chart's directory with output_name
        - Else:
            Output to base_chart if it's a directory, else to base_chart's parent directory (if exists), else current directory

    2. If output_name is not provided:
        - If base_chart is a valid HTML file path:
            Output to base_chart's directory with "[original_basename]_with_actual_rps.html"
        - Else:
            Output to base_chart if it's a directory, else to base_chart's parent directory (if exists), else current directory
            with "rps_distribution_plot_with_actual_rps.html"
    """
    if not output_name:
        if _is_valid_chart_html_file(base_chart):

            dir_name = os.path.dirname(base_chart)
            base_name = os.path.basename(base_chart)
            base_name, _ = os.path.splitext(base_name) # _is_valid_chart_html_file makes sure base_name must .endswith('.html')

            new_name = f"{base_name}_with_actual_rps.html"
            return os.path.join(dir_name, new_name)
        else:
            if os.path.isdir(base_chart):
                base_dir = base_chart
            else:
                base_dir = os.path.dirname(base_chart)
                base_dir = base_dir if os.path.isdir(base_dir) else os.getcwd()
            return os.path.join(base_dir, "rps_distribution_plot_with_actual_rps.html")

    if not output_name.lower().endswith('.html'):
        output_name += '.html'

    if _is_valid_chart_html_file(base_chart):
        dir_name = os.path.dirname(base_chart)
        return os.path.join(dir_name, output_name)
    else:
        if os.path.isdir(base_chart):
            base_dir = base_chart
        else:
            base_dir = os.path.dirname(base_chart)
            base_dir = base_dir if os.path.isdir(base_dir) else os.getcwd()
        return os.path.join(base_dir, output_name)


def _create_time_rps_trace(
    post_time_list: List[float],
    trace_name: str = "Actual RPS",
    color: str = '#FFA500',
    opacity: float = 0.5,
    mode: str = "lines+markers",
    marker_size: int = 4,
    line_width: int = 1
) -> go.Scatter:
    """
    Create a Time-RPS trace from actual request posting times

    Parameters:
        post_time_list: List of actual request posting times (global delays)
        trace_name: Name for the trace (used in legend)
        color: Trace color
        opacity: Trace opacity
        mode: Plot mode ('lines', 'markers', or 'lines+markers')
        marker_size: Size of markers
        line_width: Width of lines

    Returns:
        go.Scatter trace object
    """
    # 1. Prepare time-RPS data
    time_points, rps_values = _prepare_actual_rps_data(post_time_list)

    # 2. Apply density-based sampling for large datasets
    if len(time_points) > 5000:
        time_points, rps_values = _density_based_sampling(
            time_points, rps_values, max_samples=5000
        )

    # 3. Create and return trace
    return go.Scatter(
        x=time_points,
        y=rps_values,
        mode=mode,
        name=trace_name,
        marker=dict(
            size=marker_size,
            color=color,
            opacity=opacity
        ),
        line=dict(
            width=line_width,
            color=color
        ),
        opacity=opacity
    )


def _prepare_actual_rps_data(post_time_list: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare time-RPS data from actual request posting times

    Parameters:
        post_time_list: List of actual request posting times

    Returns:
        Tuple of (time_points, rps_values)
    """
    post_times = np.array(post_time_list, dtype=float)
    sorted_indices = np.argsort(post_times)
    sorted_times = post_times[sorted_indices]

    intervals = np.diff(sorted_times, prepend=0.0)

    rps_values = np.divide(
        1.0,
        intervals,
        where=intervals > 1e-6,
        out=np.full_like(intervals, np.inf)
    )

    global _NORMAL_INDICES_MASK
    if _NORMAL_INDICES_MASK is not None and _NORMAL_INDICES_MASK.size != 0:
        max_size = min(_NORMAL_INDICES_MASK.size, len(sorted_times))
        time_points = np.round(sorted_times, 3)[_NORMAL_INDICES_MASK[:max_size]]
        rps_values = np.round(rps_values, 3)[_NORMAL_INDICES_MASK[:max_size]]

    else:
        time_points = np.round(sorted_times, 3)
        rps_values = np.round(rps_values, 3)

    return time_points, rps_values


def _update_legend_explanation(
    fig: go.Figure,
    trace_name: str,
    description: Tuple[str, str, str, str],
    group_prefix: str = "Time - RPS: "
) -> None:
    """
    Update the legend description table by adding new trace information
    to the specified group
    Parameters:
        fig: Figure object
        trace_name: Name of the new trace
        description: Description tuple (meaning, calculation principle, interpretation method, visual representation)
        group_prefix: Group name prefix
    """
    table_trace = None
    for trace in fig.data:
        if isinstance(trace, go.Table) and trace.cells.values[0] is not None:
            table_trace = trace
            break

    if table_trace is None:
        logger.warning("Legend explanation table not found")
        return

    group_names = list(table_trace.cells.values[0])
    trace_names = list(table_trace.cells.values[1])
    meanings = list(table_trace.cells.values[2])
    calculations = list(table_trace.cells.values[3])
    criteria = list(table_trace.cells.values[4])
    visualizations = list(table_trace.cells.values[5])

    insert_index = -1
    for i in range(len(group_names)):
        if group_names[i] == group_prefix:
            j = i + 1
            while j < len(group_names) and group_names[j] == "":
                j += 1
            insert_index = j
            break

    if insert_index == -1:
        logger.info(f"Creating new group: {group_prefix}")
        group_names.append(group_prefix)
        trace_names.append("")
        meanings.append("")
        calculations.append("")
        criteria.append("")
        visualizations.append("")
        insert_index = len(group_names)

    group_names.insert(insert_index, "")
    trace_names.insert(insert_index, trace_name)
    meanings.insert(insert_index, description[0])
    calculations.insert(insert_index, description[1])
    criteria.insert(insert_index, description[2])
    visualizations.insert(insert_index, description[3])

    table_trace.cells.values = [
        group_names,
        trace_names,
        meanings,
        calculations,
        criteria,
        visualizations
    ]

    total_rows = len(group_names)
    base_row_height = 25
    max_table_height = 400
    row_height = min(base_row_height, max_table_height / max(total_rows, 1))
    base_font_size = 14
    font_size = max(10, base_font_size - max(0, total_rows - 10) // 2)

    table_trace.header.height = row_height * 1.5
    table_trace.header.font.size = font_size * 1.2
    table_trace.cells.height = row_height
    table_trace.cells.font.size = font_size

    logger.debug(f"Added '{trace_name}' to group '{group_prefix}' in legend explanation table")


# public functions

def _export_to_html(fig: go.Figure, output_path: str, save_json: bool = True) -> None:
    """Save chart to HTML file """
    try:
        if fig is not None:
            fig.write_html(
                output_path,
                config={
                    'scrollZoom': True,
                    'plotGlPixelRatio': 1,
                    'showLink': False,
                    'displaylogo': False,
                    'responsive': True
                }
            )
            logger.info(f"Successfully saved RPS distribution HTML chart to {output_path}")

            if save_json:
                filename: str = output_path.replace('.html', '.json')
                _export_to_json(fig, filename)
        else:
            logger.warning("No figure to save.")
    except Exception as e:
        logger.warning(f"Unexpected error when saving HTML file: {e}")


def _convert_numpy_to_list(obj):
    """Recursively convert numpy arrays and scalars to Python native types.
    
    Args:
        obj: Object that may contain numpy arrays
        
    Returns:
        Object with all numpy arrays converted to lists
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_to_list(item) for item in obj]
    else:
        return obj


def _export_to_json(fig: go.Figure, filename: str) -> None:
    """Serialize figure representation to JSON file"""
    try:
        fig_dict = fig.to_dict()
        fig_dict = _convert_numpy_to_list(fig_dict)
        with open(filename, 'w') as f:
            json.dump(fig_dict, f, indent=2)
        logger.info(f"Successfully saved RPS distribution JSON data to {filename}")
    except Exception as e:
        logger.warning(f"Unexpected error when saving JSON file: {e}")


def _merge_into_subplot(
    dst: Union[str, go.Figure, dict],
    src: Union[str, go.Figure, dict, go.Trace, List[go.Trace]],
    row: int,
    col: int,
    trace_names: Optional[List[str]] = None,
    visible: str = "legendonly",
    merge_layout: bool = True,
    legendgroup: Optional[str] = None,
    callback: Optional[Callable[..., None]] = None,
    rollback: Optional[Callable[..., None]] = None,
) -> go.Figure:
    """
    Merge source chart (src) into destination chart (dst) at specified subplot position
    Supports src as traces or trace lists

    Parameters:
        dst: Destination chart (JSON file path, go.Figure object, or dict)
        src: Source chart (JSON file path, go.Figure object, dict, trace, or trace list)
        row: Target subplot row position
        col: Target subplot column position
        trace_names: Optional new trace names list
        visible: Trace visibility ('legendonly' or True)
        merge_layout: Whether to merge layout settings
        legendgroup: Legend group name to assign to all source traces
        callback: Optional callback function to perform custom operations after merging
        rollback: Optional rollback function to perform custom operations if loading dst failed

    Returns:
        Merged go.Figure object
    """
    merged_fig = None

    # Helper function to load chart data
    def convert_path_to_json_path(file_path: str) -> str:
        json_path = ""
        if file_path.lower().endswith('.html'):
            json_path = os.path.splitext(file_path)[0] + '.json'
        elif file_path.lower().endswith('.json'):
            json_path = file_path
        if os.path.exists(json_path):
            return json_path
        return ""

    def load_chart_data(chart) -> dict:
        if isinstance(chart, str):
            chart_trans = convert_path_to_json_path(chart)
            if not chart_trans:
                raise AISBenchValueError(UTILS_CODES.CHART_FILE_NOT_FOUND, f"Chart json file is not found in the same dir: {chart}")
            with open(chart_trans, 'r') as f:
                return json.load(f)
        elif isinstance(chart, go.Figure):
            return chart.to_dict()
        elif isinstance(chart, dict):
            return chart
        elif hasattr(chart, 'to_plotly_json') and callable(chart.to_plotly_json):
            return {'data': [chart.to_plotly_json()], 'layout': {}}
        elif isinstance(chart, list) and all(hasattr(t, 'to_plotly_json') for t in chart):
            return {'data': [t.to_plotly_json() for t in chart], 'layout': {}}
        else:
            raise AISBenchValueError(UTILS_CODES.INVALID_TYPE, f"Unsupported chart type: {type(chart)}")

    try:
        dst_dict = load_chart_data(dst)
    except Exception as e:
        logger.warning(f"Destination chart loading failed: {str(e)}.")
        dst_dict = None

    try:
        src_dict = load_chart_data(src)
        if dst_dict is None:
            try:
                logger.warning("Falling back to source chart only due to destination chart loading failure.")
                merged_fig = go.Figure(src_dict)
                if rollback is not None:
                    try:
                        rollback(merged_fig)
                    except Exception as e:
                        logger.warning(f"Callback function failed: {str(e)}")
                return merged_fig

            except Exception as e_fig:
                logger.warning(f"Failed to create Figure from source: {str(e_fig)}")
                logger.info("No information will be updated in any chart.")
                return

    except Exception as e:
        logger.warning(f"Source chart loading failed: {str(e)}")
        logger.info("No information will be updated in any chart.")
        return

    if 'data' not in dst_dict or 'layout' not in dst_dict:
        logger.warning("Invalid destination chart format")
        logger.info("No information will be updated in any chart.")
        return

    if 'data' not in src_dict:
        src_dict = {'data': src_dict, 'layout': {}}

    dst_layout = dst_dict.get('layout', {})

    total_cols = 1
    if 'grid' in dst_layout and 'columns' in dst_layout['grid']:
        total_cols = dst_layout['grid']['columns']
    elif 'xaxis' in dst_layout:
        axis_keys = [k for k in dst_layout.keys() if k.startswith('xaxis')]
        if axis_keys:
            total_cols = len(axis_keys)

    subplot_index = (row - 1) * total_cols + col

    if subplot_index == 1:
        xaxis_ref = 'x'
        yaxis_ref = 'y'
    else:
        xaxis_ref = f'x{subplot_index}'
        yaxis_ref = f'y{subplot_index}'

    target_legendgroup = legendgroup
    if target_legendgroup is None:
        for trace in dst_dict['data']:
            if trace.get('xaxis') == xaxis_ref and trace.get('yaxis') == yaxis_ref:
                if 'legendgroup' in trace:
                    target_legendgroup = trace['legendgroup']
                    break

        if target_legendgroup is None:
            target_legendgroup = f"group_{row}_{col}"


    for i, trace in enumerate(src_dict['data']):
        trace_copy = trace.copy()

        if trace_names and i < len(trace_names):
            trace_copy['name'] = trace_names[i]

        trace_copy['visible'] = visible

        trace_copy['xaxis'] = xaxis_ref
        trace_copy['yaxis'] = yaxis_ref

        trace_copy['legendgroup'] = target_legendgroup
        trace_copy['showlegend'] = True

        dst_dict['data'].append(trace_copy)

    if merge_layout and 'layout' in src_dict:
        for axis_type in ['xaxis', 'yaxis']:
            axis_ref = xaxis_ref if axis_type == 'xaxis' else yaxis_ref
            if axis_ref in src_dict['layout']:
                if axis_ref not in dst_dict['layout']:
                    dst_dict['layout'][axis_ref] = {}

                for key, value in src_dict['layout'][axis_ref].items():
                    if key not in dst_dict['layout'][axis_ref]:
                        dst_dict['layout'][axis_ref][key] = value

        if 'title' in src_dict['layout']:
            if 'annotations' not in dst_dict['layout']:
                dst_dict['layout']['annotations'] = []

            dst_dict['layout']['annotations'].append({
                'text': src_dict['layout']['title'].get('text', ''),
                'xref': f"{xaxis_ref} domain",
                'yref': f"{yaxis_ref} domain",
                'x': 0.5,
                'y': 1.1,
                'showarrow': False,
                'font': {'size': 12}
            })

    try:
        merged_fig = go.Figure(dst_dict)
    except Exception as e:
        logger.warning(f"Failed to create Figure object: {str(e)}")
        logger.info("No information will be updated in any chart.")
        return

    if callback is not None:
        try:
            callback(merged_fig, src_dict)
        except Exception as e:
            logger.warning(f"Callback function failed: {str(e)}")

    return merged_fig

