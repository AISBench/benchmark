import unittest
import os
import tempfile
from unittest.mock import MagicMock, patch, mock_open
import numpy as np

from ais_bench.benchmark.utils.visualization import rps_distribution_plot

class TestRPSDistributionPlot(unittest.TestCase):
    def setUp(self):
        self.cumulative_delays = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.timing_anomaly_indices = np.array([1])
        self.burstiness_anomaly_indices = np.array([2])
        self.request_rate = 10.0
        self.burstiness = 1.0
        self.output_path = "test_output.html"
        # Track files that might be created during tests
        self.files_to_cleanup = []

    def tearDown(self):
        """Clean up any files created during tests."""
        for filepath in self.files_to_cleanup:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError:
                    pass  # Ignore errors when removing files

        # Also clean up common test files that might be created
        common_test_files = [
            "test_output.html",
            "test_output.json",
            "test.html",
            "test.json",
            "base.html",
            "base.json"
        ]
        for filepath in common_test_files:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError:
                    pass  # Ignore errors when removing files

    @patch('ais_bench.benchmark.utils.visualization.rps_distribution_plot._export_to_html')
    @patch('ais_bench.benchmark.utils.visualization.rps_distribution_plot._create_chart_figure')
    def test_plot_rps_distribution(self, mock_create_figure, mock_export):
        mock_fig = MagicMock()
        mock_create_figure.return_value = mock_fig

        rps_distribution_plot.plot_rps_distribution(
            self.cumulative_delays,
            self.timing_anomaly_indices,
            self.burstiness_anomaly_indices,
            self.request_rate,
            self.burstiness,
            None, None, None, # ramp up args
            self.output_path
        )

        mock_create_figure.assert_called_once()
        mock_export.assert_called_once_with(mock_fig, self.output_path)

    @patch('ais_bench.benchmark.utils.visualization.rps_distribution_plot._merge_into_subplot')
    @patch('ais_bench.benchmark.utils.visualization.rps_distribution_plot._export_to_html')
    @patch('ais_bench.benchmark.utils.visualization.rps_distribution_plot._determine_output_path')
    def test_add_actual_rps_to_chart(self, mock_determine_path, mock_export, mock_merge):
        mock_determine_path.return_value = self.output_path
        mock_merge.return_value = MagicMock()

        post_time_list = [0.1, 0.2, 0.3]
        base_chart = "base.html"

        rps_distribution_plot.add_actual_rps_to_chart(
            base_chart,
            post_time_list
        )

        mock_merge.assert_called_once()
        mock_export.assert_called_once()

    @patch('ais_bench.benchmark.utils.visualization.rps_distribution_plot._export_to_json')
    def test_export_to_html(self, mock_export_json):
        mock_fig = MagicMock()
        rps_distribution_plot._export_to_html(mock_fig, self.output_path)

        mock_fig.write_html.assert_called_once_with(
            self.output_path,
            config=unittest.mock.ANY
        )
        mock_export_json.assert_called_once()
        # Track files for cleanup in case mock fails
        self.files_to_cleanup.append(self.output_path)
        self.files_to_cleanup.append(self.output_path.replace('.html', '.json'))

    @patch('json.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_export_to_json(self, mock_file, mock_json_dump):
        mock_fig = MagicMock()
        mock_fig.to_dict.return_value = {"data": []}
        filename = "test.json"

        rps_distribution_plot._export_to_json(mock_fig, filename)

        mock_file.assert_called_once_with(filename, 'w')
        mock_json_dump.assert_called_once()
        # Track file for cleanup in case mock fails
        self.files_to_cleanup.append(filename)

    def test_calculate_rps_bins(self):
        finite_normal_rps = np.array([10, 11, 12, 10, 11])
        ramp_up_start_rps = 5.0
        ramp_up_end_rps = 15.0
        request_rate = 10.0

        display_min, display_max, num_bins = rps_distribution_plot._calculate_rps_bins(
            finite_normal_rps,
            ramp_up_start_rps,
            ramp_up_end_rps,
            request_rate
        )

        self.assertIsInstance(display_min, float)
        self.assertIsInstance(display_max, float)
        self.assertIsInstance(num_bins, int)
        self.assertLess(display_min, display_max)
        self.assertGreater(num_bins, 0)

    def test_calculate_theoretical_ramp(self):
        total_requests = 10
        ramp_up_strategy = "linear"
        ramp_up_start_rps = 1.0
        ramp_up_end_rps = 10.0
        request_rate = 5.0

        cumulative_times, theoretical_rates = rps_distribution_plot._calculate_theoretical_ramp(
            total_requests,
            ramp_up_strategy,
            ramp_up_start_rps,
            ramp_up_end_rps,
            request_rate
        )

        self.assertEqual(len(cumulative_times), total_requests)
        self.assertEqual(len(theoretical_rates), total_requests)
        self.assertEqual(theoretical_rates[0], ramp_up_start_rps)
        self.assertEqual(theoretical_rates[-1], ramp_up_end_rps)

    def test_calculate_theoretical_ramp_exponential(self):
        """Test _calculate_theoretical_ramp with exponential strategy."""
        total_requests = 10
        ramp_up_strategy = "exponential"
        ramp_up_start_rps = 1.0
        ramp_up_end_rps = 10.0
        request_rate = 5.0

        cumulative_times, theoretical_rates = rps_distribution_plot._calculate_theoretical_ramp(
            total_requests,
            ramp_up_strategy,
            ramp_up_start_rps,
            ramp_up_end_rps,
            request_rate
        )

        self.assertEqual(len(cumulative_times), total_requests)
        self.assertEqual(len(theoretical_rates), total_requests)

    def test_prepare_time_rps_data(self):
        """Test _prepare_time_rps_data function."""
        cumulative_delays = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        time_points, rps_values = rps_distribution_plot._prepare_time_rps_data(cumulative_delays)

        self.assertEqual(len(time_points), len(cumulative_delays))
        self.assertEqual(len(rps_values), len(cumulative_delays))

    def test_calculate_max_normal_y(self):
        """Test _calculate_max_normal_y function."""
        finite_normal_rps = np.array([10, 11, 12, 10, 11])
        display_min = 5.0
        display_max = 15.0
        num_bins = 10

        max_y = rps_distribution_plot._calculate_max_normal_y(
            finite_normal_rps, display_min, display_max, num_bins
        )

        self.assertIsInstance(max_y, (int, float))
        self.assertGreaterEqual(max_y, 0)

    def test_prepare_interval_data(self):
        """Test _prepare_interval_data function."""
        cumulative_delays = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        intervals = rps_distribution_plot._prepare_interval_data(cumulative_delays)

        # _prepare_interval_data uses np.diff with prepend=0.0
        # This returns an array with the same length as input
        self.assertIsInstance(intervals, np.ndarray)
        # The function returns intervals which should have same length as cumulative_delays
        # because it uses prepend=0.0 in np.diff
        self.assertEqual(len(intervals), len(cumulative_delays))
        # First interval should be 0.1 (0.1 - 0.0)
        self.assertAlmostEqual(intervals[0], 0.1)

    def test_separate_normal_anomaly_intervals(self):
        """Test _separate_normal_anomaly_intervals function."""
        intervals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        timing_anomaly_indices = np.array([1, 3])

        normal_intervals, anomaly_intervals = rps_distribution_plot._separate_normal_anomaly_intervals(
            intervals, timing_anomaly_indices
        )

        self.assertIsInstance(normal_intervals, np.ndarray)
        self.assertIsInstance(anomaly_intervals, np.ndarray)

    def test_calculate_interval_bins(self):
        """Test _calculate_interval_bins function."""
        normal_intervals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        display_min, display_max, num_bins = rps_distribution_plot._calculate_interval_bins(
            normal_intervals
        )

        self.assertIsInstance(display_min, float)
        self.assertIsInstance(display_max, float)
        self.assertIsInstance(num_bins, int)
        self.assertLess(display_min, display_max)

    def test_calculate_max_normal_y_interval(self):
        """Test _calculate_max_normal_y_interval function."""
        normal_intervals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        display_min = 0.0
        display_max = 1.0
        num_bins = 10

        max_y = rps_distribution_plot._calculate_max_normal_y_interval(
            normal_intervals, display_min, display_max, num_bins
        )

        self.assertIsInstance(max_y, (int, float))
        self.assertGreaterEqual(max_y, 0)

    def test_create_combined_title(self):
        """Test _create_combined_title function."""
        target_rate = 10.0
        ramp_up_strategy = "linear"
        ramp_up_start_rps = 1.0
        ramp_up_end_rps = 10.0

        title = rps_distribution_plot._create_combined_title(
            target_rate, ramp_up_strategy, ramp_up_start_rps, ramp_up_end_rps
        )

        self.assertIsInstance(title, str)
        self.assertIn(str(target_rate), title)

    def test_create_combined_title_no_ramp(self):
        """Test _create_combined_title without ramp up."""
        target_rate = 10.0

        title = rps_distribution_plot._create_combined_title(
            target_rate, None, None, None
        )

        self.assertIsInstance(title, str)

    @patch('ais_bench.benchmark.utils.visualization.rps_distribution_plot._create_chart_figure')
    def test_plot_rps_distribution_with_ramp_up(self, mock_create_figure):
        """Test plot_rps_distribution with ramp up parameters."""
        mock_fig = MagicMock()
        mock_create_figure.return_value = mock_fig

        rps_distribution_plot.plot_rps_distribution(
            self.cumulative_delays,
            self.timing_anomaly_indices,
            self.burstiness_anomaly_indices,
            self.request_rate,
            self.burstiness,
            "linear",  # ramp_up_strategy
            1.0,  # ramp_up_start_rps
            10.0,  # ramp_up_end_rps
            self.output_path
        )

        mock_create_figure.assert_called_once()

    def test_calculate_rps_bins_with_ramp_up(self):
        """Test _calculate_rps_bins with ramp up parameters."""
        finite_normal_rps = np.array([10, 11, 12, 10, 11])
        ramp_up_start_rps = 5.0
        ramp_up_end_rps = 15.0
        request_rate = 10.0

        display_min, display_max, num_bins = rps_distribution_plot._calculate_rps_bins(
            finite_normal_rps,
            ramp_up_start_rps,
            ramp_up_end_rps,
            request_rate
        )

        self.assertIsInstance(display_min, float)
        self.assertIsInstance(display_max, float)
        self.assertIsInstance(num_bins, int)

    def test_calculate_rps_bins_no_ramp_up(self):
        """Test _calculate_rps_bins without ramp up parameters."""
        finite_normal_rps = np.array([10, 11, 12, 10, 11])
        request_rate = 10.0

        display_min, display_max, num_bins = rps_distribution_plot._calculate_rps_bins(
            finite_normal_rps,
            None,
            None,
            request_rate
        )

        self.assertIsInstance(display_min, float)
        self.assertIsInstance(display_max, float)
        self.assertIsInstance(num_bins, int)

    def test_calculate_rps_bins_empty_array(self):
        """Test _calculate_rps_bins with empty array."""
        finite_normal_rps = np.array([])
        display_min, display_max, num_bins = rps_distribution_plot._calculate_rps_bins(
            finite_normal_rps, None, None, 10.0
        )
        self.assertEqual(display_min, 0.0)
        self.assertEqual(display_max, 1.0)
        self.assertEqual(num_bins, 1)

    def test_calculate_rps_bins_single_value(self):
        """Test _calculate_rps_bins with single value."""
        finite_normal_rps = np.array([10.0])
        display_min, display_max, num_bins = rps_distribution_plot._calculate_rps_bins(
            finite_normal_rps, None, None, 10.0
        )
        self.assertLess(display_min, display_max)
        self.assertEqual(num_bins, 1)

    def test_calculate_rps_bins_same_values(self):
        """Test _calculate_rps_bins with same values."""
        finite_normal_rps = np.array([10.0, 10.0, 10.0])
        display_min, display_max, num_bins = rps_distribution_plot._calculate_rps_bins(
            finite_normal_rps, None, None, 10.0
        )
        self.assertLess(display_min, display_max)
        self.assertEqual(num_bins, 1)

    def test_calculate_rps_bins_low_std_dev(self):
        """Test _calculate_rps_bins with low standard deviation."""
        finite_normal_rps = np.array([10.0, 10.000001, 10.000002])
        display_min, display_max, num_bins = rps_distribution_plot._calculate_rps_bins(
            finite_normal_rps, None, None, 10.0
        )
        self.assertIsInstance(display_min, float)
        self.assertIsInstance(display_max, float)
        self.assertIsInstance(num_bins, int)

    def test_calculate_max_normal_y_empty(self):
        """Test _calculate_max_normal_y with empty array."""
        finite_normal_rps = np.array([])
        max_y = rps_distribution_plot._calculate_max_normal_y(
            finite_normal_rps, 0.0, 1.0, 10
        )
        self.assertEqual(max_y, 1.0)

    def test_calculate_interval_bins_empty(self):
        """Test _calculate_interval_bins with empty array."""
        normal_intervals = np.array([])
        display_min, display_max, num_bins = rps_distribution_plot._calculate_interval_bins(
            normal_intervals
        )
        self.assertEqual(display_min, 0.0)
        self.assertEqual(display_max, 1.0)
        self.assertEqual(num_bins, 1)

    def test_calculate_interval_bins_single_value(self):
        """Test _calculate_interval_bins with single value."""
        normal_intervals = np.array([0.1])
        display_min, display_max, num_bins = rps_distribution_plot._calculate_interval_bins(
            normal_intervals
        )
        self.assertLess(display_min, display_max)
        self.assertEqual(num_bins, 1)

    def test_calculate_interval_bins_same_values(self):
        """Test _calculate_interval_bins with same values."""
        normal_intervals = np.array([0.1, 0.1, 0.1])
        display_min, display_max, num_bins = rps_distribution_plot._calculate_interval_bins(
            normal_intervals
        )
        self.assertLess(display_min, display_max)
        self.assertEqual(num_bins, 1)

    def test_calculate_interval_bins_low_std_dev(self):
        """Test _calculate_interval_bins with low standard deviation."""
        normal_intervals = np.array([0.1, 0.100001, 0.100002])
        display_min, display_max, num_bins = rps_distribution_plot._calculate_interval_bins(
            normal_intervals
        )
        self.assertIsInstance(display_min, float)
        self.assertIsInstance(display_max, float)
        self.assertIsInstance(num_bins, int)

    def test_calculate_max_normal_y_interval_empty(self):
        """Test _calculate_max_normal_y_interval with empty array."""
        normal_intervals = np.array([])
        max_y = rps_distribution_plot._calculate_max_normal_y_interval(
            normal_intervals, 0.0, 1.0, 10
        )
        self.assertEqual(max_y, 1.0)

    def test_calculate_theoretical_ramp_no_strategy(self):
        """Test _calculate_theoretical_ramp without strategy."""
        total_requests = 10
        cumulative_times, theoretical_rates = rps_distribution_plot._calculate_theoretical_ramp(
            total_requests, None, None, None, 5.0
        )
        self.assertEqual(len(cumulative_times), total_requests)
        self.assertEqual(len(theoretical_rates), total_requests)
        self.assertTrue(np.allclose(theoretical_rates, 5.0))

    def test_prepare_interval_data_empty(self):
        """Test _prepare_interval_data with empty array."""
        cumulative_delays = np.array([])
        intervals = rps_distribution_plot._prepare_interval_data(cumulative_delays)
        self.assertEqual(len(intervals), 0)

    def test_separate_normal_anomaly_intervals_empty(self):
        """Test _separate_normal_anomaly_intervals with empty array."""
        intervals = np.array([])
        timing_anomaly_indices = np.array([])
        normal_intervals, anomaly_intervals = rps_distribution_plot._separate_normal_anomaly_intervals(
            intervals, timing_anomaly_indices
        )
        self.assertEqual(len(normal_intervals), 0)
        self.assertEqual(len(anomaly_intervals), 0)

    def test_separate_normal_anomaly_intervals_no_anomalies(self):
        """Test _separate_normal_anomaly_intervals with no anomalies."""
        intervals = np.array([0.1, 0.2, 0.3])
        timing_anomaly_indices = np.array([])
        normal_intervals, anomaly_intervals = rps_distribution_plot._separate_normal_anomaly_intervals(
            intervals, timing_anomaly_indices
        )
        self.assertEqual(len(normal_intervals), 3)
        self.assertEqual(len(anomaly_intervals), 0)

    def test_density_based_sampling_small(self):
        """Test _density_based_sampling with small dataset."""
        time_points = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        values = np.array([10, 11, 12, 13, 14])
        sampled_time, sampled_values = rps_distribution_plot._density_based_sampling(
            time_points, values, max_samples=1000
        )
        self.assertEqual(len(sampled_time), len(time_points))
        self.assertEqual(len(sampled_values), len(values))

    def test_density_based_sampling_large(self):
        """Test _density_based_sampling with large dataset."""
        np.random.seed(42)
        time_points = np.random.rand(5000)
        values = np.random.rand(5000)
        sampled_time, sampled_values = rps_distribution_plot._density_based_sampling(
            time_points, values, max_samples=1000
        )
        self.assertLessEqual(len(sampled_time), 1000)
        self.assertLessEqual(len(sampled_values), 1000)

    def test_calculate_adaptive_window(self):
        """Test _calculate_adaptive_window function."""
        # Test different data sizes
        self.assertEqual(rps_distribution_plot._calculate_adaptive_window(500), 20)
        self.assertEqual(rps_distribution_plot._calculate_adaptive_window(5000), 50)
        self.assertEqual(rps_distribution_plot._calculate_adaptive_window(50000), 100)
        self.assertEqual(rps_distribution_plot._calculate_adaptive_window(200000), 200)

    def test_exponential_moving_average(self):
        """Test _exponential_moving_average function."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        window_size = 5
        result = rps_distribution_plot._exponential_moving_average(data, window_size)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(data) - window_size + 1)

    def test_exponential_moving_average_with_alpha(self):
        """Test _exponential_moving_average with custom alpha."""
        data = np.array([1, 2, 3, 4, 5])
        window_size = 3
        alpha = 0.5
        result = rps_distribution_plot._exponential_moving_average(data, window_size, alpha)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(data) - window_size + 1)

    @patch('ais_bench.benchmark.utils.visualization.rps_distribution_plot._add_legend_explanation_table')
    @patch('ais_bench.benchmark.utils.visualization.rps_distribution_plot._add_interval_traces')
    @patch('ais_bench.benchmark.utils.visualization.rps_distribution_plot._add_classic_rps_traces')
    @patch('ais_bench.benchmark.utils.visualization.rps_distribution_plot._add_time_rps_traces')
    def test_create_chart_figure(self, mock_add_time, mock_add_classic, mock_add_interval, mock_add_legend):
        """Test _create_chart_figure function."""
        mock_add_time.return_value = ["Time - RPS: Normal RPS"]
        mock_add_classic.return_value = ["RPS - Request Count: Normal Request Count"]
        mock_add_interval.return_value = ["Gamma Dist: Normal Intervals"]

        normal_time_points = np.array([0.1, 0.2, 0.3])
        normal_rps_values = np.array([10, 11, 12])
        timing_anomaly_time_points = np.array([])
        timing_anomaly_rps_values = np.array([])
        burstiness_anomaly_time_points = np.array([])
        burstiness_anomaly_rps_values = np.array([])
        finite_normal_rps = np.array([10, 11, 12])
        normal_intervals = np.array([0.1, 0.1, 0.1])
        timing_anomaly_intervals = np.array([])

        fig = rps_distribution_plot._create_chart_figure(
            normal_time_points, normal_rps_values,
            timing_anomaly_time_points, timing_anomaly_rps_values,
            burstiness_anomaly_time_points, burstiness_anomaly_rps_values,
            finite_normal_rps, 1.0,
            normal_intervals, timing_anomaly_intervals, 1.0,
            10.0, 1.0, None, None, None,
            5.0, 15.0, 10,
            0.0, 1.0, 10,
            normal_time_points, "Test Title"
        )

        self.assertIsNotNone(fig)
        mock_add_time.assert_called_once()
        mock_add_classic.assert_called_once()
        mock_add_interval.assert_called_once()
        mock_add_legend.assert_called_once()

    @patch('ais_bench.benchmark.utils.visualization.rps_distribution_plot._density_based_sampling')
    def test_add_time_rps_traces_with_anomalies(self, mock_sampling):
        """Test _add_time_rps_traces with anomalies."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=1)
        mock_sampling.return_value = (np.array([0.1]), np.array([10]))

        normal_time_points = np.array([0.1, 0.2, 0.3])
        normal_rps_values = np.array([10, 11, 12])
        timing_anomaly_time_points = np.array([0.15])
        timing_anomaly_rps_values = np.array([100])
        burstiness_anomaly_time_points = np.array([0.25])
        burstiness_anomaly_rps_values = np.array([50])

        traces = rps_distribution_plot._add_time_rps_traces(
            fig, normal_time_points, normal_rps_values,
            timing_anomaly_time_points, timing_anomaly_rps_values,
            burstiness_anomaly_time_points, burstiness_anomaly_rps_values,
            10.0, None, None, None, "Test: ", row=1, col=1
        )

        self.assertIsInstance(traces, list)
        self.assertGreater(len(traces), 0)

    @patch('ais_bench.benchmark.utils.visualization.rps_distribution_plot._density_based_sampling')
    def test_add_time_rps_traces_with_ewma(self, mock_sampling):
        """Test _add_time_rps_traces with EWMA."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=1)
        mock_sampling.return_value = (np.array([0.1]), np.array([10]))

        # Create large dataset to trigger EWMA
        normal_time_points = np.array([0.1 * i for i in range(100)])
        normal_rps_values = np.array([10 + i * 0.1 for i in range(100)])

        traces = rps_distribution_plot._add_time_rps_traces(
            fig, normal_time_points, normal_rps_values,
            np.array([]), np.array([]),
            np.array([]), np.array([]),
            10.0, None, None, None, "Test: ", row=1, col=1
        )

        self.assertIsInstance(traces, list)

    @patch('ais_bench.benchmark.utils.visualization.rps_distribution_plot._density_based_sampling')
    def test_add_time_rps_traces_with_ramp_up(self, mock_sampling):
        """Test _add_time_rps_traces with ramp up."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=1)
        mock_sampling.return_value = (np.array([0.1]), np.array([10]))

        normal_time_points = np.array([0.1, 0.2, 0.3])
        normal_rps_values = np.array([10, 11, 12])

        traces = rps_distribution_plot._add_time_rps_traces(
            fig, normal_time_points, normal_rps_values,
            np.array([]), np.array([]),
            np.array([]), np.array([]),
            10.0, "linear", 1.0, 10.0, "Test: ", row=1, col=1
        )

        self.assertIsInstance(traces, list)

    @patch('ais_bench.benchmark.utils.visualization.rps_distribution_plot._density_based_sampling')
    def test_add_classic_rps_traces_with_anomalies(self, mock_sampling):
        """Test _add_classic_rps_traces with anomalies."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=1)
        mock_sampling.return_value = (np.array([100]), np.array([100]))

        finite_normal_rps = np.array([10, 11, 12])
        timing_anomaly_rps_values = np.array([100])

        traces = rps_distribution_plot._add_classic_rps_traces(
            fig, finite_normal_rps, timing_anomaly_rps_values,
            5.0, 15.0, 10, "Test: ", row=1, col=1
        )

        self.assertIsInstance(traces, list)
        self.assertGreater(len(traces), 0)

    @patch('ais_bench.benchmark.utils.visualization.rps_distribution_plot._density_based_sampling')
    def test_add_interval_traces_with_anomalies(self, mock_sampling):
        """Test _add_interval_traces with anomalies."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=1)
        mock_sampling.return_value = (np.array([0.001]), np.array([0.001]))

        normal_intervals = np.array([0.1, 0.2, 0.3])
        timing_anomaly_intervals = np.array([0.001])

        traces = rps_distribution_plot._add_interval_traces(
            fig, normal_intervals, timing_anomaly_intervals,
            0.0, 1.0, 10, "Test: ", row=1, col=1
        )

        self.assertIsInstance(traces, list)
        self.assertGreater(len(traces), 0)

    def test_add_legend_explanation_table(self):
        """Test _add_legend_explanation_table function."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create subplot with table support at (2, 2)
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "table"}]]
        )
        traces_names_dict = {
            "Time - RPS: ": ["Time - RPS: Normal RPS"],
            "RPS - Request Count: ": ["RPS - Request Count: Normal Request Count"],
            "Gamma Dist: ": ["Gamma Dist: Normal Intervals"]
        }

        rps_distribution_plot._add_legend_explanation_table(fig, traces_names_dict, row=2, col=2)

        # Check that a table trace was added
        table_traces = [trace for trace in fig.data if isinstance(trace, go.Table)]
        self.assertGreater(len(table_traces), 0)

    @patch('ais_bench.benchmark.utils.visualization.rps_distribution_plot.logger')
    def test_log_statistics(self, mock_logger):
        """Test _log_statistics function."""
        cumulative_delays = np.array([0.1, 0.2, 0.3])
        finite_normal_rps = np.array([10, 11, 12])
        timing_anomaly_rps_values = np.array([])
        burstiness_anomaly_rps_values = np.array([])
        time_points = np.array([0.1, 0.2, 0.3])
        intervals = np.array([0.1, 0.1, 0.1])
        normal_intervals = np.array([0.1, 0.1, 0.1])
        timing_anomaly_intervals = np.array([])

        rps_distribution_plot._log_statistics(
            cumulative_delays, finite_normal_rps, timing_anomaly_rps_values,
            burstiness_anomaly_rps_values, time_points, intervals,
            normal_intervals, timing_anomaly_intervals, 10.0, 1.0, 0
        )

        mock_logger.info.assert_called()

    @patch('ais_bench.benchmark.utils.visualization.rps_distribution_plot.logger')
    def test_log_statistics_mismatch(self, mock_logger):
        """Test _log_statistics with count mismatch."""
        cumulative_delays = np.array([0.1, 0.2, 0.3, 0.4])
        finite_normal_rps = np.array([10, 11])
        timing_anomaly_rps_values = np.array([100])
        burstiness_anomaly_rps_values = np.array([])
        time_points = np.array([0.1, 0.2, 0.3, 0.4])
        intervals = np.array([0.1, 0.1, 0.1, 0.1])
        normal_intervals = np.array([0.1, 0.1])
        timing_anomaly_intervals = np.array([0.001])

        rps_distribution_plot._log_statistics(
            cumulative_delays, finite_normal_rps, timing_anomaly_rps_values,
            burstiness_anomaly_rps_values, time_points, intervals,
            normal_intervals, timing_anomaly_intervals, 10.0, 1.0, 0
        )

        mock_logger.warning.assert_called()

    def test_is_valid_chart_html_file(self):
        """Test _is_valid_chart_html_file function."""
        import tempfile
        import os

        # Test with None
        self.assertFalse(rps_distribution_plot._is_valid_chart_html_file(None))

        # Test with non-HTML file
        self.assertFalse(rps_distribution_plot._is_valid_chart_html_file("test.txt"))

        # Test with non-existent file
        self.assertFalse(rps_distribution_plot._is_valid_chart_html_file("nonexistent.html"))

        # Test with valid HTML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write("<html></html>")
            temp_path = f.name

        try:
            self.assertTrue(rps_distribution_plot._is_valid_chart_html_file(temp_path))
        finally:
            os.unlink(temp_path)

    def test_determine_output_path(self):
        """Test _determine_output_path function."""
        import tempfile
        import os

        # Test with valid HTML file and no output_name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write("<html></html>")
            temp_path = f.name

        try:
            output_path = rps_distribution_plot._determine_output_path(temp_path, None)
            self.assertTrue(output_path.endswith('_with_actual_rps.html'))
        finally:
            os.unlink(temp_path)

        # Test with directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = rps_distribution_plot._determine_output_path(tmpdir, None)
            self.assertIn('rps_distribution_plot_with_actual_rps.html', output_path)

        # Test with output_name
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = rps_distribution_plot._determine_output_path(tmpdir, "test_output")
            self.assertTrue(output_path.endswith('.html'))
            self.assertIn('test_output', output_path)

    def test_create_time_rps_trace(self):
        """Test _create_time_rps_trace function."""
        post_time_list = [0.1, 0.2, 0.3, 0.4, 0.5]
        trace = rps_distribution_plot._create_time_rps_trace(post_time_list)
        self.assertIsNotNone(trace)
        self.assertEqual(trace.mode, "lines+markers")

    def test_create_time_rps_trace_large(self):
        """Test _create_time_rps_trace with large dataset."""
        post_time_list = [0.1 * i for i in range(6000)]
        trace = rps_distribution_plot._create_time_rps_trace(post_time_list)
        self.assertIsNotNone(trace)

    def test_prepare_actual_rps_data(self):
        """Test _prepare_actual_rps_data function."""
        # Ensure global mask is None
        original_mask = rps_distribution_plot._NORMAL_INDICES_MASK
        rps_distribution_plot._NORMAL_INDICES_MASK = None
        try:
            post_time_list = [0.3, 0.1, 0.2, 0.5, 0.4]
            time_points, rps_values = rps_distribution_plot._prepare_actual_rps_data(post_time_list)
            self.assertEqual(len(time_points), len(post_time_list))
            self.assertEqual(len(rps_values), len(post_time_list))
            # Check that times are sorted
            self.assertTrue(np.all(time_points[:-1] <= time_points[1:]))
        finally:
            rps_distribution_plot._NORMAL_INDICES_MASK = original_mask

    def test_prepare_actual_rps_data_with_mask(self):
        """Test _prepare_actual_rps_data with global mask."""
        # Set global mask
        rps_distribution_plot._NORMAL_INDICES_MASK = np.array([0, 1, 2])
        try:
            post_time_list = [0.1, 0.2, 0.3, 0.4, 0.5]
            time_points, rps_values = rps_distribution_plot._prepare_actual_rps_data(post_time_list)
            self.assertEqual(len(time_points), 3)
            self.assertEqual(len(rps_values), 3)
        finally:
            rps_distribution_plot._NORMAL_INDICES_MASK = None

    def test_update_legend_explanation(self):
        """Test _update_legend_explanation function."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create subplot with table support
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "table"}]]
        )
        # Add a table trace
        fig.add_trace(go.Table(
            header=dict(values=['Group', 'Trace', 'Meaning', 'Calc', 'Criteria', 'Visual']),
            cells=dict(values=[
                ['Time - RPS: ', '', ''],
                ['', 'Normal RPS', 'Test'],
                ['', '', ''],
                ['', '', ''],
                ['', '', ''],
                ['', '', '']
            ])
        ), row=1, col=1)

        description = ("Test", "Test calc", "Test criteria", "Test visual")
        rps_distribution_plot._update_legend_explanation(fig, "Test Trace", description)

        # Check that table was updated
        table_traces = [trace for trace in fig.data if isinstance(trace, go.Table)]
        self.assertGreater(len(table_traces), 0)

    def test_update_legend_explanation_no_table(self):
        """Test _update_legend_explanation without table."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=1)
        description = ("Test", "Test calc", "Test criteria", "Test visual")
        # Should not raise error
        rps_distribution_plot._update_legend_explanation(fig, "Test Trace", description)

    @patch('ais_bench.benchmark.utils.visualization.rps_distribution_plot.logger')
    def test_export_to_html_exception(self, mock_logger):
        """Test _export_to_html with exception."""
        mock_fig = MagicMock()
        mock_fig.write_html.side_effect = Exception("Test error")
        test_file = "test.html"
        rps_distribution_plot._export_to_html(mock_fig, test_file)
        mock_logger.warning.assert_called()
        # Track file for cleanup
        self.files_to_cleanup.append(test_file)
        self.files_to_cleanup.append(test_file.replace('.html', '.json'))

    def test_convert_numpy_to_list(self):
        """Test _convert_numpy_to_list function."""
        # Test with numpy array
        arr = np.array([1, 2, 3])
        result = rps_distribution_plot._convert_numpy_to_list(arr)
        self.assertIsInstance(result, list)

        # Test with numpy scalar
        scalar = np.int64(42)
        result = rps_distribution_plot._convert_numpy_to_list(scalar)
        self.assertIsInstance(result, int)

        # Test with dict
        d = {'a': np.array([1, 2]), 'b': 3}
        result = rps_distribution_plot._convert_numpy_to_list(d)
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result['a'], list)

        # Test with list
        lst = [np.array([1, 2]), np.array([3, 4])]
        result = rps_distribution_plot._convert_numpy_to_list(lst)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], list)

    @patch('ais_bench.benchmark.utils.visualization.rps_distribution_plot.logger')
    def test_export_to_json_exception(self, mock_logger):
        """Test _export_to_json with exception."""
        mock_fig = MagicMock()
        mock_fig.to_dict.side_effect = Exception("Test error")
        test_file = "test.json"
        rps_distribution_plot._export_to_json(mock_fig, test_file)
        mock_logger.warning.assert_called()
        # Track file for cleanup
        self.files_to_cleanup.append(test_file)

    def test_merge_into_subplot_with_figure(self):
        """Test _merge_into_subplot with Figure objects."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        dst_fig = make_subplots(rows=2, cols=2)
        src_trace = go.Scatter(x=[1, 2, 3], y=[1, 2, 3], name="Test")

        result = rps_distribution_plot._merge_into_subplot(
            dst_fig, src_trace, row=1, col=1
        )

        self.assertIsNotNone(result)

    def test_merge_into_subplot_with_dict(self):
        """Test _merge_into_subplot with dict."""
        import plotly.graph_objects as go

        dst_dict = {
            'data': [{'type': 'scatter', 'x': [1, 2], 'y': [1, 2]}],
            'layout': {}
        }
        src_trace = go.Scatter(x=[1, 2, 3], y=[1, 2, 3], name="Test")

        result = rps_distribution_plot._merge_into_subplot(
            dst_dict, src_trace, row=1, col=1
        )

        self.assertIsNotNone(result)

    def test_merge_into_subplot_with_callback(self):
        """Test _merge_into_subplot with callback."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        dst_fig = make_subplots(rows=2, cols=2)
        src_trace = go.Scatter(x=[1, 2, 3], y=[1, 2, 3], name="Test")

        callback_called = [False]
        def callback(fig, src_dict):
            callback_called[0] = True

        result = rps_distribution_plot._merge_into_subplot(
            dst_fig, src_trace, row=1, col=1, callback=callback
        )

        self.assertTrue(callback_called[0])

    def test_merge_into_subplot_dst_fails(self):
        """Test _merge_into_subplot when dst loading fails."""
        import plotly.graph_objects as go

        src_trace = go.Scatter(x=[1, 2, 3], y=[1, 2, 3], name="Test")

        # Use invalid path
        result = rps_distribution_plot._merge_into_subplot(
            "nonexistent.json", src_trace, row=1, col=1
        )

        # Should return None or handle gracefully
        # The function may return None or a figure depending on implementation
        self.assertTrue(result is None or isinstance(result, go.Figure))

    def test_plot_rps_distribution_no_anomalies(self):
        """Test plot_rps_distribution with no anomalies."""
        with patch('ais_bench.benchmark.utils.visualization.rps_distribution_plot._export_to_html') as mock_export, \
             patch('ais_bench.benchmark.utils.visualization.rps_distribution_plot._create_chart_figure') as mock_create, \
             patch('ais_bench.benchmark.utils.visualization.rps_distribution_plot._log_statistics') as mock_log:
            mock_fig = MagicMock()
            mock_create.return_value = mock_fig

            cumulative_delays = np.array([0.1, 0.2, 0.3])
            timing_anomaly_indices = np.array([])
            burstiness_anomaly_indices = np.array([])

            test_file = "test.html"
            rps_distribution_plot.plot_rps_distribution(
                cumulative_delays,
                timing_anomaly_indices,
                burstiness_anomaly_indices,
                10.0, 1.0, None, None, None, test_file
            )

            mock_create.assert_called_once()
            mock_export.assert_called_once()
            # Track files for cleanup in case mock fails
            self.files_to_cleanup.append(test_file)
            self.files_to_cleanup.append(test_file.replace('.html', '.json'))

if __name__ == '__main__':
    unittest.main()
