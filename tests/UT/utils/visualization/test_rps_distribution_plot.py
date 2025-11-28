import unittest
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

    @patch('json.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_export_to_json(self, mock_file, mock_json_dump):
        mock_fig = MagicMock()
        mock_fig.to_dict.return_value = {"data": []}
        filename = "test.json"
        
        rps_distribution_plot._export_to_json(mock_fig, filename)
        
        mock_file.assert_called_once_with(filename, 'w')
        mock_json_dump.assert_called_once()

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

if __name__ == '__main__':
    unittest.main()
