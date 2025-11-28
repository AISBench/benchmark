import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from ais_bench.benchmark.utils.visualization import summarize_plot

class TestSummarizePlot(unittest.TestCase):
    def setUp(self):
        self.start_times = np.array([0.0, 1.0, 2.0])
        self.prefill_latencies = np.array([0.1, 0.1, 0.1])
        self.end_times = np.array([0.5, 1.5, 2.5])
        self.multiturn_group_id_list = [0, 1, 2]
        self.output_file = "test_timeline.html"

    def test_validate_input_data_valid(self):
        result = summarize_plot.validate_input_data(
            [0.0, 1.0], [0.1, 0.1], [0.5, 1.5]
        )
        self.assertTrue(result)

    def test_validate_input_data_empty(self):
        result = summarize_plot.validate_input_data([], [], [])
        self.assertFalse(result)

    def test_validate_input_data_mismatch(self):
        result = summarize_plot.validate_input_data(
            [0.0], [0.1, 0.1], [0.5]
        )
        self.assertFalse(result)

    def test_is_non_streaming_scenario(self):
        self.assertTrue(summarize_plot.is_non_streaming_scenario([0.0, 0.0]))
        self.assertFalse(summarize_plot.is_non_streaming_scenario([0.1, 0.0]))

    def test_preprocess_data_streaming(self):
        start_times = np.array([0.0, 1.0])
        prefill_latencies = np.array([0.1, 0.1])
        end_times = np.array([0.5, 1.5])
        
        first_tokens, adj_starts, adj_ends, is_non_streaming = summarize_plot.preprocess_data(
            start_times, prefill_latencies, end_times
        )
        
        self.assertFalse(is_non_streaming)
        np.testing.assert_array_almost_equal(first_tokens, np.array([0.1, 1.1]))
        np.testing.assert_array_almost_equal(adj_starts, np.array([0.0, 1.0]))
        np.testing.assert_array_almost_equal(adj_ends, np.array([0.5, 1.5]))

    def test_preprocess_data_non_streaming(self):
        start_times = np.array([0.0, 1.0])
        prefill_latencies = np.array([0.0, 0.0])
        end_times = np.array([0.5, 1.5])
        
        first_tokens, adj_starts, adj_ends, is_non_streaming = summarize_plot.preprocess_data(
            start_times, prefill_latencies, end_times
        )
        
        self.assertTrue(is_non_streaming)
        self.assertIsNone(first_tokens)
        np.testing.assert_array_almost_equal(adj_starts, np.array([0.0, 1.0]))
        np.testing.assert_array_almost_equal(adj_ends, np.array([0.5, 1.5]))

    @patch('ais_bench.benchmark.utils.visualization.summarize_plot.make_subplots')
    @patch('plotly.graph_objects.Figure')
    def test_plot_sorted_request_timelines_streaming(self, mock_figure, mock_make_subplots):
        mock_fig = MagicMock()
        mock_make_subplots.return_value = mock_fig
        
        result = summarize_plot.plot_sorted_request_timelines(
            self.start_times,
            self.end_times,
            self.prefill_latencies,
            self.multiturn_group_id_list,
            self.output_file
        )
        
        self.assertTrue(result)
        mock_make_subplots.assert_called_once()
        mock_fig.write_html.assert_called_once()

    @patch('plotly.graph_objects.Figure')
    def test_plot_sorted_request_timelines_non_streaming(self, mock_figure):
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        prefill_latencies = np.zeros_like(self.prefill_latencies)
        
        result = summarize_plot.plot_sorted_request_timelines(
            self.start_times,
            self.end_times,
            prefill_latencies,
            self.multiturn_group_id_list,
            self.output_file
        )
        
        self.assertTrue(result)
        mock_figure.assert_called_once()
        mock_fig.write_html.assert_called_once()

    def test_generate_timeline_traces(self):
        adjusted_starts = np.array([0.0, 1.0])
        adjusted_ends = np.array([0.5, 1.5])
        adjusted_first_tokens = np.array([0.1, 1.1])
        multiturn_group_id_list = [0, 1]
        unit = "s"
        
        traces = summarize_plot.generate_timeline_traces(
            adjusted_starts,
            adjusted_ends,
            adjusted_first_tokens,
            multiturn_group_id_list,
            unit
        )
        
        # Should return traces for red lines and blue lines
        self.assertGreater(len(traces), 0)

    def test_generate_concurrency_traces(self):
        adjusted_starts = np.array([0.0, 1.0])
        adjusted_ends = np.array([0.5, 1.5])
        unit = "s"
        
        traces = summarize_plot.generate_concurrency_traces(
            adjusted_starts,
            adjusted_ends,
            unit
        )
        
        self.assertGreater(len(traces), 0)

if __name__ == '__main__':
    unittest.main()
