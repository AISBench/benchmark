"""Unit tests for LEval TV Show Summ dataset loader."""

import pytest
from unittest.mock import patch
from datasets import Dataset, DatasetDict

from ais_bench.benchmark.datasets.leval.tv_show_summ import LEvalTVShowSummDataset
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError


class TestLEvalTVShowSummDataset:
    @patch('ais_bench.benchmark.datasets.leval.tv_show_summ.get_data_path')
    @patch('ais_bench.benchmark.datasets.leval.tv_show_summ.load_dataset')
    def test_tv_show_summ_load(self, mock_load_dataset, mock_get_data_path):
        # Mock the path resolution
        mock_get_data_path.return_value = "/mock/path"

        # Mock the dataset
        mock_dataset = DatasetDict({
            'test': Dataset.from_list([
                {
                    'instructions': ['Summarize this TV show episode'],
                    'outputs': ['The episode follows the characters through dramatic events and revelations'],
                    'input': 'TV show transcript content...'
                },
                {
                    'instructions': ['What happens in the climax?'],
                    'outputs': ['The climax features intense confrontation and unexpected plot twists'],
                    'input': 'TV show transcript content...'
                }
            ])})
        mock_load_dataset.return_value = mock_dataset

        # Call the load method
        result = LEvalTVShowSummDataset.load(path="dummy_path")

        # Verify get_data_path was called
        mock_get_data_path.assert_called_once_with(
            "dummy_path", local_mode=True)

        # Verify load_dataset was called
        mock_load_dataset.assert_called_once_with(
            "json", data_files={'test': "/mock/path"})

        # Check the result structure
        assert 'test' in result
        test_data = result['test']
        assert isinstance(test_data, Dataset)

        # Check the processed data
        data_list = test_data.to_list()
        assert len(data_list) == 2

        # First item
        assert data_list[0]['question'] == 'Summarize this TV show episode'
        assert data_list[0]['context'] == 'TV show transcript content...'
        assert data_list[0]['answer'] == 'The episode follows the characters through dramatic events and revelations'
        assert data_list[0]['length'] == 10  # 10 words

        # Second item
        assert data_list[1]['question'] == 'What happens in the climax?'
        assert data_list[1]['context'] == 'TV show transcript content...'
        assert data_list[1]['answer'] == 'The climax features intense confrontation and unexpected plot twists'
        assert data_list[1]['length'] == 9  # 9 words

    def test_tv_show_summ_load_missing_path(self):
        """Test that ConfigError is raised when 'path' argument is missing."""
        with pytest.raises(AISBenchConfigError) as exc_info:
            LEvalTVShowSummDataset.load()
        
        # Verify the error message contains helpful information
        assert "path" in str(exc_info.value).lower()
