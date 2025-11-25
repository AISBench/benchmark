"""Unit tests for LEval Meeting Summ dataset loader."""

import pytest
from unittest.mock import patch
from datasets import Dataset, DatasetDict

from ais_bench.benchmark.datasets.leval.meeting_summ import LEvalMeetingSummDataset
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError


class TestLEvalMeetingSummDataset:
    @patch('ais_bench.benchmark.datasets.leval.meeting_summ.get_data_path')
    @patch('ais_bench.benchmark.datasets.leval.meeting_summ.load_dataset')
    def test_meeting_summ_load(self, mock_load_dataset, mock_get_data_path):
        # Mock the path resolution
        mock_get_data_path.return_value = "/mock/path"

        # Mock the dataset
        mock_dataset = DatasetDict({
            'test': Dataset.from_list([
                {
                    'instructions': ['Summarize the key points from this meeting'],
                    'outputs': ['The meeting discussed project timeline and budget allocation'],
                    'input': 'Meeting transcript content...'
                },
                {
                    'instructions': ['What decisions were made?'],
                    'outputs': ['Decisions included approving the new budget and extending deadlines'],
                    'input': 'Meeting transcript content...'
                }
            ])})
        mock_load_dataset.return_value = mock_dataset

        # Call the load method
        result = LEvalMeetingSummDataset.load(path="dummy_path")

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
        assert data_list[0]['question'] == 'Summarize the key points from this meeting'
        assert data_list[0]['context'] == 'Meeting transcript content...'
        assert data_list[0]['answer'] == 'The meeting discussed project timeline and budget allocation'
        assert data_list[0]['length'] == 8  # 8 words

        # Second item
        assert data_list[1]['question'] == 'What decisions were made?'
        assert data_list[1]['context'] == 'Meeting transcript content...'
        assert data_list[1]['answer'] == 'Decisions included approving the new budget and extending deadlines'
        assert data_list[1]['length'] == 9  # 9 words

    def test_meeting_summ_load_missing_path(self):
        """Test that ConfigError is raised when 'path' argument is missing."""
        with pytest.raises(AISBenchConfigError) as exc_info:
            LEvalMeetingSummDataset.load()
        
        # Verify the error message contains helpful information
        assert "path" in str(exc_info.value).lower()
