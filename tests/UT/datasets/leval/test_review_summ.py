"""Unit tests for LEval Review Summ dataset loader."""

import pytest
from unittest.mock import patch
from datasets import Dataset, DatasetDict

from ais_bench.benchmark.datasets.leval.review_summ import LEvalReviewSummDataset
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError


class TestLEvalReviewSummDataset:
    @patch('ais_bench.benchmark.datasets.leval.review_summ.get_data_path')
    @patch('ais_bench.benchmark.datasets.leval.review_summ.load_dataset')
    def test_review_summ_load(self, mock_load_dataset, mock_get_data_path):
        # Mock the path resolution
        mock_get_data_path.return_value = "/mock/path"

        # Mock the dataset
        mock_dataset = DatasetDict({
            'test': Dataset.from_list([
                {
                    'instructions': ['Summarize this product review'],
                    'outputs': ['The review highlights both positive aspects and some concerns about quality'],
                    'input': 'Product review content...'
                },
                {
                    'instructions': ['What is the overall rating?'],
                    'outputs': ['The overall rating is four stars with mixed feedback'],
                    'input': 'Product review content...'
                }
            ])})
        mock_load_dataset.return_value = mock_dataset

        # Call the load method
        result = LEvalReviewSummDataset.load(path="dummy_path")

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
        assert data_list[0]['question'] == 'Summarize this product review'
        assert data_list[0]['context'] == 'Product review content...'
        assert data_list[0]['answer'] == 'The review highlights both positive aspects and some concerns about quality'
        assert data_list[0]['length'] == 11  # 11 words

        # Second item
        assert data_list[1]['question'] == 'What is the overall rating?'
        assert data_list[1]['context'] == 'Product review content...'
        assert data_list[1]['answer'] == 'The overall rating is four stars with mixed feedback'
        assert data_list[1]['length'] == 9  # 9 words

    def test_review_summ_load_missing_path(self):
        """Test that ConfigError is raised when 'path' argument is missing."""
        with pytest.raises(AISBenchConfigError) as exc_info:
            LEvalReviewSummDataset.load()
        
        # Verify the error message contains helpful information
        assert "path" in str(exc_info.value).lower()
