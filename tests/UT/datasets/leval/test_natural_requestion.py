"""Unit tests for LEval Natural Question dataset loader."""

import pytest
from unittest.mock import patch
from datasets import Dataset, DatasetDict

from ais_bench.benchmark.datasets.leval.natural_requestion import LEvalNaturalQuestionDataset
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError


class TestLEvalNaturalQuestionDataset:
    @patch('ais_bench.benchmark.datasets.leval.natural_requestion.get_data_path')
    @patch('ais_bench.benchmark.datasets.leval.natural_requestion.load_dataset')
    def test_natural_question_load(self, mock_load_dataset, mock_get_data_path):
        # Mock the path resolution
        mock_get_data_path.return_value = "/mock/path"

        # Mock the dataset
        mock_dataset = DatasetDict({
            'test': Dataset.from_list([
                {
                    'instructions': ['How does this work?'],
                    'outputs': ['This works by following a specific process and methodology'],
                    'input': 'Natural question context...'
                },
                {
                    'instructions': ['What are the benefits?'],
                    'outputs': ['The benefits include efficiency and improved outcomes'],
                    'input': 'Natural question context...'
                }
            ])})
        mock_load_dataset.return_value = mock_dataset

        # Call the load method
        result = LEvalNaturalQuestionDataset.load(path="dummy_path")

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
        assert data_list[0]['question'] == 'How does this work?'
        assert data_list[0]['context'] == 'Natural question context...'
        assert data_list[0]['answer'] == 'This works by following a specific process and methodology'
        assert data_list[0]['length'] == 9  # 9 words

        # Second item
        assert data_list[1]['question'] == 'What are the benefits?'
        assert data_list[1]['context'] == 'Natural question context...'
        assert data_list[1]['answer'] == 'The benefits include efficiency and improved outcomes'
        assert data_list[1]['length'] == 7  # 7 words

    def test_natural_question_load_missing_path(self):
        """Test that ConfigError is raised when 'path' argument is missing."""
        with pytest.raises(AISBenchConfigError) as exc_info:
            LEvalNaturalQuestionDataset.load()
        
        # Verify the error message contains helpful information
        assert "path" in str(exc_info.value).lower()
