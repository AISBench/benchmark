"""Unit tests for LEval Scientific QA dataset loader."""

import pytest
from unittest.mock import patch
from datasets import Dataset, DatasetDict

from ais_bench.benchmark.datasets.leval.scientific_qa import LEvalScientificQADataset
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError


class TestLEvalScientificQADataset:
    @patch('ais_bench.benchmark.datasets.leval.scientific_qa.get_data_path')
    @patch('ais_bench.benchmark.datasets.leval.scientific_qa.load_dataset')
    def test_scientific_qa_load(self, mock_load_dataset, mock_get_data_path):
        # Mock the path resolution
        mock_get_data_path.return_value = "/mock/path"

        # Mock the dataset
        mock_dataset = DatasetDict({
            'test': Dataset.from_list([
                {
                    'instructions': ['What is the scientific finding?'],
                    'outputs': ['The research demonstrates significant correlations between variables studied'],
                    'input': 'Scientific paper content...'
                },
                {
                    'instructions': ['Explain the methodology used'],
                    'outputs': ['The study employed experimental methods with statistical analysis techniques'],
                    'input': 'Scientific paper content...'
                }
            ])})
        mock_load_dataset.return_value = mock_dataset

        # Call the load method
        result = LEvalScientificQADataset.load(path="dummy_path")

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
        assert data_list[0]['question'] == 'What is the scientific finding?'
        assert data_list[0]['context'] == 'Scientific paper content...'
        assert data_list[0]['answer'] == 'The research demonstrates significant correlations between variables studied'
        assert data_list[0]['length'] == 8  # 8 words

        # Second item
        assert data_list[1]['question'] == 'Explain the methodology used'
        assert data_list[1]['context'] == 'Scientific paper content...'
        assert data_list[1]['answer'] == 'The study employed experimental methods with statistical analysis techniques'
        assert data_list[1]['length'] == 9  # 9 words

    def test_scientific_qa_load_missing_path(self):
        """Test that ConfigError is raised when 'path' argument is missing."""
        with pytest.raises(AISBenchConfigError) as exc_info:
            LEvalScientificQADataset.load()
        
        # Verify the error message contains helpful information
        assert "path" in str(exc_info.value).lower()
