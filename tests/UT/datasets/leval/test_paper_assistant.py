"""Unit tests for LEval Paper Assistant dataset loader."""

import pytest
from unittest.mock import patch
from datasets import Dataset, DatasetDict

from ais_bench.benchmark.datasets.leval.paper_assistant import LEvalPaperAssistantDataset
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError


class TestLEvalPaperAssistantDataset:
    @patch('ais_bench.benchmark.datasets.leval.paper_assistant.get_data_path')
    @patch('ais_bench.benchmark.datasets.leval.paper_assistant.load_dataset')
    def test_paper_assistant_load(self, mock_load_dataset, mock_get_data_path):
        # Mock the path resolution
        mock_get_data_path.return_value = "/mock/path"

        # Mock the dataset
        mock_dataset = DatasetDict({
            'test': Dataset.from_list([
                {
                    'instructions': ['Help analyze this research paper'],
                    'outputs': ['The paper presents novel findings in machine learning applications'],
                    'input': 'Research paper content...'
                },
                {
                    'instructions': ['What are the key contributions?'],
                    'outputs': ['Key contributions include new algorithms and experimental validation'],
                    'input': 'Research paper content...'
                }
            ])})
        mock_load_dataset.return_value = mock_dataset

        # Call the load method
        result = LEvalPaperAssistantDataset.load(path="dummy_path")

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
        assert data_list[0]['question'] == 'Help analyze this research paper'
        assert data_list[0]['context'] == 'Research paper content...'
        assert data_list[0]['answer'] == 'The paper presents novel findings in machine learning applications'
        assert data_list[0]['length'] == 9  # 9 words

        # Second item
        assert data_list[1]['question'] == 'What are the key contributions?'
        assert data_list[1]['context'] == 'Research paper content...'
        assert data_list[1]['answer'] == 'Key contributions include new algorithms and experimental validation'
        assert data_list[1]['length'] == 8  # 8 words

    def test_paper_assistant_load_missing_path(self):
        """Test that ConfigError is raised when 'path' argument is missing."""
        with pytest.raises(AISBenchConfigError) as exc_info:
            LEvalPaperAssistantDataset.load()
        
        # Verify the error message contains helpful information
        assert "path" in str(exc_info.value).lower()