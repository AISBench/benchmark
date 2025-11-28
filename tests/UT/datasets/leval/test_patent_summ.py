"""Unit tests for LEval Patent Summ dataset loader."""

import pytest
from unittest.mock import patch
from datasets import Dataset, DatasetDict

from ais_bench.benchmark.datasets.leval.patent_summ import LEvalPatentSummDataset
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError


class TestLEvalPatentSummDataset:
    @patch('ais_bench.benchmark.datasets.leval.patent_summ.get_data_path')
    @patch('ais_bench.benchmark.datasets.leval.patent_summ.load_dataset')
    def test_patent_summ_load(self, mock_load_dataset, mock_get_data_path):
        # Mock the path resolution
        mock_get_data_path.return_value = "/mock/path"

        # Mock the dataset
        mock_dataset = DatasetDict({
            'test': Dataset.from_list([
                {
                    'instructions': ['Summarize this patent document'],
                    'outputs': ['The patent describes a novel invention in the field of technology'],
                    'input': 'Patent document content...'
                },
                {
                    'instructions': ['What is claimed in this patent?'],
                    'outputs': ['The patent claims a new method and system for data processing'],
                    'input': 'Patent document content...'
                }
            ])})
        mock_load_dataset.return_value = mock_dataset

        # Call the load method
        result = LEvalPatentSummDataset.load(path="dummy_path")

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
        assert data_list[0]['question'] == 'Summarize this patent document'
        assert data_list[0]['context'] == 'Patent document content...'
        assert data_list[0]['answer'] == 'The patent describes a novel invention in the field of technology'
        assert data_list[0]['length'] == 11  # 11 words

        # Second item
        assert data_list[1]['question'] == 'What is claimed in this patent?'
        assert data_list[1]['context'] == 'Patent document content...'
        assert data_list[1]['answer'] == 'The patent claims a new method and system for data processing'
        assert data_list[1]['length'] == 11  # 11 words

    def test_patent_summ_load_missing_path(self):
        """Test that ConfigError is raised when 'path' argument is missing."""
        with pytest.raises(AISBenchConfigError) as exc_info:
            LEvalPatentSummDataset.load()
        
        # Verify the error message contains helpful information
        assert "path" in str(exc_info.value).lower()