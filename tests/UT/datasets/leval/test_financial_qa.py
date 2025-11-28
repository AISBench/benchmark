"""Unit tests for LEval Financial QA dataset loader."""

import pytest
from unittest.mock import patch
from datasets import Dataset, DatasetDict

from ais_bench.benchmark.datasets.leval.financial_qa import LEvalFinancialQADataset
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError


class TestLEvalFinancialQADataset:
    @patch('ais_bench.benchmark.datasets.leval.financial_qa.get_data_path')
    @patch('ais_bench.benchmark.datasets.leval.financial_qa.load_dataset')
    def test_financial_qa_load(self, mock_load_dataset, mock_get_data_path):
        # Mock the path resolution
        mock_get_data_path.return_value = "/mock/path"

        # Mock the dataset
        mock_dataset = DatasetDict({
            'test': Dataset.from_list([
                {
                    'instructions': ['Q1'],
                    'outputs': ['This is a short answer'],
                    'input': 'Context1'
                },
                {
                    'instructions': ['Q2'],
                    'outputs': ['This is another answer with more words'],
                    'input': 'Context2'
                }
            ])})
        mock_load_dataset.return_value = mock_dataset

        # Call the load method
        result = LEvalFinancialQADataset.load(path="dummy_path")

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

        # First item - verify length calculation
        assert data_list[0]['question'] == 'Q1'
        assert data_list[0]['context'] == 'Context1'
        assert data_list[0]['answer'] == 'This is a short answer'
        assert data_list[0]['length'] == 5  # 5 words

        # Second item
        assert data_list[1]['question'] == 'Q2'
        assert data_list[1]['context'] == 'Context2'
        assert data_list[1]['answer'] == 'This is another answer with more words'
        assert data_list[1]['length'] == 7  # 7 words

    def test_financial_qa_load_missing_path(self):
        """Test that ConfigError is raised when 'path' argument is missing."""
        with pytest.raises(AISBenchConfigError) as exc_info:
            LEvalFinancialQADataset.load()
        
        # Verify the error message contains helpful information
        assert "path" in str(exc_info.value).lower()
