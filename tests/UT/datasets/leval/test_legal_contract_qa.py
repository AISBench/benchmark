"""Unit tests for LEval Legal Contract QA dataset loader."""

import pytest
from unittest.mock import patch
from datasets import Dataset, DatasetDict

from ais_bench.benchmark.datasets.leval.legal_contract_qa import LEvalLegalContractQADataset
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError


class TestLEvalLegalContractQADataset:
    @patch('ais_bench.benchmark.datasets.leval.legal_contract_qa.get_data_path')
    @patch('ais_bench.benchmark.datasets.leval.legal_contract_qa.load_dataset')
    def test_legal_contract_qa_load(self, mock_load_dataset, mock_get_data_path):
        # Mock the path resolution
        mock_get_data_path.return_value = "/mock/path"

        # Mock the dataset
        mock_dataset = DatasetDict({
            'test': Dataset.from_list([
                {
                    'instructions': ['What are the terms of this contract?'],
                    'outputs': ['The contract terms include payment and delivery conditions'],
                    'input': 'Contract text content...'
                },
                {
                    'instructions': ['Who are the parties involved?'],
                    'outputs': ['Party A and Party B are the contracting parties'],
                    'input': 'Contract text content...'
                }
            ])})
        mock_load_dataset.return_value = mock_dataset

        # Call the load method
        result = LEvalLegalContractQADataset.load(path="dummy_path")

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
        assert data_list[0]['question'] == 'What are the terms of this contract?'
        assert data_list[0]['context'] == 'Contract text content...'
        assert data_list[0]['answer'] == 'The contract terms include payment and delivery conditions'
        assert data_list[0]['length'] == 8  # 8 words

        # Second item
        assert data_list[1]['question'] == 'Who are the parties involved?'
        assert data_list[1]['context'] == 'Contract text content...'
        assert data_list[1]['answer'] == 'Party A and Party B are the contracting parties'
        assert data_list[1]['length'] == 9  # 9 words

    def test_legal_contract_qa_load_missing_path(self):
        """Test that ConfigError is raised when 'path' argument is missing."""
        with pytest.raises(AISBenchConfigError) as exc_info:
            LEvalLegalContractQADataset.load()
        
        # Verify the error message contains helpful information
        assert "path" in str(exc_info.value).lower()