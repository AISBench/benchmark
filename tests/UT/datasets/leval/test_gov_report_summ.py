
import pytest
from unittest.mock import patch
from datasets import Dataset, DatasetDict

from ais_bench.benchmark.datasets.leval.gov_report_summ import LEvalGovReportSummDataset
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError

class TestLEvalGovReportSummDataset:
    @patch('ais_bench.benchmark.datasets.leval.gov_report_summ.get_data_path')
    @patch('ais_bench.benchmark.datasets.leval.gov_report_summ.load_dataset')
    def test_gov_report_summ_load(self, mock_load_dataset, mock_get_data_path):
        # Mock the path resolution
        mock_get_data_path.return_value = "/mock/path"

        # Mock the dataset
        mock_dataset = DatasetDict({
            'test': Dataset.from_list([
                {
                    'instructions': ['Summarize this report'],
                    'outputs': ['This is a summary of the government report'],
                    'input': 'Report content here...'
                },
                {
                    'instructions': ['What is the main finding?'],
                    'outputs': ['The main finding is that...'],
                    'input': 'Report content here...'
                }
            ])})
        mock_load_dataset.return_value = mock_dataset

        # Call the load method
        result = LEvalGovReportSummDataset.load(path="dummy_path")

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
        assert data_list[0]['question'] == 'Summarize this report'
        assert data_list[0]['context'] == 'Report content here...'
        assert data_list[0]['answer'] == 'This is a summary of the government report'
        assert data_list[0]['length'] == 8  # 8 words

        # Second item
        assert data_list[1]['question'] == 'What is the main finding?'
        assert data_list[1]['context'] == 'Report content here...'
        assert data_list[1]['answer'] == 'The main finding is that...'
        assert data_list[1]['length'] == 5  # 5 words

    def test_gov_report_summ_load_missing_path(self):
        """Test that ConfigError is raised when 'path' argument is missing."""
        with pytest.raises(AISBenchConfigError) as exc_info:
            LEvalGovReportSummDataset.load()
        
        # Verify the error message contains helpful information
        assert "path" in str(exc_info.value).lower()
