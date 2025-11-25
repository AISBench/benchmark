"""Unit test for LEval Sci-Fi dataset loader."""

import pytest
from unittest.mock import patch
from datasets import Dataset, DatasetDict

from ais_bench.benchmark.datasets.leval.sci_fi import LEvalSciFiDataset
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError


class TestLEvalSciFiDataset:
    @patch('ais_bench.benchmark.datasets.leval.sci_fi.get_data_path')
    @patch('ais_bench.benchmark.datasets.leval.sci_fi.load_dataset')
    def test_sci_fi_load(self, mock_load_dataset, mock_get_data_path):
        # Mock path resolution
        mock_get_data_path.return_value = "/mock/sci_fi.json"

        # Mock upstream dataset structure (nested instructions/outputs)
        mock_dataset = DatasetDict({
            'test': Dataset.from_list([
                {
                    'instructions': ['Is the protagonist loyal to the mission?'],
                    'outputs': ['True'],
                    'input': 'A long sci-fi narrative about a space mission and its crew.'
                },
                {
                    'instructions': ['Does the AI act against the crew?'],
                    'outputs': ['False'],
                    'input': 'A sequel narrative where the AI supports the crew decisions.'
                }
            ])
        })
        mock_load_dataset.return_value = mock_dataset

        # Invoke loader
        result = LEvalSciFiDataset.load(path="dummy_path")

        # Assertions: path resolution & dataset loading
        mock_get_data_path.assert_called_once_with("dummy_path", local_mode=True)
        mock_load_dataset.assert_called_once_with('json', data_files={'test': '/mock/sci_fi.json'})

        # Check structure
        assert 'test' in result
        flattened = result['test']
        assert isinstance(flattened, Dataset)

        rows = flattened.to_list()
        assert len(rows) == 2

        # Row 1
        assert rows[0]['question'] == 'Is the protagonist loyal to the mission?'
        assert 'space mission' in rows[0]['context']
        assert rows[0]['answer'] == 'True'

        # Row 2
        assert rows[1]['question'] == 'Does the AI act against the crew?'
        assert 'AI supports the crew'[:10] in rows[1]['context']  # loose contains
        assert rows[1]['answer'] == 'False'

        # Ensure no extraneous fields
        assert set(rows[0].keys()) == {'question', 'context', 'answer'}
        assert set(rows[1].keys()) == {'question', 'context', 'answer'}

    def test_sci_fi_load_missing_path(self):
        """Test that ConfigError is raised when 'path' argument is missing."""
        with pytest.raises(AISBenchConfigError) as exc_info:
            LEvalSciFiDataset.load()
        
        # Verify the error message contains helpful information
        assert "path" in str(exc_info.value).lower()
