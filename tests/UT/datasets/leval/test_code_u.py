"""Unit test for LEval Code U dataset loader."""

import pytest
from unittest.mock import patch
from datasets import Dataset, DatasetDict

from ais_bench.benchmark.datasets.leval.code_u import LEvalCodeUDataset
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError


class TestLEvalCodeUDataset:
    @patch('ais_bench.benchmark.datasets.leval.code_u.get_data_path')
    @patch('ais_bench.benchmark.datasets.leval.code_u.load_dataset')
    def test_code_u_load(self, mock_load_dataset, mock_get_data_path):
        # Mock path resolution
        mock_get_data_path.return_value = "/mock/code_u.json"

        # Mock upstream dataset structure (nested instructions/outputs)
        mock_dataset = DatasetDict({
            'test': Dataset.from_list([
                {
                    'instructions': ['What does this function return?'],
                    'outputs': ['It returns the sum of the inputs'],
                    'input': 'def add(a, b):\n    return a + b'
                },
                {
                    'instructions': ['Identify side effects'],
                    'outputs': ['No side effects; function is pure'],
                    'input': 'def square(x):\n    return x * x'
                }
            ])
        })
        mock_load_dataset.return_value = mock_dataset

        # Invoke loader
        result = LEvalCodeUDataset.load(path="dummy_path")

        # Assertions: path resolution & dataset loading
        mock_get_data_path.assert_called_once_with("dummy_path", local_mode=True)
        mock_load_dataset.assert_called_once_with('json', data_files={'test': '/mock/code_u.json'})

        # Check structure
        assert 'test' in result
        flattened = result['test']
        assert isinstance(flattened, Dataset)

        rows = flattened.to_list()
        assert len(rows) == 2

        # Row 1
        assert rows[0]['question'] == 'What does this function return?'
        assert 'def add(a, b):' in rows[0]['context']
        assert rows[0]['answer'] == 'It returns the sum of the inputs'

        # Row 2
        assert rows[1]['question'] == 'Identify side effects'
        assert 'def square(x):' in rows[1]['context']
        assert rows[1]['answer'] == 'No side effects; function is pure'

        # Ensure no length field (Code U loader does not add it)
        assert 'length' not in rows[0]
        assert 'length' not in rows[1]

    def test_code_u_load_missing_path(self):
        """Test that ConfigError is raised when 'path' argument is missing."""
        with pytest.raises(AISBenchConfigError) as exc_info:
            LEvalCodeUDataset.load()
        
        # Verify the error message contains helpful information
        assert "path" in str(exc_info.value).lower()
