from unittest.mock import patch
from datasets import DatasetDict, Dataset
import pytest

from ais_bench.benchmark.datasets.leval.gsm100 import (
    LEvalGSM100Dataset,
    gsm100_postprocess,
    gsm100_dataset_postprocess,
)
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError


def test_gsm100_dataset_postprocess():
    # Test removing commas
    input_text = "1,234,567"
    result = gsm100_dataset_postprocess(input_text)
    assert result == "1234567"

    # Test no commas
    input_text = "1234567"
    result = gsm100_dataset_postprocess(input_text)
    assert result == "1234567"

    # Test empty string
    input_text = ""
    result = gsm100_dataset_postprocess(input_text)
    assert result == ""


def test_gsm100_postprocess_valid():
    # Test extracting number after "The answer is"
    input_text = "Some explanation. The answer is 42."
    result = gsm100_postprocess(input_text)
    assert result == "42"

    # Test with multiple numbers, takes first
    input_text = "The answer is 123 and then 456."
    result = gsm100_postprocess(input_text)
    assert result == "123"

    # Test number in word
    input_text = "The answer is forty-two, but 42 is the number."
    result = gsm100_postprocess(input_text)
    assert result == "42"


def test_gsm100_postprocess_no_answer():
    # Test no "The answer is"
    input_text = "Just some text without the phrase."
    result = gsm100_postprocess(input_text)
    assert result == ""


def test_gsm100_postprocess_no_number():
    # Test "The answer is" but no number
    input_text = "The answer is yes."
    result = gsm100_postprocess(input_text)
    assert result == ""


def test_gsm100_postprocess_empty_after_answer():
    # Test "The answer is" at end
    input_text = "The answer is"
    result = gsm100_postprocess(input_text)
    assert result == ""


@patch('ais_bench.benchmark.datasets.leval.gsm100.load_dataset')
@patch('ais_bench.benchmark.datasets.leval.gsm100.get_data_path')
def test_leval_gsm100_dataset_load(mock_get_data_path, mock_load_dataset):
    # Mock the path resolution
    mock_get_data_path.return_value = "/mock/path"

    # Mock the dataset
    mock_dataset = DatasetDict({
        'test': Dataset.from_list([
            {
                'instructions': ['Q1'],
                'outputs': ['A1'],
                'input': 'Context1'
            },
            {
                'instructions': ['Q2'],
                'outputs': ['A2'],
                'input': 'Context1'
            }
        ])
    })
    mock_load_dataset.return_value = mock_dataset

    # Call the load method
    result = LEvalGSM100Dataset.load(path="dummy_path")

    # Verify get_data_path was called
    mock_get_data_path.assert_called_once_with("dummy_path", local_mode=True)

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
    assert data_list[0]['question'] == 'Q1'
    assert data_list[0]['context'] == 'Context1'
    assert data_list[0]['answer'] == 'A1'

    # Second item
    assert data_list[1]['question'] == 'Q2'
    assert data_list[1]['context'] == 'Context1'
    assert data_list[1]['answer'] == 'A2'


@patch('ais_bench.benchmark.datasets.leval.gsm100.load_dataset')
@patch('ais_bench.benchmark.datasets.leval.gsm100.get_data_path')
def test_leval_gsm100_dataset_load_empty(
        mock_get_data_path, mock_load_dataset):
    # Mock empty dataset
    mock_get_data_path.return_value = "/mock/path"
    mock_dataset = DatasetDict({
        'test': Dataset.from_list([])
    })
    mock_load_dataset.return_value = mock_dataset

    result = LEvalGSM100Dataset.load(path="dummy_path")

    data_list = result['test'].to_list()
    assert len(data_list) == 0


def test_leval_gsm100_dataset_load_missing_path():
    """Test that ConfigError is raised when 'path' argument is missing."""
    with pytest.raises(AISBenchConfigError) as exc_info:
        LEvalGSM100Dataset.load()
    
    # Verify the error message contains helpful information
    assert "path" in str(exc_info.value).lower()
