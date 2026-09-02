import argparse
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import herding
from herding import __main__ as cli
from herding import algorithm, eval_datasets, features


@pytest.mark.parametrize("value", ["0.1", "1", "1.0"])
def test_coreset_ratio_accepts_valid_values(value):
    assert cli._coreset_ratio(value) == float(value)


@pytest.mark.parametrize("value", ["0", "-0.1", "1.01", "bad"])
def test_coreset_ratio_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        cli._coreset_ratio(value)


def test_model_name_and_parse_defaults(monkeypatch):
    assert cli._model_name("/models/example/") == "example"
    with pytest.raises(ValueError, match="Unable to infer"):
        cli._model_name("/")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "herding",
            "--eval-dataset", "aime2025",
            "--dataset-path", "/datasets/aime2025",
            "--model-path", "/models/example",
        ],
    )
    args = cli.parse_args()
    assert args.coreset_ratio == cli.DEFAULT_CORESET_RATIO
    assert args.output_dir == cli.DEFAULT_OUTPUT_DIR


def test_main_generates_and_saves_coreset(tmp_path, monkeypatch, capsys):
    args = SimpleNamespace(
        eval_dataset="gpqa",
        dataset_path="/datasets/gpqa",
        model_path="/models/example",
        coreset_ratio=0.4,
        output_dir=str(tmp_path),
    )
    dataset = Mock()
    dataset.load_indices.return_value = [0, 1, 2, 3, 4]
    dataset.save_data_by_indices.side_effect = ["/saved/origin", "/saved/coreset"]
    get_dataset = Mock(return_value=dataset)
    generate = Mock(return_value=[3, 1])
    monkeypatch.setattr(cli, "parse_args", lambda: args)
    monkeypatch.setattr(eval_datasets, "get_eval_dataset", get_dataset)
    monkeypatch.setattr(herding, "generate_coreset", generate)

    cli.main()

    get_dataset.assert_called_once_with(
        "gpqa",
        dataset_path="/datasets/gpqa",
        output_dir=str(tmp_path / "gpqa" / "herding" / "example"),
    )
    assert dataset.save_data_by_indices.call_args_list[0].args == (
        [0, 1, 2, 3, 4], "origin"
    )
    generate.assert_called_once_with(2, eval_dataset=dataset, model_path="/models/example")
    assert dataset.save_data_by_indices.call_args_list[1].args == ([3, 1], "coreset")
    assert "output: /saved/coreset" in capsys.readouterr().out


def test_generate_coreset_pipeline(monkeypatch):
    dataset = Mock()
    dataset.dataset_prompts.return_value = iter(["one", "two"])
    dataset.dataset_size.return_value = 2
    load = Mock(return_value=("model", "tokenizer"))
    generate = Mock(return_value=iter(["feature-1", "feature-2"]))
    matrix = np.array([[1.0], [2.0]])
    to_matrix = Mock(return_value=matrix)
    select = Mock(return_value=[1])
    monkeypatch.setattr(features, "load_model", load)
    monkeypatch.setattr(features, "generate_logits", generate)
    monkeypatch.setattr(algorithm, "features_to_coreset_matrix", to_matrix)
    monkeypatch.setattr(algorithm, "coreset_indices", select)

    assert herding.generate_coreset(1, dataset, "/model") == [1]
    load.assert_called_once_with("/model")
    generate.assert_called_once()
    to_matrix.assert_called_once()
    np.testing.assert_array_equal(select.call_args.args[0], matrix)
    assert select.call_args.args[1] == 1
