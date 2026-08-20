import argparse
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import herding
from herding import __main__ as cli
from herding import algorithm, features
from herding import eval_datasets


@pytest.mark.parametrize("value", ["0.1", "1", "1.0"])
def test_coreset_ratio_accepts_valid_values(value):
    assert cli._coreset_ratio(value) == float(value)


@pytest.mark.parametrize("value", ["0", "-0.1", "1.01", "invalid"])
def test_coreset_ratio_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        cli._coreset_ratio(value)


def test_model_name_supports_trailing_path_separator():
    assert cli._model_name("/models/Qwen2.5-7B-Instruct/") == "Qwen2.5-7B-Instruct"
    assert cli._model_name("C:\\models\\example\\") == "example"


def test_model_name_rejects_path_without_a_name():
    with pytest.raises(ValueError, match="Unable to infer model name"):
        cli._model_name("/")


def test_parse_args_uses_defaults(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "herding",
            "--eval-dataset",
            "aime2025",
            "--dataset-path",
            "/datasets/aime2025",
            "--model-path",
            "/models/example",
        ],
    )

    args = cli.parse_args()

    assert args.eval_dataset == "aime2025"
    assert args.coreset_ratio == cli.DEFAULT_CORESET_RATIO
    assert args.output_dir == cli.DEFAULT_OUTPUT_DIR


class FakeEvalDataset:
    def __init__(self, indices):
        self.indices = indices
        self.saved = []

    def load_indices(self):
        return self.indices

    def save_data_by_indices(self, indices, outpath):
        self.saved.append((list(indices), outpath))
        return f"/saved/{outpath}"


def test_main_generates_and_saves_coreset(tmp_path, monkeypatch, capsys):
    args = SimpleNamespace(
        eval_dataset="gpqa",
        dataset_path="/datasets/gpqa",
        model_path="/models/example/",
        coreset_ratio=0.4,
        output_dir=str(tmp_path),
    )
    dataset = FakeEvalDataset([0, 1, 2, 3, 4])
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
    assert dataset.saved == [([0, 1, 2, 3, 4], "origin"), ([3, 1], "coreset")]
    generate.assert_called_once_with(
        2,
        eval_dataset=dataset,
        model_path="/models/example/",
    )
    assert "output: /saved/coreset" in capsys.readouterr().out


def test_main_rejects_empty_dataset(tmp_path, monkeypatch):
    args = SimpleNamespace(
        eval_dataset="gpqa",
        dataset_path="/datasets/gpqa",
        model_path="/models/example",
        coreset_ratio=0.2,
        output_dir=str(tmp_path),
    )
    dataset = FakeEvalDataset([])
    monkeypatch.setattr(cli, "parse_args", lambda: args)
    monkeypatch.setattr(eval_datasets, "get_eval_dataset", Mock(return_value=dataset))

    with pytest.raises(ValueError, match="dataset is empty"):
        cli.main()


def test_generate_coreset_orchestrates_feature_extraction(monkeypatch, capsys):
    dataset = Mock()
    dataset.dataset_prompts.return_value = iter(["prompt one", "prompt two"])
    dataset.dataset_size.return_value = 2
    model = object()
    tokenizer = object()
    feature_tensors = [torch.tensor([1.0]), torch.tensor([2.0])]
    load_model = Mock(return_value=(model, tokenizer))
    generate_logits = Mock(return_value=iter(feature_tensors))
    to_matrix = Mock(return_value=np.array([[1.0], [2.0]]))
    select = Mock(return_value=[1])
    monkeypatch.setattr(features, "load_model", load_model)
    monkeypatch.setattr(features, "generate_logits", generate_logits)
    monkeypatch.setattr(algorithm, "features_to_coreset_matrix", to_matrix)
    monkeypatch.setattr(algorithm, "coreset_indices", select)

    result = herding.generate_coreset(1, dataset, "/models/example")

    assert result == [1]
    load_model.assert_called_once_with("/models/example")
    generate_logits.assert_called_once()
    passed_model, passed_tokenizer, passed_prompts = generate_logits.call_args.args
    assert passed_model is model
    assert passed_tokenizer is tokenizer
    assert list(passed_prompts) == ["prompt one", "prompt two"]
    to_matrix.assert_called_once()
    select.assert_called_once()
    np.testing.assert_array_equal(select.call_args.args[0], np.array([[1.0], [2.0]]))
    assert select.call_args.args[1] == 1
    assert "herding:" in capsys.readouterr().out
