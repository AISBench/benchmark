import csv
import json
from types import SimpleNamespace

import pytest

from herding.eval_datasets import dataset_base
from herding.eval_datasets.aime2025 import Aime2025Dataset
from herding.eval_datasets.gpqa import FILENAME as GPQA_FILENAME
from herding.eval_datasets.gpqa import GpqaDataset


class DummyDataset(dataset_base.EvalDatasetBase):
    def __init__(self, dataset_path, output_dir, size=3):
        super().__init__(dataset_path, output_dir)
        self.size = size

    def dataset_size(self):
        return self.size

    def dataset_prompts(self):
        return iter(())

    def save_data_by_indices(self, indices, outpath):
        return indices, outpath


def test_base_dataset_indices_round_trip(tmp_path):
    dataset = DummyDataset(tmp_path / "data", tmp_path / "output")
    strategy_dir = tmp_path / "output" / "existing"
    strategy_dir.mkdir(parents=True)

    dataset.save_indices([2, 0], strategy_dir)

    assert dataset.load_indices() == [0, 1, 2]
    assert dataset.load_indices_from_strategy("existing") == [2, 0]
    assert dataset.load_indices_from_strategy("missing") is None


def test_dataset_registry_constructs_registered_adapter(tmp_path):
    name = "unit_test_dataset"
    dataset_base.EVAL_DATASETS.pop(name, None)

    @dataset_base.reg_eval_dataset(name)
    class RegisteredDataset(DummyDataset):
        pass

    try:
        result = dataset_base.get_eval_dataset(name, tmp_path, tmp_path / "out")
        assert isinstance(result, RegisteredDataset)
    finally:
        dataset_base.EVAL_DATASETS.pop(name, None)


def test_dataset_registry_rejects_unknown_name(tmp_path):
    with pytest.raises(ValueError, match="Unknown dataset"):
        dataset_base.get_eval_dataset("not-registered", tmp_path, tmp_path / "out")


def test_aime2025_load_prompts_and_save_selected_rows(tmp_path):
    dataset_path = tmp_path / "source"
    dataset_path.mkdir()
    source_rows = [
        {"question": "What is 1 + 1?", "answer": "2"},
        {"question": "What is 2 + 2?", "answer": "4"},
    ]
    source_file = dataset_path / "aime2025.jsonl"
    source_file.write_text(
        "\n".join(json.dumps(row) for row in source_rows) + "\n\n",
        encoding="utf-8",
    )
    dataset = Aime2025Dataset(dataset_path, tmp_path / "result")

    prompts = list(dataset.dataset_prompts())
    output_dir = dataset.save_data_by_indices([1], "coreset")

    assert dataset.dataset_size() == 2
    assert "What is 1 + 1?" in prompts[0]
    saved_lines = (tmp_path / "result" / "coreset" / "aime2025.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert [json.loads(line) for line in saved_lines] == [source_rows[1]]
    assert json.loads(
        (tmp_path / "result" / "coreset" / "indices.json").read_text(
            encoding="utf-8"
        )
    ) == [1]
    assert output_dir == str(tmp_path / "result" / "coreset")


def test_gpqa_builds_prompts_and_preserves_csv_rows(tmp_path, monkeypatch):
    dataset_path = tmp_path / "source"
    dataset_path.mkdir()
    csv_path = dataset_path / GPQA_FILENAME
    rows = [
        ["question", "A", "B", "C", "D", "answer"],
        ["question one", "a1", "b1", "c1", "d1", "A"],
        ["question two", "a2", "b2", "c2", "d2", "B"],
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        csv.writer(output).writerows(rows)

    built_rows = [
        {"question": "question one", "A": "a1", "B": "b1", "C": "c1", "D": "d1"},
        {"question": "question two", "A": "a2", "B": "b2", "C": "c2", "D": "d2"},
    ]
    build_mock = lambda _cfg: SimpleNamespace(test=built_rows)
    monkeypatch.setattr(
        "herding.eval_datasets.gpqa.build_dataset_from_cfg",
        build_mock,
    )
    dataset = GpqaDataset(dataset_path, tmp_path / "result")

    prompts = list(dataset.dataset_prompts())
    output_dir = dataset.save_data_by_indices([1, 0], "coreset")

    assert dataset.dataset_size() == 2
    assert "question one" in prompts[0]
    assert "A) a1" in prompts[0]
    with (tmp_path / "result" / "coreset" / GPQA_FILENAME).open(
        newline="", encoding="utf-8"
    ) as saved_file:
        assert list(csv.reader(saved_file)) == [rows[0], rows[2], rows[1]]
    assert json.loads(
        (tmp_path / "result" / "coreset" / "indices.json").read_text(
            encoding="utf-8"
        )
    ) == [1, 0]
    assert output_dir == str(tmp_path / "result" / "coreset")
