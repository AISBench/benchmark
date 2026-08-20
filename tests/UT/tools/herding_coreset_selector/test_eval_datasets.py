import csv
import json
from types import SimpleNamespace

import pytest

from herding.eval_datasets import dataset_base
from herding.eval_datasets.aime2025 import Aime2025Dataset
from herding.eval_datasets.gpqa import FILENAME as GPQA_FILENAME
from herding.eval_datasets.gpqa import GpqaDataset


class DummyDataset(dataset_base.EvalDatasetBase):
    def dataset_size(self):
        return 3

    def dataset_prompts(self):
        return iter(())

    def save_data_by_indices(self, indices, outpath):
        return indices, outpath


def test_base_indices_and_registry(tmp_path):
    dataset = DummyDataset(tmp_path, tmp_path / "output")
    strategy_dir = tmp_path / "output" / "strategy"
    strategy_dir.mkdir(parents=True)
    dataset.save_indices([2, 0], strategy_dir)

    assert dataset.load_indices() == [0, 1, 2]
    assert dataset.load_indices_from_strategy("strategy") == [2, 0]
    assert dataset.load_indices_from_strategy("missing") is None

    name = "unit_test_dataset"
    dataset_base.reg_eval_dataset(name)(DummyDataset)
    try:
        assert isinstance(
            dataset_base.get_eval_dataset(name, tmp_path, tmp_path / "out"),
            DummyDataset,
        )
        with pytest.raises(ValueError, match="Unknown dataset"):
            dataset_base.get_eval_dataset("unknown", tmp_path, tmp_path)
    finally:
        dataset_base.EVAL_DATASETS.pop(name)


def test_aime2025_load_prompt_and_save(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rows = [
        {"question": "What is 1 + 1?", "answer": "2"},
        {"question": "What is 2 + 2?", "answer": "4"},
    ]
    (source / "aime2025.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n\n",
        encoding="utf-8",
    )
    dataset = Aime2025Dataset(source, tmp_path / "result")

    assert dataset.dataset_size() == 2
    assert "What is 1 + 1?" in next(dataset.dataset_prompts())
    output = dataset.save_data_by_indices([1], "coreset")
    saved = (tmp_path / "result" / "coreset" / "aime2025.jsonl").read_text()
    assert json.loads(saved) == rows[1]
    assert json.loads((tmp_path / "result" / "coreset" / "indices.json").read_text()) == [1]
    assert output == str(tmp_path / "result" / "coreset")


def test_gpqa_prompt_and_save(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    rows = [
        ["question", "A", "B", "C", "D", "answer"],
        ["q1", "a1", "b1", "c1", "d1", "A"],
        ["q2", "a2", "b2", "c2", "d2", "B"],
    ]
    with (source / GPQA_FILENAME).open("w", newline="") as output:
        csv.writer(output).writerows(rows)
    items = [dict(zip(rows[0][:-1], row[:-1])) for row in rows[1:]]
    monkeypatch.setattr(
        "herding.eval_datasets.gpqa.build_dataset_from_cfg",
        lambda _cfg: SimpleNamespace(test=items),
    )
    dataset = GpqaDataset(source, tmp_path / "result")

    assert "A) a1" in next(dataset.dataset_prompts())
    dataset.save_data_by_indices([1, 0], "coreset")
    with (tmp_path / "result" / "coreset" / GPQA_FILENAME).open() as saved:
        assert list(csv.reader(saved)) == [rows[0], rows[2], rows[1]]
    indices = tmp_path / "result" / "coreset" / "indices.json"
    assert json.loads(indices.read_text()) == [1, 0]
