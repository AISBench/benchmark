import csv
import json
from types import SimpleNamespace

import pytest

from ais_bench.benchmark.datasets.simpleqa_verified import (
    SimpleQAVerifiedDataset,
    SimpleQAVerifiedEvaluator,
    SimpleQAVerifiedJudgeDataset,
)


@pytest.fixture
def data_file(tmp_path):
    path = tmp_path / "simpleqa_verified.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["original_index", "problem", "answer"]
        )
        writer.writeheader()
        writer.writerow(
            {"original_index": "5", "problem": "Question?", "answer": "Target"}
        )
    return path


def test_dataset(data_file):
    dataset = SimpleQAVerifiedDataset(
        path=str(data_file),
        judge_infer_cfg={},
        reader_cfg={"input_columns": ["problem"], "output_column": "answer"},
    )
    assert len(dataset.test) == 1
    assert len(dataset.train) == 0
    assert dataset.test[0]["answer"] == "Target"


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{}],
        [{"original_index": "1", "problem": "", "answer": "a"}],
        [{"original_index": "1", "problem": "q", "answer": "a"}] * 2,
    ],
)
def test_bad_data(tmp_path, rows):
    path = tmp_path / "data.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError):
        SimpleQAVerifiedDataset.load(str(path))


@pytest.mark.parametrize(
    "text,grade",
    [
        ("A", "A"),
        ("B", "B"),
        ("C", "C"),
        ("CORRECT", "A"),
        ("INCORRECT", "B"),
        ("NOT_ATTEMPTED", "C"),
        ("not attempted", "C"),
        (" B\n", "B"),
        ("A or B", None),
        ("probably correct", None),
        ("", None),
        (None, None),
    ],
)
def test_parse(text, grade):
    assert SimpleQAVerifiedEvaluator.parse_grade(text) == grade


def test_metrics():
    result = SimpleQAVerifiedEvaluator().score(
        ["A", "B", "C", "A"], ["0", "1", "2", "3"]
    )
    assert result["accuracy"] == 0.5
    assert result["accuracy_given_attempted"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(4 / 7)
    assert result["incorrect"] == 0.25
    assert result["not_attempted"] == 0.25
    assert SimpleQAVerifiedEvaluator().score(["C"], ["0"])["f1"] == 0
    assert "error" in SimpleQAVerifiedEvaluator().score(["invalid"], ["0"])
    assert "error" in SimpleQAVerifiedEvaluator().score([], [])
    assert "error" in SimpleQAVerifiedEvaluator().score(["A"], [])


def test_judge_merge(data_file):
    judge = object.__new__(SimpleQAVerifiedJudgeDataset)
    row = judge._process_single_item(
        [{"problem": "q", "answer": "gold"}],
        {"id": 0, "uuid": "uuid", "prediction": "candidate"},
    )
    assert row["model_pred_uuid"] == "uuid"
    assert "Gold target: gold" in row["judge_prompt"]
    assert "Predicted answer: candidate" in row["judge_prompt"]
    assert judge._get_dataset_class() is SimpleQAVerifiedDataset


def test_config():
    from ais_bench.benchmark.cli.config_manager import ConfigManager

    cfg = ConfigManager(
        SimpleNamespace(datasets=["simpleqa_verified_gen_0_shot_chat"], config_dir="")
    )._load_datasets_config()[0]
    judge = cfg["judge_infer_cfg"]
    assert (
        judge["prompt_template"]["template"]["round"][0]["prompt"] == "{judge_prompt}"
    )
    assert judge["judge_model"]["model"] == "gpt-4.1"
