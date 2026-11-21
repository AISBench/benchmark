import csv
import json

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


def test_dataset_reader_expect_valid_test_and_empty_train(data_file):
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
def test_invalid_data_expect_value_error(tmp_path, rows):
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
def test_parse_grade_expect_exact_label(text, grade):
    assert SimpleQAVerifiedEvaluator.parse_grade(text) == grade


def test_metrics_expect_official_f1():
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


def test_judge_merge_expect_matching_uuid_and_prompt(data_file):
    judge = object.__new__(SimpleQAVerifiedJudgeDataset)
    row = judge._process_single_item(
        [{"problem": "q", "answer": "gold"}],
        {"id": 0, "uuid": "uuid", "prediction": "candidate"},
    )
    assert row["model_pred_uuid"] == "uuid"
    assert "Gold target: gold" in row["judge_prompt"]
    assert "Predicted answer: candidate" in row["judge_prompt"]
    assert judge._get_dataset_class() is SimpleQAVerifiedDataset


def test_public_config_expect_constructible_components(data_file):
    from pathlib import Path

    from mmengine.config import Config

    from ais_bench.benchmark.utils.config.build import build_dataset_from_cfg

    path = (
        Path(__file__).resolve().parents[3]
        / "ais_bench/benchmark/configs/datasets/simpleqa_verified/simpleqa_verified_gen_0_shot_chat.py"
    )
    cfg = Config.fromfile(str(path))["simpleqa_verified_datasets"][0]
    assert cfg["type"] is SimpleQAVerifiedDataset
    assert cfg["path"] == "ais_bench/datasets/simpleqa_verified"
    cfg["path"] = str(data_file)
    dataset = build_dataset_from_cfg(cfg)
    assert dataset.test[0]["problem"] == "Question?"
    assert cfg["reader_cfg"]["input_columns"] == ["problem"]
    assert cfg["reader_cfg"]["output_column"] == "answer"
    for infer in (cfg["infer_cfg"], cfg["judge_infer_cfg"]):
        prompt_cfg = dict(infer["prompt_template"])
        prompt_type = prompt_cfg.pop("type")
        prompt_type(**prompt_cfg)
    evaluator_cfg = dict(cfg["eval_cfg"]["evaluator"])
    evaluator_type = evaluator_cfg.pop("type")
    assert evaluator_type(**evaluator_cfg).score(["A"], ["uuid"])["f1"] == 1
    assert cfg["judge_infer_cfg"]["judge_model"]["model"] == "gpt-4.1"


@pytest.mark.parametrize(
    "row",
    [
        None,
        [],
        3,
        {"original_index": True, "problem": "q", "answer": "a"},
        {"original_index": 1, "problem": 3, "answer": "a"},
        {"original_index": 1, "problem": "q", "answer": False},
    ],
)
def test_malformed_record_expect_value_error(tmp_path, row):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        SimpleQAVerifiedDataset.load(str(path))


def test_jsonl_metadata_expect_preserved(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps(
            {"original_index": 7, "problem": "q", "answer": "a", "topic": "science"}
        )
        + "\n",
        encoding="utf-8",
    )
    result = SimpleQAVerifiedDataset.load(str(path))["test"][0]
    assert result == {
        "original_index": "7",
        "problem": "q",
        "answer": "a",
        "topic": "science",
    }
