"""Compatibility and corruption coverage for optional Harbor reporting."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from ais_bench.benchmark.summarizers.harbor import HarborSummarizer
from ais_bench.benchmark.tasks.custom_tasks.harbor_task import HarborTask
from ais_bench.benchmark.utils.harbor_reporting import read_completed_report


def report_file(tmp_path, raw_hash=False):
    path = tmp_path / "normalized_result.json"
    report = {
        "schema_version": "1.0",
        "metrics": {"accuracy": 0.5},
        "integrity": {"status": "valid"},
        "acceptance": {"status": "blocked"},
        "label": "测试",
    }
    raw = json.dumps(report, ensure_ascii=True, separators=(",", ":")).encode()
    path.write_bytes(raw)
    payload = (
        raw
        if raw_hash
        else json.dumps(
            report, sort_keys=True, ensure_ascii=False, allow_nan=False
        ).encode()
    )
    key = "result_file_sha256" if raw_hash else "result_sha256"
    path.with_name("reporting_complete.json").write_text(
        json.dumps({key: hashlib.sha256(payload).hexdigest()})
    )
    return path


@pytest.mark.parametrize("raw_hash", [False, True])
def test_report_hash_protocols_and_corruption(tmp_path, raw_hash):
    path = report_file(tmp_path, raw_hash)
    assert read_completed_report(path)["metrics"]["accuracy"] == 0.5
    path.write_text(path.read_text().replace("0.5", "0.6"))
    with pytest.raises(ValueError, match="hash mismatch"):
        read_completed_report(path)


@pytest.mark.parametrize("reference", [None, 42, {}, "missing.json"])
def test_bad_reference_does_not_abort_summary(tmp_path, reference):
    result = tmp_path / "results" / "m" / "d.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({"avg_score": 0.75, "normalized_result": reference}))
    summarizer = HarborSummarizer(
        {"work_dir": str(tmp_path), "models": [], "datasets": []}
    )
    row = summarizer._build_row({"abbr": "m"}, "d")
    assert row["integrity"] == "invalid"
    assert row["accuracy"] is None
    assert row["avg_score"] == 0.75


def test_normalized_accuracy_preserves_reward_and_portable_evidence(tmp_path):
    path = report_file(tmp_path)
    result = tmp_path / "results" / "m" / "d.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps(
            {"avg_score": 0.75, "normalized_result": "../../normalized_result.json"}
        )
    )
    summarizer = HarborSummarizer(
        {"work_dir": str(tmp_path), "models": [], "datasets": []}
    )
    row = summarizer._build_row({"abbr": "m"}, "d")
    assert row["accuracy"] == 0.5
    assert row["avg_score"] == 0.75
    assert row["evidence"] == path.name


@pytest.mark.parametrize(
    "reporting",
    [
        [],
        {},
        {"publisher": "bad"},
        {"publisher": "missing.module:publish"},
        {"publisher": "fake:publish", "kwargs": []},
        {"publisher": "fake:publish"},
    ],
)
def test_optional_publisher_failure_preserves_legacy_results(
    tmp_path, monkeypatch, reporting
):
    import importlib

    monkeypatch.setattr(
        importlib,
        "import_module",
        Mock(
            return_value=SimpleNamespace(
                publish=Mock(side_effect=RuntimeError("failed"))
            )
        ),
    )
    task = HarborTask.__new__(HarborTask)
    task.dataset_cfgs = [{"abbr": "d", "reporting": reporting}]
    task.out_dir = str(tmp_path)
    task.logger = Mock()
    task._get_task_count = Mock(return_value=1)
    trial = SimpleNamespace(
        exception_info=None, verifier_result=SimpleNamespace(rewards={"reward": 1.0})
    )
    result = SimpleNamespace(trial_results=[trial], n_total_trials=1, stats=None)
    task._dump_eval_results(SimpleNamespace(config=None), result)
    published = json.loads((tmp_path / "d.json").read_text())
    assert published["avg_score"] == 1.0
    assert published["n_total_trials"] == 1
    assert "reporting_error" in published
    assert "normalized_result" not in published


def test_publisher_success_keeps_legacy_and_closes_before_replace(
    tmp_path, monkeypatch
):
    import importlib
    import os

    monkeypatch.setattr(
        importlib,
        "import_module",
        Mock(
            return_value=SimpleNamespace(
                publish=lambda *a, **kw: tmp_path / "summary" / "normalized_result.json"
            )
        ),
    )
    task = HarborTask.__new__(HarborTask)
    task.dataset_cfgs = [{"abbr": "d", "reporting": {"publisher": "fake:publish"}}]
    task.out_dir = str(tmp_path)
    task.logger = Mock()
    task._get_task_count = Mock(return_value=0)
    replace = os.replace

    def check_replace(source, target):
        # JSON must be flushed and readable before the atomic rename.
        assert json.loads(Path(source).read_text())["n_total_trials"] == 0
        replace(source, target)

    monkeypatch.setattr(os, "replace", check_replace)
    task._dump_eval_results(
        SimpleNamespace(config=None),
        SimpleNamespace(trial_results=[], n_total_trials=0, stats=None),
    )
    published = json.loads((tmp_path / "d.json").read_text())
    assert published["avg_score"] == 0.0
    assert published["normalized_result"] == str(
        Path("summary") / "normalized_result.json"
    )
    assert published["result_format"] == "harbor-normalized-index-v1"
    assert list(tmp_path.iterdir()) == [tmp_path / "d.json"]


def test_atomic_write_failure_removes_temporary_file(tmp_path, monkeypatch):
    import os

    task = HarborTask.__new__(HarborTask)
    task.dataset_cfgs = [{"abbr": "d"}]
    task.out_dir = str(tmp_path)
    task.logger = Mock()
    task._get_task_count = Mock(return_value=0)
    monkeypatch.setattr(os, "replace", Mock(side_effect=PermissionError("locked")))
    with pytest.raises(PermissionError):
        task._dump_eval_results(
            SimpleNamespace(config=None),
            SimpleNamespace(trial_results=[], n_total_trials=0, stats=None),
        )
    assert not list(tmp_path.iterdir())


def test_column_selection_depends_on_config_not_results(tmp_path):
    legacy = HarborSummarizer({"work_dir": str(tmp_path), "models": [], "datasets": []})
    normalized = HarborSummarizer(
        {
            "work_dir": str(tmp_path),
            "models": [],
            "datasets": [{"reporting": {"publisher": "x:y"}}],
        }
    )
    assert not legacy.normalized_columns
    assert normalized.normalized_columns


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "9.0"),
        ("metrics", {"accuracy": 2}),
        ("metrics", {"accuracy": True}),
        ("integrity", {"status": "unknown"}),
        ("acceptance", {"status": "unknown"}),
        ("integrity", {"status": "invalid"}),
    ],
)
def test_valid_hash_does_not_bypass_schema_validation(tmp_path, field, value):
    path = report_file(tmp_path, True)
    report = json.loads(path.read_bytes())
    report[field] = value
    path.write_text(json.dumps(report))
    path.with_name("reporting_complete.json").write_text(
        json.dumps(
            {"result_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    )
    with pytest.raises(ValueError):
        read_completed_report(path)


def test_csv_headers_stable_for_valid_and_corrupt_results(tmp_path):
    import csv

    path = report_file(tmp_path, True)
    index = tmp_path / "results" / "m" / "d.json"
    index.parent.mkdir(parents=True)
    index.write_text(json.dumps({"normalized_result": "../../normalized_result.json"}))
    summarizer = HarborSummarizer(
        {
            "work_dir": str(tmp_path),
            "models": [{"abbr": "m"}],
            "datasets": [{"abbr": "d", "reporting": {"publisher": "x:y"}}],
        }
    )
    summarizer.summarize("valid")
    path.unlink()
    summarizer.summarize("invalid")
    with (tmp_path / "summary" / "summary_valid.csv").open() as stream:
        valid = list(csv.reader(stream))
    with (tmp_path / "summary" / "summary_invalid.csv").open() as stream:
        invalid = list(csv.reader(stream))
    assert valid[0] == invalid[0] == summarizer.COLUMNS + summarizer.NORMALIZED_COLUMNS
    assert valid[1][valid[0].index("accuracy")] == "0.5"
    assert invalid[1][invalid[0].index("accuracy")] == ""
