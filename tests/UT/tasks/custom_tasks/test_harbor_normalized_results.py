"""Normalized export regression tests using the imported HarborTask class."""

import hashlib
import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def consumer():
    from ais_bench.benchmark.tasks.custom_tasks.harbor_task import HarborTask

    return HarborTask._dump_eval_results


def export(
    consumer,
    root,
    signal,
    blocked=False,
    *,
    normalized=None,
    fault=None,
    rewards=None,
    trials=1,
    configured=False,
):
    summary = root / "summary"
    summary.mkdir()
    data = (
        normalized
        if normalized is not None
        else {
            "schema_version": "1.0",
            "metrics": {"accuracy": None if blocked else signal},
            "acceptance": {
                "status": "blocked" if blocked else "pass" if signal else "fail"
            },
            "integrity": {"status": "invalid" if blocked else "valid"},
        }
    )
    path = summary / "normalized_result.json"
    path.write_text(json.dumps(data))
    marker = summary / "reporting_complete.json"
    marker.write_text(
        json.dumps(
            {
                "result_sha256": hashlib.sha256(
                    json.dumps(
                        data, sort_keys=True, ensure_ascii=False, allow_nan=False
                    ).encode()
                ).hexdigest()
            }
        )
    )
    if fault == "missing":
        path.unlink()
    elif fault == "marker":
        marker.unlink()
    elif fault == "hash":
        path.write_text(path.read_text() + " ")
        marker.write_text('{"result_sha256":"wrong"}')
    elif fault == "json":
        path.write_text("{")
    task = SimpleNamespace(
        dataset_cfgs=[
            {
                "abbr": "fixture",
                "args": {"normalized_result_path": str(path)} if configured else {},
            }
        ],
        out_dir=str(root),
        _get_task_count=lambda config: 1,
        logger=SimpleNamespace(info=lambda *args: None, error=lambda *args: None),
    )
    result = SimpleNamespace(
        trial_results=[
            SimpleNamespace(
                exception_info=None,
                verifier_result=SimpleNamespace(
                    rewards=rewards if rewards is not None else {"accuracy": signal}
                ),
            )
        ],
        n_total_trials=trials,
        stats=None,
    )
    consumer(task, SimpleNamespace(config=None, job_dir=root), result)
    return json.loads((root / "fixture.json").read_text())


@pytest.mark.parametrize("signal", [0.0, 1.0])
def test_tc013_strict_accuracy_is_preserved(consumer, tmp_path, signal):
    result = export(consumer, tmp_path, signal)
    assert result["avg_score"] == signal


def test_tc013_invalid_normalized_result_remains_blocked(consumer, tmp_path):
    result = export(consumer, tmp_path, 0.0, blocked=True)
    assert result.get("acceptance", {}).get("status") == "blocked"
    assert "accuracy" in result and result["accuracy"] is None


@pytest.mark.parametrize("fault", ["missing", "marker", "hash", "json"])
def test_tc013_incomplete_publication_blocks_instead_of_zero(consumer, tmp_path, fault):
    result = export(consumer, tmp_path, 1.0, fault=fault)
    assert result["avg_score"] is None
    assert result["acceptance"]["status"] == "blocked"
    assert result["integrity"]["status"] == "invalid"


@pytest.mark.parametrize(
    "status,score", [("blocked", 0.625), ("fail", 0.125), ("pass", 0.6033333333333334)]
)
def test_tc013_normalized_is_authoritative_and_not_rounded(
    consumer, tmp_path, status, score
):
    data = {
        "schema_version": "1.0",
        "metrics": {
            "accuracy": score,
            "partial_credit_mean": 0.9,
            "per_run_accuracy": [0.6, 0.59, 0.62],
        },
        "integrity": {"status": "valid"},
        "acceptance": {"status": status, "reasons": ["fixture"]},
    }
    result = export(
        consumer, tmp_path, 0.0, normalized=data, trials=99, configured=True
    )
    assert result["avg_score"] == score
    assert result["metrics"] == data["metrics"]
    assert result["acceptance"] == data["acceptance"]


@pytest.mark.parametrize("score", [True, -0.01, 1.01, "1"])
def test_tc013_invalid_accuracy_blocks(consumer, tmp_path, score):
    result = export(consumer, tmp_path, score)
    assert result["avg_score"] is None
    assert result["acceptance"]["status"] == "blocked"


def test_tc013_legacy_reward_remains_compatible(consumer, tmp_path):
    result = export(
        consumer,
        tmp_path,
        0.0,
        fault="missing",
        rewards={"reward": 0.75, "f2p_total": 99},
    )
    assert result["avg_score"] == 0.75
    assert "acceptance" not in result


def test_tc013_repeat_export_preserves_normalized_result(consumer, tmp_path):
    first = export(consumer, tmp_path, 1.0)
    (tmp_path / "fixture.json").unlink()
    # A resumed export consumes the same published artifact, not trial statistics.
    task = SimpleNamespace(
        dataset_cfgs=[{"abbr": "fixture"}],
        out_dir=str(tmp_path),
        logger=SimpleNamespace(info=lambda *a: None),
    )
    consumer(
        task,
        SimpleNamespace(job_dir=tmp_path),
        SimpleNamespace(trial_results=[], n_total_trials=1),
    )
    assert json.loads((tmp_path / "fixture.json").read_text()) == first


@pytest.mark.parametrize("trials", [1, 6])
def test_export_then_generic_display_and_csv(consumer, tmp_path, trials):
    from ais_bench.benchmark.summarizers.harbor import HarborSummarizer

    output = tmp_path / "results" / "model"
    output.mkdir(parents=True)
    result = export(consumer, output, 0.0, blocked=True, trials=trials)
    summarizer = HarborSummarizer(
        {
            "work_dir": str(tmp_path),
            "models": [{"abbr": "model"}],
            "datasets": [{"abbr": "fixture"}],
        }
    )
    row = summarizer._build_row({"abbr": "model"}, "fixture")
    assert row["avg_score"] is None
    assert row["correct"] == row["wrong"] == "-"
    assert row["acceptance"] == result["acceptance"]["status"] == "blocked"
    summarizer.summarize(time_str="fixture")
    import csv

    with (tmp_path / "summary/summary_fixture.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["avg_score"] == ""
    assert rows[0]["acceptance"] == "blocked"
