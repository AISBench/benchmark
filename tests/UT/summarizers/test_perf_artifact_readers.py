"""Regression tests for latency derivation and the batch artifact readers."""

import json
import sqlite3
from contextlib import closing
from unittest.mock import Mock

import numpy as np
import pytest

from ais_bench.benchmark.openicl.icl_inferencer.output_handler import db_utils
from ais_bench.benchmark.summarizers import perf_batch
from ais_bench.benchmark.utils.perf_details import derive_request_metrics


@pytest.mark.parametrize(
    "tokens",
    [None, False, True, -1, 0, 1, np.int64(1)],
    ids=["missing", "false", "true", "negative", "zero", "single", "numpy-single"],
)
def test_missing_tpot_does_not_discard_observed_token_gaps(tokens):
    metrics = derive_request_metrics([100.0, 100.25, 100.75, 101.0], tokens)

    assert metrics["latency"] == 1.0
    assert metrics["ttft"] == 0.25
    assert metrics["tpot"] is None
    np.testing.assert_array_equal(metrics["itl"], [0.5, 0.25])


def test_two_integer_points_preserve_dtype_and_have_no_observed_token_gaps():
    points = np.array([16, 23], dtype=np.int32)
    metrics = derive_request_metrics(points, np.int64(4))

    assert metrics["latency"] == metrics["ttft"] == 7
    assert isinstance(metrics["latency"], np.int32)
    assert isinstance(metrics["ttft"], np.int32)
    assert metrics["tpot"] == 0.0
    assert metrics["itl"].shape == (0,)
    np.testing.assert_array_equal(points, [16, 23])


def test_tpot_uses_service_token_count_not_number_of_recorded_chunks():
    points = np.array([200, 201, 201, 205], dtype=np.int64)
    metrics = derive_request_metrics(points, 9)

    assert metrics["latency"] == 5
    assert metrics["ttft"] == 1
    assert metrics["tpot"] == 0.5
    np.testing.assert_array_equal(metrics["itl"], [0, 4])
    assert metrics["itl"].dtype == points.dtype


def _write_details(directory, entries):
    path = directory / "synthetic_details.jsonl"
    path.write_text(
        "\n".join(entry if isinstance(entry, str) else json.dumps(entry) for entry in entries),
        encoding="utf-8",
    )
    return path


def _store_points(path, points):
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(db_utils.init_db(str(path))) as conn:
        key = db_utils.save_numpy_to_db(conn, np.array(points))
        conn.commit()
    return key


@pytest.mark.parametrize("streaming", [True, False], ids=["streaming", "non-streaming"])
def test_reader_resolves_references_in_the_correct_database_and_keeps_units(tmp_path, streaming):
    # Both databases allocate the same row ID. They must not share a lookup.
    a = _store_points(tmp_path / "db_data" / "a.db", [[10, 10.125], [10.25, 10.5]])
    b = _store_points(tmp_path / "tmp" / "tmp_b.db", [20, 20.25, 21])
    assert a == b
    _write_details(tmp_path, [
        {"db_name": "a.db", "time_points": {"__db_ref__": a},
         "input_tokens": 8, "output_tokens": 4},
        {"db_name": "b.db", "time_points": {"__db_ref__": b},
         "input_tokens": 9, "output_tokens": 5},
    ])

    rows = perf_batch._load_detail_rows(tmp_path, "synthetic", ttft_valid=streaming)

    assert rows == [
        {"success": True, "latency": 500.0, "input_len": 8, "output_len": 4,
         "ttft": 125.0 if streaming else None, "tpot": 125.0 if streaming else None,
         "itl": 187.5 if streaming else None},
        {"success": True, "latency": 1000.0, "input_len": 9, "output_len": 5,
         "ttft": 250.0 if streaming else None, "tpot": 187.5 if streaming else None,
         "itl": 750.0 if streaming else None},
    ]


def test_detail_reader_ignores_broken_json_but_keeps_failed_requests(tmp_path):
    key = _store_points(tmp_path / "db_data" / "requests.db", [128, 128.5])
    _write_details(tmp_path, [
        "  ",
        '{"db_name":"requests.db",',
        {"input_tokens": 999, "time_points": [0, 1]},
        {"db_name": "requests.db", "input_tokens": 2, "success": False,
         "time_points": "must not be parsed for failures", "output_tokens": 99},
        {"db_name": "requests.db", "input_tokens": 3},
        {"db_name": "requests.db", "input_tokens": 5, "time_points": [128]},
        {"db_name": "requests.db", "input_tokens": 7, "time_points": []},
        {"db_name": "requests.db", "input_tokens": 11,
         "time_points": {"__db_ref__": key + 1}},
        {"db_name": "requests.db", "input_tokens": 13,
         "time_points": {"__db_ref__": key}, "output_tokens": 1},
    ])

    rows = perf_batch._load_detail_rows(tmp_path, "synthetic", ttft_valid=True)

    assert rows[:-1] == [
        {"success": False, "latency": None, "ttft": None, "tpot": None,
         "itl": None, "input_len": count, "output_len": None}
        for count in [2, 3, 5, 7, 11]
    ]
    assert rows[-1] == {
        "success": True, "latency": 500.0, "ttft": 500.0, "tpot": None,
        "itl": None, "input_len": 13, "output_len": 1,
    }


def test_inline_details_survive_missing_database_and_missing_output_count(tmp_path):
    _write_details(tmp_path, [
        {"db_name": "absent.db", "time_points": {"__db_ref__": 1}},
        {"db_name": "absent.db", "time_points": [32, 32.125, 32.375]},
    ])

    unresolved, inline = perf_batch._load_detail_rows(tmp_path, "synthetic", ttft_valid=True)

    assert not unresolved["success"]
    assert inline == {
        "success": True, "latency": 375.0, "ttft": 125.0, "tpot": None,
        "itl": 250.0, "input_len": None, "output_len": None,
    }
    assert not (tmp_path / "db_data").exists()
    assert not (tmp_path / "tmp").exists()


def test_detail_reader_returns_empty_if_jsonl_cannot_be_opened(tmp_path, monkeypatch):
    path = tmp_path / "synthetic_details.jsonl"
    path.mkdir()  # A directory at the expected filename fails on Windows and POSIX.
    warning = Mock()
    resolver = Mock()
    monkeypatch.setattr(perf_batch.logger, "warning", warning)
    monkeypatch.setattr(perf_batch, "_resolve_db_data", resolver)

    assert perf_batch._load_detail_rows(tmp_path, "synthetic", ttft_valid=True) == []

    warning.assert_called_once()
    assert str(path) in warning.call_args.args[0]
    resolver.assert_not_called()


def test_db_reader_prefers_finalized_db_over_temporary_copy(tmp_path):
    final_id = _store_points(tmp_path / "db_data" / "worker.db", [0, 1])
    _store_points(tmp_path / "tmp" / "tmp_worker.db", [3, 5])

    data = perf_batch._resolve_db_data(tmp_path, "worker.db")

    assert data.keys() == {final_id}
    np.testing.assert_array_equal(data[final_id], [0, 1])


def test_db_reader_falls_back_to_temporary_copy_if_finalized_db_is_corrupt(tmp_path, monkeypatch):
    broken = tmp_path / "db_data" / "worker.db"
    broken.parent.mkdir()
    broken.write_bytes(b"interrupted SQLite file")
    key = _store_points(tmp_path / "tmp" / "tmp_worker.db", [1, 3, 7])
    warning = Mock()
    monkeypatch.setattr(perf_batch.logger, "warning", warning)

    data = perf_batch._resolve_db_data(tmp_path, "worker.db")

    assert data.keys() == {key}
    np.testing.assert_array_equal(data[key], [1, 3, 7])
    warning.assert_called_once()
    assert str(broken) in warning.call_args.args[0]
    assert broken.read_bytes() == b"interrupted SQLite file"


def test_db_reader_closes_connections_even_if_numpy_select_fails(tmp_path, monkeypatch):
    final = tmp_path / "db_data" / "worker.db"
    temp = tmp_path / "tmp" / "tmp_worker.db"
    for candidate in [final, temp]:
        candidate.parent.mkdir()
        candidate.touch()
    first, second = Mock(), Mock()
    open_db = Mock(side_effect=[first, second])
    load = Mock(side_effect=sqlite3.OperationalError("numpy_store unreadable"))
    monkeypatch.setattr(db_utils, "init_db", open_db)
    monkeypatch.setattr(db_utils, "load_all_numpy_from_db", load)

    assert perf_batch._resolve_db_data(tmp_path, "worker.db") == {}

    assert [call.args for call in open_db.call_args_list] == [(str(final),), (str(temp),)]
    assert [call.args for call in load.call_args_list] == [(first,), (second,)]
    first.close.assert_called_once_with()
    second.close.assert_called_once_with()


@pytest.mark.parametrize(
    ("payload", "warns"),
    [(None, False), ("[]", False), ("null", False), ('{"E2EL":', True)],
    ids=["missing", "not-a-map", "null", "truncated"],
)
def test_common_results_reader_handles_absent_or_unusable_artifacts(tmp_path, monkeypatch, payload, warns):
    path = tmp_path / "synthetic.json"
    if payload is not None:
        path.write_text(payload, encoding="utf-8")
    warning = Mock()
    monkeypatch.setattr(perf_batch.logger, "warning", warning)

    assert perf_batch._load_perf_json(path) is None

    assert warning.called is warns
    if warns:
        assert str(path) in warning.call_args.args[0]


@pytest.mark.parametrize("state", ["missing", "empty", "header-only", "unreadable"])
def test_metric_csv_reader_returns_no_stats_without_rows(tmp_path, monkeypatch, state):
    path = tmp_path / "metrics.csv"
    if state in {"empty", "header-only"}:
        path.write_text("" if state == "empty" else "Param,Stage,Average,P99\n", encoding="utf-8")
    elif state == "unreadable":
        path.mkdir()
    warning = Mock()
    monkeypatch.setattr(perf_batch.logger, "warning", warning)

    assert perf_batch._load_perf_csv(path) == {}

    assert warning.called is (state == "unreadable")
    if warning.called:
        assert str(path) in warning.call_args.args[0]


def test_metric_csv_reader_parses_partial_rows_and_respects_explicit_stage(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text(
        "Param,Stage,Average,P90,P99\n"
        "broken-row\n"
        "E2EL,warmup,2.5e2 ms,NA,\n"
        "E2EL,total,125 ms,140 ms,199 ms\n"
        "TTFT,stable,0.25 ms\n"
        "ITL,stable,,5,10,unexpected field\n",
        encoding="utf-8",
    )

    metrics = perf_batch._load_perf_csv(path)

    assert metrics == {
        ("E2EL", "warmup"): {"Average": 250.0, "P90": None, "P99": None},
        ("E2EL", "total"): {"Average": 125.0, "P90": 140.0, "P99": 199.0},
        ("TTFT", "stable"): {"Average": 0.25, "P90": None, "P99": None},
        ("ITL", "stable"): {"Average": None, "P90": 5.0, "P99": 10.0},
    }
    assert perf_batch._csv_stat(metrics, "E2EL", "Average") == 125.0
    assert perf_batch._csv_stat(metrics, "E2EL", "Average", stage="warmup") == 250.0
    assert perf_batch._csv_stat(metrics, "TTFT", "Average") == 0.25
    assert perf_batch._csv_stat(metrics, "TTFT", "Average", stage="total") is None
    assert perf_batch._csv_stat(metrics, "TTFT", "P99", stage="stable") is None
    assert perf_batch._csv_stat(metrics, "Absent", "Average") is None
