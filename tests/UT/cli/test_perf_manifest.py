"""Tests for the on-disk sweep manifest shared by the CLI and batch report."""

import csv
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ais_bench.benchmark.cli import config_manager
from ais_bench.benchmark.cli.config_manager import ConfigManager
from ais_bench.benchmark.summarizers import perf_batch


@pytest.fixture
def manager(tmp_path):
    # Bypass unrelated CLI argument parsing and bootstrap, not sweep expansion.
    manager = ConfigManager.__new__(ConfigManager)
    manager.args = SimpleNamespace(mode="perf")
    manager.cfg = {"work_dir": str(tmp_path / "run")}
    manager.logger = Mock()
    return manager


def _manifest_path(manager):
    # Use the public filename literally so a producer/consumer rename fails.
    return Path(manager.cfg["work_dir"]) / "combos_manifest.json"


def test_sweep_manifest_round_trips_into_batch_summary(manager, monkeypatch, tmp_path):
    tokenizer_path = tmp_path / "private-tokenizer-directory"
    tokenizer_path.mkdir()
    manager.cfg.update(
        models=[{
            "type": "ais_bench.benchmark.models.VLLMCustomAPIChat",
            "abbr": "chat",
            "attr": "service",
            "path": str(tokenizer_path),
            "api_key": "private-credential",
            "url": "https://private-endpoint.invalid",
            "stream": True,
            "batch_size": 1,
            "max_out_len": 48,
            "generation_kwargs": {},
        }],
        datasets=[{
            "type": "ais_bench.benchmark.datasets.synthetic.SyntheticDataset",
            "abbr": "synthetic",
            "reader_cfg": {},
            "infer_cfg": {"retriever": {}},
            "config": {
                "Type": "tokenid",
                "RequestCount": 16,
                "TokenIdConfig": {"RequestSize": 64, "PrefixLen": 0},
            },
        }],
        cli_args={
            "mode": "perf",
            "batch_size": [8, 2],
            "num_prompts": [16, 4],
            "num_warmups": 0,
            "use_timestamp": False,
            "tokenid_config_prefix_len": 0,
            "api_key": "private-cli-credential",
            "url": "https://private-cli-endpoint.invalid",
            "path": "private-cli-path",
        },
    )
    manager._expand_sweep_configs()
    path = _manifest_path(manager)
    payload = path.read_text(encoding="utf-8")
    manifest = json.loads(payload)
    assert manifest["schema_version"] == 1
    assert manifest["ais_bench_version"]
    assert datetime.fromisoformat(manifest["created_at"]).microsecond == 0
    assert manifest["cli_args"] == {
        "batch_size": [8, 2],
        "num_prompts": [16, 4],
        "num_warmups": 0,
        "use_timestamp": False,
        "prefix_len": 0,
    }
    assert "private-" not in payload
    assert [
        (c["model_abbr"], c["dataset_abbr"], c["batch_size"], c["num_prompts"])
        for c in manifest["combos"]
    ] == [("chat-bs8", "synthetic", 8, 16), ("chat-bs2", "synthetic", 2, 4)]
    for combo in manifest["combos"]:
        assert combo["model_config"]["abbr"] == "chat"
        assert combo["model_config"]["max_out_len"] == 48
        assert combo["dataset_config"] == {
            "Type": "tokenid", "RequestSize": 64, "PrefixLen": 0,
        }
        folder = path.parent / "performances" / combo["model_abbr"]
        folder.mkdir(parents=True)
        (folder / "synthetic.json").write_text(
            json.dumps({"Output Token Throughput": {"total": combo["batch_size"] * 10}}),
            encoding="utf-8",
        )

    renderer = Mock(return_value=None)
    monkeypatch.setattr(perf_batch, "render_batch_perf_report", renderer)
    result = perf_batch.aggregate_batch_perf(path.parent)
    assert result.html_path is None
    with result.csv_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))
    assert rows[0][:3] == ["Model Abbr", "Dataset Abbr", "Batch Size"]
    separator = rows.index([])
    assert [row[:5] for row in rows[1:separator]] == [
        ["chat-bs2", "synthetic", "2", "4", "success"],
        ["chat-bs8", "synthetic", "8", "16", "success"],
    ]
    numeric_rows, report_manifest, _, work_dir = renderer.call_args.args
    assert [row["throughput_tok_s"] for row in numeric_rows] == [20, 80]
    assert report_manifest == manifest
    assert work_dir == path.parent
    assert path.read_text(encoding="utf-8") == payload


def test_writer_replaces_existing_manifest_only_after_fsync(manager, monkeypatch):
    path = _manifest_path(manager)
    before = {"schema_version": 1, "warnings": ["old"]}
    after = {"schema_version": 1, "warnings": ["测试完成"]}
    manager._write_manifest(before)
    real_fsync = config_manager.os.fsync
    real_replace = config_manager.os.replace
    events = []

    def fsync(fd):
        events.append("fsync")
        return real_fsync(fd)

    def replace(source, target):
        assert Path(source).parent == path.parent
        assert Path(target) == path
        assert json.loads(path.read_text(encoding="utf-8")) == before
        assert json.loads(Path(source).read_text(encoding="utf-8")) == after
        events.append("replace")
        return real_replace(source, target)

    monkeypatch.setattr(config_manager.os, "fsync", fsync)
    monkeypatch.setattr(config_manager.os, "replace", replace)
    manager._write_manifest(after)

    assert events == ["fsync", "replace"]
    assert perf_batch._load_manifest(path.parent) == after
    assert "测试完成" in path.read_text(encoding="utf-8")
    assert list(path.parent.iterdir()) == [path]
    manager.logger.warning.assert_not_called()


@pytest.mark.parametrize("failure", ["serialize", "fsync", "replace"])
def test_writer_failure_keeps_previous_manifest_and_cleans_tempfile(
    manager, monkeypatch, failure,
):
    path = _manifest_path(manager)
    manager._write_manifest({"previous": True})
    previous = path.read_bytes()

    def fail(*args):
        raise OSError("injected write failure")

    payload = {"next": True}
    if failure == "serialize":
        payload["invalid"] = object()
    else:
        monkeypatch.setattr(config_manager.os, failure, fail)
    manager.logger.reset_mock()

    manager._write_manifest(payload)

    assert path.read_bytes() == previous
    assert list(path.parent.iterdir()) == [path]
    manager.logger.warning.assert_called_once()
    assert str(path) in manager.logger.warning.call_args.args[0]
    manager.logger.info.assert_not_called()


def test_writer_warns_if_the_run_directory_cannot_be_created(manager):
    path = _manifest_path(manager)
    path.parent.write_text("not a directory", encoding="utf-8")

    manager._write_manifest({"combos": []})

    assert path.parent.read_text(encoding="utf-8") == "not a directory"
    manager.logger.warning.assert_called_once()
    manager.logger.info.assert_not_called()


def test_writer_ignores_cleanup_error_after_failed_publication(manager, monkeypatch):
    path = _manifest_path(manager)
    manager._write_manifest({"previous": True})
    cleanup_paths = []

    def replace(*args):
        raise OSError("publication failed")

    def remove(filename):
        cleanup_paths.append(Path(filename))
        raise OSError("cleanup failed")

    with monkeypatch.context() as patches:
        patches.setattr(config_manager.os, "replace", replace)
        patches.setattr(config_manager.os, "remove", remove)
        manager._write_manifest({"next": True})

    assert perf_batch._load_manifest(path.parent) == {"previous": True}
    assert len(cleanup_paths) == 1
    assert cleanup_paths[0].parent == path.parent
    assert json.loads(cleanup_paths[0].read_text(encoding="utf-8")) == {"next": True}
    assert "publication failed" in manager.logger.warning.call_args.args[0]


def test_writer_without_work_dir_does_not_open_a_manifest(manager, monkeypatch):
    manager.cfg.clear()
    mkdir = Mock(side_effect=AssertionError("missing work_dir must not write"))
    monkeypatch.setattr(config_manager.os, "makedirs", mkdir)

    manager._write_manifest({"combos": []})

    mkdir.assert_not_called()
    assert not manager.logger.mock_calls


@pytest.mark.parametrize(
    ("content", "warning_expected"),
    [("{", True), ("null", False), ("[]", False), ('"not an object"', False)],
    ids=["truncated", "null", "array", "string"],
)
def test_invalid_manifest_is_ignored_by_aggregation(tmp_path, monkeypatch, content, warning_expected):
    (tmp_path / "combos_manifest.json").write_text(content, encoding="utf-8")
    warning = Mock()
    renderer = Mock()
    monkeypatch.setattr(perf_batch.logger, "warning", warning)
    monkeypatch.setattr(perf_batch, "render_batch_perf_report", renderer)

    artifacts = perf_batch.aggregate_batch_perf(tmp_path)

    assert artifacts.csv_path is None
    assert artifacts.html_path is None
    assert warning.called is warning_expected
    renderer.assert_not_called()
    assert [path.name for path in tmp_path.iterdir()] == ["combos_manifest.json"]


def test_explicit_manifest_is_used_instead_of_disk_and_warnings_are_deduplicated(
    tmp_path, monkeypatch,
):
    (tmp_path / "combos_manifest.json").write_text("{}", encoding="utf-8")
    explicit = {
        "combos": [
            {"model_abbr": "m8", "dataset_abbr": "d", "batch_size": 8},
            {"model_abbr": "m2", "dataset_abbr": "d", "batch_size": 2},
        ],
        "warnings": ["first", "shared", "first"],
    }
    renderer = Mock(return_value=None)
    monkeypatch.setattr(perf_batch, "render_batch_perf_report", renderer)

    artifacts = perf_batch.aggregate_batch_perf(
        tmp_path, manifest=explicit, warnings=["shared", "last"],
    )

    assert artifacts.csv_path.is_file()
    rows, manifest, warnings, work_dir = renderer.call_args.args
    assert [row["model_abbr"] for row in rows] == ["m2", "m8"]
    assert [row["status"] for row in rows] == ["fail", "fail"]
    assert manifest == explicit
    assert warnings == ["first", "shared", "last"]
    assert work_dir == tmp_path
    assert explicit["warnings"] == ["first", "shared", "first"]
