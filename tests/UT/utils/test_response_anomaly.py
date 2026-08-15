import json
import sys
import threading
import types
from collections import Counter

import pytest

from ais_bench.benchmark.utils.response_anomaly import ResponseAnomalyCoordinator
from ais_bench.benchmark.utils.response_anomaly_jsonl import (
    ResponseAnomalyJsonlWriter,
    iter_jsonl_zstd_records,
)


class FakeDetector:
    def __init__(self, result):
        self.result = result

    def run(self, topk_logprobs, tokens, model_configs):
        return [self.result]


class TokenDetector:
    def run(self, topk_logprobs, tokens, model_configs):
        return [[tokens[0][0] == 2, 1 if tokens[0][0] == 2 else 0]]


def _payload_record(case_id):
    return {
        "data_abbr": "ds",
        "id": case_id,
        "uuid": f"u{case_id}",
        "response_anomaly_payload": {
            "tokens": [case_id],
            "topk_logprobs": [{str(case_id): -0.1}],
        },
    }


def _completed_anomaly_result(case_id):
    return {
        "id": case_id,
        "uuid": f"u{case_id}",
        "is_anomaly": True,
        "anomaly_type": 2,
        "anomaly_type_name": "garbled",
        "detection_status": "completed",
    }


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _anomaly_cfg(tmp_path):
    return {
        "work_dir": str(tmp_path),
        "models": [{"abbr": "modelA", "attr": "service"}],
        "datasets": [{"abbr": "ds"}],
        "response_anomaly": {"payload_retention": "anomalies"},
    }


def test_detect_case_calls_msprobe_detector():
    result = ResponseAnomalyCoordinator()._detect_case(
        {
            "id": 2,
            "uuid": "case-2",
            "response_anomaly_payload": {
                "tokens": [1, 2],
                "topk_logprobs": [{"1": -0.1}, {"2": -0.2}],
            },
        },
        {"model_name": "model"},
        FakeDetector([True, 3]),
        None,
    )

    assert result["is_anomaly"] is True
    assert result["anomaly_type"] == 3
    assert result["anomaly_type_name"] == "repetition"


def test_detect_case_skips_missing_token_payload():
    result = ResponseAnomalyCoordinator()._detect_case(
        {"id": 2, "uuid": "case-2"}, {}, None, None
    )

    assert result["detection_status"] == "skipped"
    assert result["is_anomaly"] is False


def test_detect_case_skips_inconsistent_payload():
    result = ResponseAnomalyCoordinator()._detect_case(
        {
            "id": 3,
            "uuid": "case-3",
            "response_anomaly_payload": {
                "tokens": [1, 2],
                "topk_logprobs": [{"1": -0.1}],
            },
        },
        {},
        None,
        None,
    )

    assert result["detection_status"] == "skipped"
    assert "equal length" in result["reason"]


def test_detect_case_reports_detector_init_error():
    result = ResponseAnomalyCoordinator()._detect_case(
        {
            "id": 4,
            "uuid": "case-4",
            "response_anomaly_payload": {
                "tokens": [1],
                "topk_logprobs": [{"1": -0.1}],
            },
        },
        {},
        None,
        ("unavailable", "mindstudio-probe is required"),
    )

    assert result["detection_status"] == "unavailable"
    assert result["anomaly_type_name"] == "unavailable"


def test_build_detector_reports_missing_msprobe(monkeypatch):
    monkeypatch.setitem(sys.modules, "msprobe", None)

    detector, init_error = ResponseAnomalyCoordinator._build_detector({})

    assert detector is None
    assert init_error[0] == "unavailable"
    assert "mindstudio-probe" in init_error[1]


def test_build_detector_reports_ill_detector_init_failure(monkeypatch):
    msprobe_pkg = types.ModuleType("msprobe")
    response_anomaly_pkg = types.ModuleType("msprobe.response_anomaly")
    response_anomaly_pkg.__file__ = (
        "/fake/msprobe/response_anomaly/__init__.py"
    )
    detector_module = types.ModuleType("msprobe.response_anomaly.detector")

    class FailingILLDetector:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("boom")

    detector_module.ILLDetector = FailingILLDetector
    monkeypatch.setitem(sys.modules, "msprobe", msprobe_pkg)
    monkeypatch.setitem(
        sys.modules, "msprobe.response_anomaly", response_anomaly_pkg
    )
    monkeypatch.setitem(
        sys.modules, "msprobe.response_anomaly.detector", detector_module
    )

    detector, init_error = ResponseAnomalyCoordinator._build_detector({})

    assert detector is None
    assert init_error[0] == "failed"
    assert "boom" in init_error[1]


def test_cache_detector_token_categories_loads_each_key_once():
    class Detector:
        def __init__(self):
            self.calls = 0

        def get_tk2cat(self, eos_token, model_config=None):
            self.calls += 1
            return {"1": "latin"}, 100

    detector = Detector()
    ResponseAnomalyCoordinator._cache_detector_token_categories(detector)

    first = detector.get_tk2cat(2, "model")
    second = detector.get_tk2cat(2, "model")
    third = detector.get_tk2cat(3, "model")

    assert first == second
    assert third == first
    assert detector.calls == 2


def test_merge_model_anomaly_config_prefers_model_level():
    merged = ResponseAnomalyCoordinator._merge_model_anomaly_config(
        {
            "abbr": "qwen",
            "response_anomaly": {
                "model_name": "Qwen3-30B-A3B",
                "msprobe_mtype_path": "/custom/mtype.json",
            },
        },
        {
            "enabled": True,
            "model_name": "global-name",
            "msprobe_mtype_path": None,
        },
    )

    assert merged["model_name"] == "Qwen3-30B-A3B"
    assert merged["msprobe_mtype_path"] == "/custom/mtype.json"


def test_prepare_model_config_auto_generates_when_paths_missing(
    tmp_path, monkeypatch
):
    generated = {
        "msprobe_config_path": str(tmp_path / "config.yaml"),
        "msprobe_mtype_path": str(tmp_path / "mtype.json"),
        "msprobe_token2category_dir": str(tmp_path / "tk2cat"),
    }
    monkeypatch.setattr(
        "ais_bench.tools.response_anomaly.gen_model_config.generate_model_config",
        lambda **kwargs: generated,
    )

    cfg = ResponseAnomalyCoordinator()._prepare_model_config(
        "qwen",
        {"model_path": "/models/qwen", "model_name": "Qwen3-30B-A3B"},
        str(tmp_path),
    )

    assert cfg["msprobe_mtype_path"] == str(tmp_path / "mtype.json")
    assert cfg["msprobe_token2category_dir"] == str(tmp_path / "tk2cat")
    assert cfg["model_name"] == "Qwen3-30B-A3B"


def test_prepare_model_config_overwrites_none_config_path(tmp_path, monkeypatch):
    """自动生成的 config.yaml 路径应覆盖 anomaly_cfg 中的 None 值。"""
    generated = {
        "msprobe_config_path": str(tmp_path / "config.yaml"),
        "msprobe_mtype_path": str(tmp_path / "mtype.json"),
        "msprobe_token2category_dir": str(tmp_path / "tk2cat"),
    }
    monkeypatch.setattr(
        "ais_bench.tools.response_anomaly.gen_model_config.generate_model_config",
        lambda **kwargs: generated,
    )

    cfg = ResponseAnomalyCoordinator()._prepare_model_config(
        "qwen",
        {
            "model_path": "/models/qwen",
            "model_name": "Qwen3-30B-A3B",
            "msprobe_config_path": None,
        },
        str(tmp_path),
    )

    assert cfg["msprobe_config_path"] == str(tmp_path / "config.yaml")


def test_prepare_model_config_requires_both_custom_paths(tmp_path):
    with pytest.raises(RuntimeError):
        ResponseAnomalyCoordinator()._prepare_model_config(
            "qwen",
            {
                "model_path": "/models/qwen",
                "msprobe_mtype_path": "/tmp/mtype.json",
            },
            str(tmp_path),
        )


def test_post_status_is_atomic(tmp_path):
    coordinator = ResponseAnomalyCoordinator()
    status_file = tmp_path / ResponseAnomalyCoordinator.STATUS_FILE_NAME

    coordinator._post_status(
        status_file,
        completed=1,
        total=2,
        counts=Counter({"normal": 1}),
        description="detecting",
    )

    data = json.loads(status_file.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["task_name"] == "ResponseAnomaly"
    assert data[0]["finish_count"] == 1
    assert data[0]["other_kwargs"] == {"normal": 1}
    assert not status_file.with_name(status_file.name + ".tmp").exists()


def test_post_status_keeps_each_model_dataset_task(tmp_path):
    coordinator = ResponseAnomalyCoordinator()
    status_file = tmp_path / ResponseAnomalyCoordinator.STATUS_FILE_NAME

    for dataset_abbr in ("ds1", "ds2"):
        coordinator._post_status(
            status_file,
            completed=1,
            total=2,
            counts=Counter({"normal": 1}),
            description="detecting",
            task_name=coordinator.task_name("modelA", dataset_abbr),
            task_log_path=coordinator.task_log_path("modelA", dataset_abbr),
        )

    data = json.loads(status_file.read_text(encoding="utf-8"))
    assert [item["task_name"] for item in data] == [
        "ResponseAnomaly/modelA/ds1",
        "ResponseAnomaly/modelA/ds2",
    ]
    assert [item["task_log_path"] for item in data] == [
        "logs/response_anomaly/modelA/ds1.out",
        "logs/response_anomaly/modelA/ds2.out",
    ]


def test_task_log_only_captures_response_anomaly_thread(tmp_path):
    coordinator = ResponseAnomalyCoordinator()
    handler = coordinator._open_task_log(str(tmp_path), "modelA", "ds")
    coordinator.logger.info("response anomaly message")
    other_thread = threading.Thread(
        target=coordinator.logger.info,
        args=("unrelated workflow message",),
    )
    other_thread.start()
    other_thread.join()
    coordinator._close_task_log(handler)

    content = (
        tmp_path
        / "logs"
        / "response_anomaly"
        / "modelA"
        / "ds.out"
    ).read_text(encoding="utf-8")
    assert "response anomaly message" in content
    assert "unrelated workflow message" not in content


def test_read_jsonl_skips_broken_lines(tmp_path):
    path = tmp_path / "pred.jsonl"
    path.write_text('{"id": 1}\nnot-json\n{"id": 2}\n', encoding="utf-8")

    records = ResponseAnomalyCoordinator()._read_jsonl(path)

    assert [item["id"] for item in records] == [1, 2]


def test_load_inherited_results_only_keeps_completed(tmp_path):
    result_file = tmp_path / "result.jsonl"
    result_file.write_text(
        json.dumps({"id": 1, "uuid": "abc", "detection_status": "completed", "anomaly_type_name": "normal"})
        + "\n"
        + json.dumps({"id": 2, "uuid": "def", "detection_status": "skipped", "anomaly_type_name": "skipped"})
        + "\n",
        encoding="utf-8",
    )

    inherited = ResponseAnomalyCoordinator()._load_inherited_results(
        result_file, {"1:abc"}
    )

    assert set(inherited) == {"1:abc"}


def test_load_inherited_results_rejects_different_uuid(tmp_path):
    """同 id 不同 uuid 的旧结果不应被继承。"""
    result_file = tmp_path / "result.jsonl"
    result_file.write_text(
        json.dumps({"id": 1, "uuid": "old-uuid", "detection_status": "completed", "anomaly_type_name": "normal"})
        + "\n",
        encoding="utf-8",
    )

    inherited = ResponseAnomalyCoordinator()._load_inherited_results(
        result_file, {"1:new-uuid"}
    )

    assert len(inherited) == 0


@pytest.mark.parametrize(
    ("retention", "result", "expected"),
    [
        ("all", {"is_anomaly": False, "detection_status": "completed"}, True),
        ("anomalies", {"is_anomaly": True, "detection_status": "completed"}, True),
        ("anomalies", {"is_anomaly": False, "detection_status": "failed"}, True),
        ("anomalies", {"is_anomaly": False, "detection_status": "unavailable"}, True),
        ("anomalies", {"is_anomaly": False, "detection_status": "completed"}, False),
        ("anomalies", {"is_anomaly": False, "detection_status": "skipped"}, False),
        ("none", {"is_anomaly": True, "detection_status": "completed"}, False),
    ],
)
def test_should_retain_payload(retention, result, expected):
    assert (
        ResponseAnomalyCoordinator._should_retain_payload(retention, result)
        is expected
    )


def test_strip_payloads_from_predictions_is_atomic(tmp_path):
    prediction_file = tmp_path / "ds.jsonl"
    predictions = [
        {"id": 1, "response_anomaly_payload": {"tokens": [1]}},
        {"id": 2, "prediction": "ok"},
    ]
    prediction_file.write_text("old", encoding="utf-8")

    ResponseAnomalyCoordinator._strip_payloads_from_predictions(
        prediction_file, predictions
    )

    restored = [
        json.loads(line)
        for line in prediction_file.read_text(encoding="utf-8").splitlines()
    ]
    assert restored == [{"id": 1}, {"id": 2, "prediction": "ok"}]
    assert not prediction_file.with_name("ds.jsonl.tmp").exists()


def test_detect_runs_full_workflow(tmp_path, monkeypatch):
    """端到端驱动 _detect 主循环：读预测、逐条检测、写结果、收尾状态。"""
    work_dir = tmp_path
    prediction_file = work_dir / "predictions" / "modelA" / "ds.jsonl"
    prediction_file.parent.mkdir(parents=True)
    prediction_file.write_text(
        json.dumps(
            {
                "id": 1,
                "uuid": "u1",
                "response_anomaly_payload": {
                    "tokens": [1],
                    "topk_logprobs": [{"1": -0.1}],
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": 2,
                "uuid": "u2",
                "response_anomaly_payload": {
                    "tokens": [1, 2],
                    "topk_logprobs": [{"1": -0.1}, {"2": -0.2}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = {
        "work_dir": str(work_dir),
        "models": [{"abbr": "modelA", "attr": "service"}],
        "datasets": [{"abbr": "ds"}],
        "response_anomaly": {},
    }

    coordinator = ResponseAnomalyCoordinator()
    monkeypatch.setattr(
        coordinator,
        "_build_detector",
        lambda cfg: (FakeDetector([False, 0]), None),
    )
    coordinator._detect(cfg)

    result_lines = (
        work_dir / "response_anomaly" / "modelA" / "ds.jsonl"
    ).read_text(encoding="utf-8").strip().splitlines()
    assert len(result_lines) == 2
    results = [json.loads(line) for line in result_lines]
    assert all(item["detection_status"] == "completed" for item in results)
    assert coordinator.summary == {"normal": 2}

    status = json.loads(
        (
            work_dir / "status_tmp" / ResponseAnomalyCoordinator.STATUS_FILE_NAME
        ).read_text(encoding="utf-8")
    )[0]
    assert status["task_name"] == "ResponseAnomaly/modelA/ds"
    assert status["task_log_path"] == (
        "logs/response_anomaly/modelA/ds.out"
    )
    assert status["status"] == "finish"
    assert status["finish_count"] == 2
    assert status["total_count"] == 2
    log_file = work_dir / status["task_log_path"]
    assert log_file.exists()
    log_content = log_file.read_text(encoding="utf-8")
    assert "Task [ResponseAnomaly/modelA/ds]" in log_content
    assert "Found 2 predictions" in log_content
    assert "Response anomaly detection completed: {'normal': 2}" in log_content


def test_detect_writes_separate_status_and_log_for_each_dataset(
    tmp_path, monkeypatch
):
    for dataset_abbr in ("ds1", "ds2"):
        prediction_file = (
            tmp_path
            / "predictions"
            / "modelA"
            / f"{dataset_abbr}.jsonl"
        )
        prediction_file.parent.mkdir(parents=True, exist_ok=True)
        prediction_file.write_text(
            json.dumps(
                {
                    "id": 1,
                    "uuid": f"{dataset_abbr}-u1",
                    "response_anomaly_payload": {
                        "tokens": [1],
                        "topk_logprobs": [{"1": -0.1}],
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
    cfg = {
        "work_dir": str(tmp_path),
        "models": [{"abbr": "modelA", "attr": "service"}],
        "datasets": [{"abbr": "ds1"}, {"abbr": "ds2"}],
        "response_anomaly": {},
    }
    coordinator = ResponseAnomalyCoordinator()
    monkeypatch.setattr(
        coordinator,
        "_build_detector",
        lambda cfg: (FakeDetector([False, 0]), None),
    )

    coordinator._detect(cfg)

    statuses = json.loads(
        (
            tmp_path
            / "status_tmp"
            / ResponseAnomalyCoordinator.STATUS_FILE_NAME
        ).read_text(encoding="utf-8")
    )
    assert [item["task_name"] for item in statuses] == [
        "ResponseAnomaly/modelA/ds1",
        "ResponseAnomaly/modelA/ds2",
    ]
    assert all(item["status"] == "finish" for item in statuses)
    assert all(item["finish_count"] == 1 for item in statuses)
    for dataset_abbr in ("ds1", "ds2"):
        log_file = (
            tmp_path
            / "logs"
            / "response_anomaly"
            / "modelA"
            / f"{dataset_abbr}.out"
        )
        assert log_file.exists()
        assert (
            f"Task [ResponseAnomaly/modelA/{dataset_abbr}]"
            in log_file.read_text(encoding="utf-8")
        )


@pytest.mark.parametrize(
    ("retention", "expected_ids"),
    [("all", [1, 2]), ("anomalies", [2]), ("none", [])],
)
def test_detect_compresses_selected_payloads_and_strips_predictions(
    tmp_path, monkeypatch, retention, expected_ids
):
    import zstandard

    prediction_file = tmp_path / "predictions" / "modelA" / "ds.jsonl"
    prediction_file.parent.mkdir(parents=True)
    predictions = [
        {
            "data_abbr": "ds",
            "id": case_id,
            "uuid": f"u{case_id}",
            "prediction": "ok",
            "response_anomaly_payload": {
                "tokens": [case_id],
                "topk_logprobs": [{str(case_id): -0.1}],
            },
        }
        for case_id in (1, 2)
    ]
    prediction_file.write_text(
        "".join(json.dumps(item) + "\n" for item in predictions),
        encoding="utf-8",
    )
    cfg = {
        "work_dir": str(tmp_path),
        "models": [{"abbr": "modelA", "attr": "service"}],
        "datasets": [{"abbr": "ds"}],
        "response_anomaly": {
            "payload_retention": retention,
            "payload_storage": {
                "compression_level": 3,
                "rows_per_shard": 1,
            },
        },
    }
    coordinator = ResponseAnomalyCoordinator()
    monkeypatch.setattr(
        coordinator, "_build_detector", lambda cfg: (TokenDetector(), None)
    )

    coordinator._detect(cfg)

    restored_predictions = [
        json.loads(line)
        for line in prediction_file.read_text(encoding="utf-8").splitlines()
    ]
    assert all(
        "response_anomaly_payload" not in item
        for item in restored_predictions
    )
    payload_dir = (
        tmp_path / "response_anomaly" / "modelA" / "payload" / "ds"
    )
    if retention == "none":
        assert not payload_dir.exists()
        return
    manifest = json.loads(
        (payload_dir / "payload_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["payload_retention"] == retention
    assert manifest["total_rows"] == len(expected_ids)
    archived_ids = []
    for shard in sorted(payload_dir.glob("*.jsonl.zst")):
        with shard.open("rb") as file:
            reader = zstandard.ZstdDecompressor().stream_reader(file)
            archived_ids.extend(
                json.loads(line)["id"]
                for line in reader.read().decode("utf-8").splitlines()
            )
    assert archived_ids == expected_ids


def test_resume_backfills_inherited_anomaly_without_published_archive(
    tmp_path, monkeypatch
):
    prediction_file = tmp_path / "predictions" / "modelA" / "ds.jsonl"
    _write_jsonl(
        prediction_file,
        [{"data_abbr": "ds", "id": 2, "uuid": "u2", "prediction": "ok"}],
    )
    result_file = tmp_path / "response_anomaly" / "modelA" / "ds.jsonl"
    _write_jsonl(result_file, [_completed_anomaly_result(2)])
    source_dir = (
        tmp_path
        / "response_anomaly"
        / "modelA"
        / "payload_staging"
        / "ds"
    )
    source_writer = ResponseAnomalyJsonlWriter(source_dir, 3, 10)
    source_writer.write(_payload_record(2))
    source_writer.close(write_manifest=False)

    coordinator = ResponseAnomalyCoordinator()
    monkeypatch.setattr(
        coordinator, "_build_detector", lambda cfg: (TokenDetector(), None)
    )
    monkeypatch.setattr(
        coordinator,
        "_detect_case",
        lambda *args: pytest.fail("inherited cases must not be detected again"),
    )

    coordinator._detect(_anomaly_cfg(tmp_path))

    payload_dir = tmp_path / "response_anomaly" / "modelA" / "payload" / "ds"
    assert [
        record["id"] for record in iter_jsonl_zstd_records(payload_dir)
    ] == [2]


def test_resume_backfills_only_inherited_payloads_missing_from_archive(
    tmp_path, monkeypatch
):
    prediction_file = tmp_path / "predictions" / "modelA" / "ds.jsonl"
    _write_jsonl(
        prediction_file,
        [
            {"data_abbr": "ds", "id": 1, "uuid": "u1"},
            {"data_abbr": "ds", "id": 2, "uuid": "u2"},
        ],
    )
    result_file = tmp_path / "response_anomaly" / "modelA" / "ds.jsonl"
    _write_jsonl(
        result_file,
        [_completed_anomaly_result(1), _completed_anomaly_result(2)],
    )
    payload_dir = tmp_path / "response_anomaly" / "modelA" / "payload" / "ds"
    archive_writer = ResponseAnomalyJsonlWriter(payload_dir, 3, 10)
    archive_writer.write(_payload_record(1))
    archive_writer.close("anomalies")
    source_dir = (
        tmp_path
        / "response_anomaly"
        / "modelA"
        / "payload_staging"
        / "ds"
    )
    source_writer = ResponseAnomalyJsonlWriter(source_dir, 3, 10)
    source_writer.write(_payload_record(1))
    source_writer.write(_payload_record(2))
    source_writer.close(write_manifest=False)

    coordinator = ResponseAnomalyCoordinator()
    monkeypatch.setattr(
        coordinator, "_build_detector", lambda cfg: (TokenDetector(), None)
    )
    monkeypatch.setattr(
        coordinator,
        "_detect_case",
        lambda *args: pytest.fail("inherited cases must not be detected again"),
    )

    coordinator._detect(_anomaly_cfg(tmp_path))

    archived_ids = [
        record["id"] for record in iter_jsonl_zstd_records(payload_dir)
    ]
    assert sorted(archived_ids) == [1, 2]
    assert len(archived_ids) == len(set(archived_ids))


def test_resume_backfills_inherited_legacy_prediction_payload(
    tmp_path, monkeypatch
):
    prediction_file = tmp_path / "predictions" / "modelA" / "ds.jsonl"
    prediction = _payload_record(2)
    prediction["prediction"] = "ok"
    _write_jsonl(prediction_file, [prediction])
    result_file = tmp_path / "response_anomaly" / "modelA" / "ds.jsonl"
    _write_jsonl(result_file, [_completed_anomaly_result(2)])

    coordinator = ResponseAnomalyCoordinator()
    monkeypatch.setattr(
        coordinator, "_build_detector", lambda cfg: (TokenDetector(), None)
    )
    monkeypatch.setattr(
        coordinator,
        "_detect_case",
        lambda *args: pytest.fail("inherited cases must not be detected again"),
    )

    coordinator._detect(_anomaly_cfg(tmp_path))

    payload_dir = tmp_path / "response_anomaly" / "modelA" / "payload" / "ds"
    assert [
        record["id"] for record in iter_jsonl_zstd_records(payload_dir)
    ] == [2]


def test_resume_preserves_inherited_legacy_payload_in_all_mode(
    tmp_path, monkeypatch
):
    prediction_file = tmp_path / "predictions" / "modelA" / "ds.jsonl"
    prediction = _payload_record(2)
    prediction["prediction"] = "ok"
    _write_jsonl(prediction_file, [prediction])
    result_file = tmp_path / "response_anomaly" / "modelA" / "ds.jsonl"
    _write_jsonl(result_file, [_completed_anomaly_result(2)])
    cfg = _anomaly_cfg(tmp_path)
    cfg["response_anomaly"]["payload_retention"] = "all"

    coordinator = ResponseAnomalyCoordinator()
    monkeypatch.setattr(
        coordinator, "_build_detector", lambda cfg: (TokenDetector(), None)
    )
    monkeypatch.setattr(
        coordinator,
        "_detect_case",
        lambda *args: pytest.fail("inherited cases must not be detected again"),
    )

    coordinator._detect(cfg)

    payload_dir = tmp_path / "response_anomaly" / "modelA" / "payload" / "ds"
    assert [
        record["id"] for record in iter_jsonl_zstd_records(payload_dir)
    ] == [2]


def test_detect_noop_resume_keeps_payload_archive_unchanged(
    tmp_path, monkeypatch
):
    import ais_bench.benchmark.utils.response_anomaly as anomaly_module
    import ais_bench.benchmark.utils.response_anomaly_jsonl as jsonl_module

    prediction_file = tmp_path / "predictions" / "modelA" / "ds.jsonl"
    prediction_file.parent.mkdir(parents=True)
    prediction_file.write_text(
        json.dumps(
            {
                "data_abbr": "ds",
                "id": 1,
                "uuid": "u1",
                "prediction": "ok",
                "response_anomaly_payload": {
                    "tokens": [1],
                    "topk_logprobs": [{"1": -0.1}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = {
        "work_dir": str(tmp_path),
        "models": [{"abbr": "modelA", "attr": "service"}],
        "datasets": [{"abbr": "ds"}],
        "response_anomaly": {"payload_retention": "all"},
    }
    coordinator = ResponseAnomalyCoordinator()
    monkeypatch.setattr(
        coordinator, "_build_detector", lambda cfg: (TokenDetector(), None)
    )
    coordinator._detect(cfg)

    payload_dir = tmp_path / "response_anomaly" / "modelA" / "payload" / "ds"
    archive_before = {
        path.name: path.read_bytes() for path in payload_dir.iterdir()
    }

    def unexpected_copy(*args, **kwargs):
        raise AssertionError("no-op resume must not copy the payload archive")

    def unexpected_decode(path):
        raise AssertionError("no-op resume must not decode payload shards")

    monkeypatch.setattr(anomaly_module.shutil, "copytree", unexpected_copy)
    monkeypatch.setattr(anomaly_module.shutil, "copy2", unexpected_copy)
    monkeypatch.setattr(
        jsonl_module, "_iter_jsonl_zstd_shard", unexpected_decode
    )

    resumed = ResponseAnomalyCoordinator()
    monkeypatch.setattr(
        resumed, "_build_detector", lambda cfg: (TokenDetector(), None)
    )
    resumed._detect(cfg)

    assert resumed.summary == {"normal": 1}
    assert archive_before == {
        path.name: path.read_bytes() for path in payload_dir.iterdir()
    }


def test_detect_legacy_payload_uses_writer_row_counts(tmp_path, monkeypatch):
    import ais_bench.benchmark.utils.response_anomaly_jsonl as jsonl_module

    prediction_file = tmp_path / "predictions" / "modelA" / "ds.jsonl"
    prediction_file.parent.mkdir(parents=True)
    prediction_file.write_text(
        json.dumps(
            {
                "data_abbr": "ds",
                "id": 1,
                "uuid": "u1",
                "response_anomaly_payload": {
                    "tokens": [1],
                    "topk_logprobs": [{"1": -0.1}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = {
        "work_dir": str(tmp_path),
        "models": [{"abbr": "modelA", "attr": "service"}],
        "datasets": [{"abbr": "ds"}],
        "response_anomaly": {"payload_retention": "all"},
    }

    def unexpected_decode(path):
        raise AssertionError("new legacy shards already have writer row counts")

    monkeypatch.setattr(
        jsonl_module, "_iter_jsonl_zstd_shard", unexpected_decode
    )
    coordinator = ResponseAnomalyCoordinator()
    monkeypatch.setattr(
        coordinator, "_build_detector", lambda cfg: (TokenDetector(), None)
    )

    coordinator._detect(cfg)

    manifest = json.loads(
        (
            tmp_path
            / "response_anomaly"
            / "modelA"
            / "payload"
            / "ds"
            / "payload_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["total_rows"] == 1


def test_detect_cleans_payload_build_directory_after_failure(
    tmp_path, monkeypatch
):
    prediction_file = tmp_path / "predictions" / "modelA" / "ds.jsonl"
    prediction_file.parent.mkdir(parents=True)
    prediction_file.write_text(
        json.dumps(
            {
                "data_abbr": "ds",
                "id": 2,
                "uuid": "u2",
                "response_anomaly_payload": {
                    "tokens": [2],
                    "topk_logprobs": [{"2": -0.1}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = {
        "work_dir": str(tmp_path),
        "models": [{"abbr": "modelA", "attr": "service"}],
        "datasets": [{"abbr": "ds"}],
        "response_anomaly": {"payload_retention": "anomalies"},
    }
    coordinator = ResponseAnomalyCoordinator()
    monkeypatch.setattr(
        coordinator, "_build_detector", lambda cfg: (TokenDetector(), None)
    )
    original_post_status = coordinator._post_status

    def fail_during_finalization(*args, **kwargs):
        if len(args) > 4 and str(args[4]).startswith("finalizing"):
            raise RuntimeError("injected finalization failure")
        return original_post_status(*args, **kwargs)

    monkeypatch.setattr(coordinator, "_post_status", fail_during_finalization)

    coordinator._detect(cfg)

    payload_parent = tmp_path / "response_anomaly" / "modelA" / "payload"
    assert not list(payload_parent.glob(".ds.payload-build-*"))


def test_cleanup_stale_payload_build_directories(tmp_path):
    payload_dir = tmp_path / "payload" / "ds"
    payload_dir.parent.mkdir(parents=True)
    stale = payload_dir.parent / ".ds.payload-build-deadbeef"
    unrelated = payload_dir.parent / ".other.payload-build-deadbeef"
    stale.mkdir()
    unrelated.mkdir()

    ResponseAnomalyCoordinator()._cleanup_stale_payload_build_dirs(payload_dir)

    assert not stale.exists()
    assert unrelated.exists()


def test_cleanup_stale_payload_build_failure_warns_and_continues(
    tmp_path, monkeypatch
):
    payload_dir = tmp_path / "payload" / "ds"
    payload_dir.parent.mkdir(parents=True)
    stale = payload_dir.parent / ".ds.payload-build-deadbeef"
    stale.mkdir()
    warnings = []
    coordinator = ResponseAnomalyCoordinator()

    def fail_rmtree(path, *args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(
        "ais_bench.benchmark.utils.response_anomaly.shutil.rmtree",
        fail_rmtree,
    )
    monkeypatch.setattr(
        coordinator.logger,
        "warning",
        lambda message, *args: warnings.append(message % args),
    )

    coordinator._cleanup_stale_payload_build_dirs(payload_dir)

    assert stale.exists()
    assert any(str(stale) in warning for warning in warnings)


def test_seed_payload_directory_falls_back_to_copy(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    destination_dir = tmp_path / "destination"
    source_dir.mkdir()
    shard = source_dir / "part-00000.jsonl.zst"
    shard.write_bytes(b"payload")
    (source_dir / "payload_manifest.json").write_text(
        '{"total_rows": 1}', encoding="utf-8"
    )

    def fail_link(*args):
        raise OSError("unsupported")

    monkeypatch.setattr(
        "ais_bench.benchmark.utils.response_anomaly.os.link",
        fail_link,
    )

    ResponseAnomalyCoordinator._seed_payload_directory(
        source_dir, destination_dir, include_manifest=True
    )

    assert (destination_dir / shard.name).read_bytes() == b"payload"
    assert json.loads(
        (destination_dir / "payload_manifest.json").read_text(encoding="utf-8")
    ) == {"total_rows": 1}

    ResponseAnomalyCoordinator._seed_payload_directory(
        source_dir, destination_dir
    )
    assert not (destination_dir / "payload_manifest.json").exists()


def test_detect_warns_when_no_predictions_found(tmp_path, monkeypatch):
    """没有任何预测样本时应告警而不是静默完成。"""
    warnings = []
    coordinator = ResponseAnomalyCoordinator()
    monkeypatch.setattr(
        coordinator,
        "_build_detector",
        lambda cfg: (FakeDetector([False, 0]), None),
    )
    monkeypatch.setattr(
        coordinator.logger,
        "warning",
        lambda msg, *args: warnings.append(msg),
    )
    cfg = {
        "work_dir": str(tmp_path),
        "models": [{"abbr": "modelA", "attr": "service"}],
        "datasets": [{"abbr": "ds"}],
        "response_anomaly": {},
    }

    coordinator._detect(cfg)

    assert coordinator.summary == {}
    status = json.loads(
        (
            tmp_path / "status_tmp" / ResponseAnomalyCoordinator.STATUS_FILE_NAME
        ).read_text(encoding="utf-8")
    )[0]
    assert status["status"] == "finish"
    assert status["total_count"] == 0
    assert any("No predictions" in message for message in warnings)
