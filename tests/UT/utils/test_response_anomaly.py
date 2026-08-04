import json
import sys
import types
from collections import Counter

import pytest

from ais_bench.benchmark.utils.response_anomaly import ResponseAnomalyCoordinator


class FakeDetector:
    def __init__(self, result):
        self.result = result

    def run(self, topk_logprobs, tokens, model_configs):
        return [self.result]


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
            "top_logprobs": 20,
            "msprobe_mtype_path": None,
        },
    )

    assert merged["model_name"] == "Qwen3-30B-A3B"
    assert merged["msprobe_mtype_path"] == "/custom/mtype.json"
    assert merged["top_logprobs"] == 20


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
