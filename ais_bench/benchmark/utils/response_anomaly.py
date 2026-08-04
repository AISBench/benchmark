"""Background msProbe response anomaly detection for completed AISBench predictions."""

import json
import os
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.utils.results import safe_write


_ANOMALY_TYPE_NAMES = {
    0: "normal",
    1: "rare_character",
    2: "garbled",
    3: "repetition",
    4: "nan_value",
}


class ResponseAnomalyCoordinator:
    """Run response anomaly detection independently from the evaluation stage."""

    STATUS_TASK_NAME = "ResponseAnomaly"
    STATUS_FILE_NAME = "tmp_ResponseAnomaly.json"

    def __init__(self) -> None:
        self.logger = AISLogger()
        self._thread: Optional[threading.Thread] = None
        self._summary: Dict[str, int] = {}

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def summary(self) -> Dict[str, int]:
        return dict(self._summary)

    def start(self, cfg: Dict[str, Any]) -> None:
        if self.is_running:
            return
        self._summary = {}
        self._thread = threading.Thread(
            target=self._detect,
            args=(cfg,),
            name="response-anomaly",
            daemon=False,
        )
        self._thread.start()

    def join(self) -> None:
        if self._thread:
            self._thread.join()

    def _detect(self, cfg: Dict[str, Any]) -> None:
        work_dir = cfg["work_dir"]
        status_dir = Path(work_dir) / "status_tmp"
        status_file = status_dir / self.STATUS_FILE_NAME
        total = 0
        completed = 0
        counts: Counter[str] = Counter()
        try:
            # The feature only supports service-model generation chains.
            pairs = [
                (model["abbr"], dataset["abbr"], model)
                for model in cfg.get("models", [])
                if model.get("attr", "service") == "service"
                for dataset in cfg.get("datasets", [])
            ]

            task_groups = []
            for model_abbr, dataset_abbr, model_cfg in pairs:
                prediction_file = (
                    Path(work_dir) / "predictions" / model_abbr / f"{dataset_abbr}.jsonl"
                )
                predictions = self._read_jsonl(prediction_file)
                task_groups.append(
                    (model_abbr, dataset_abbr, model_cfg, prediction_file, predictions)
                )
                total += len(predictions)

            self._post_status(
                status_file, completed, total, counts, "response anomaly detecting"
            )

            model_name_warned = False
            for model_abbr, dataset_abbr, model_cfg, prediction_file, predictions in task_groups:
                anomaly_cfg = self._merge_model_anomaly_config(
                    model_cfg, cfg["response_anomaly"]
                )
                try:
                    anomaly_cfg = self._prepare_model_config(
                        model_abbr, anomaly_cfg, work_dir
                    )
                    detector, init_error = self._build_detector(anomaly_cfg)
                except Exception as exc:
                    self.logger.error(
                        "Failed to prepare response anomaly detection for model "
                        "%s: %s",
                        model_abbr,
                        exc,
                    )
                    detector = None
                    init_error = (
                        "failed",
                        f"Failed to prepare msProbe configuration: {exc}",
                    )
                result_file = (
                    Path(work_dir)
                    / "response_anomaly"
                    / model_abbr
                    / f"{dataset_abbr}.jsonl"
                )
                prediction_keys = {
                    f"{item.get('id')}:{item.get('uuid')}" for item in predictions
                }
                inherited = self._load_inherited_results(result_file, prediction_keys)
                completed += len(inherited)
                counts.update(
                    item.get("anomaly_type_name", "unknown")
                    for item in inherited.values()
                )

                if not model_name_warned and not anomaly_cfg.get("model_name"):
                    self.logger.warning(
                        "response_anomaly.model_name is not set; falling back to model "
                        "abbr '%s'. msProbe model matching may be degraded.",
                        model_cfg.get("abbr"),
                    )
                    model_name_warned = True

                for prediction in predictions:
                    case_id = str(prediction.get("id"))
                    case_key = f"{prediction.get('id')}:{prediction.get('uuid')}"
                    if case_key in inherited:
                        continue
                    result = self._detect_case(
                        prediction, anomaly_cfg, detector, init_error
                    )
                    result_file.parent.mkdir(parents=True, exist_ok=True)
                    safe_write({case_id: result}, result_file)
                    completed += 1
                    counts[result["anomaly_type_name"]] += 1
                    self._post_status(
                        status_file,
                        completed,
                        total,
                        counts,
                        "response anomaly detecting",
                    )

            self._summary = dict(counts)
            self._post_status(
                status_file,
                completed,
                total,
                counts,
                "response anomaly finished",
                "finish",
            )
        except Exception as exc:
            self.logger.error("Response anomaly detection failed: %s", exc)
            self._summary = dict(counts)
            self._post_status(
                status_file,
                completed,
                total,
                counts,
                f"response anomaly failed: {exc}",
                "error",
            )

    def _load_inherited_results(
        self, result_file: Path, prediction_keys: Iterable[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Return previously completed results whose id+uuid still exist in predictions.

        Matching on id+uuid ensures that a re-inferred response (different
        uuid) is not incorrectly assigned a stale anomaly result.
        Non-final statuses (skipped/unavailable/failed) are intentionally not
        inherited so they can be retried on resume.
        """
        existing_by_key: Dict[str, Dict[str, Any]] = {}
        for item in self._read_jsonl(result_file):
            key = f"{item.get('id')}:{item.get('uuid')}"
            existing_by_key[key] = item
        return {
            key: item
            for key, item in existing_by_key.items()
            if key in prediction_keys and item.get("detection_status") == "completed"
        }

    @staticmethod
    def _merge_model_anomaly_config(
        model_cfg: Dict[str, Any], global_cfg: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge global response_anomaly config with the model-level overrides."""
        merged = dict(global_cfg)
        model_cfg_anomaly = dict(model_cfg.get("response_anomaly") or {})
        for key, value in model_cfg_anomaly.items():
            if value is not None:
                merged[key] = value
        return merged

    def _prepare_model_config(
        self,
        model_abbr: str,
        anomaly_cfg: Dict[str, Any],
        work_dir: str,
    ) -> Dict[str, Any]:
        """Auto-generate msProbe model files when a local model path is given."""
        model_path = anomaly_cfg.get("model_path")
        if not model_path:
            return anomaly_cfg

        has_mtype = bool(anomaly_cfg.get("msprobe_mtype_path"))
        has_tk2cat = bool(anomaly_cfg.get("msprobe_token2category_dir"))
        if has_mtype and has_tk2cat:
            return anomaly_cfg
        if has_mtype != has_tk2cat:
            raise RuntimeError(
                "response_anomaly.msprobe_mtype_path and "
                "response_anomaly.msprobe_token2category_dir must be configured "
                "together; either provide both or rely on model_path "
                "auto-generation."
            )

        from ais_bench.tools.response_anomaly.gen_model_config import (
            generate_model_config,
        )

        output_dir = Path(work_dir) / "response_anomaly_config" / model_abbr
        generated = generate_model_config(
            model_path=str(model_path),
            model_name=anomaly_cfg.get("model_name"),
            output_dir=str(output_dir),
        )
        merged = dict(anomaly_cfg)
        for key, value in generated.items():
            # Overwrite None/empty values (e.g. msprobe_config_path set to
            # None by ConfigManager) so auto-generated paths take effect.
            if not merged.get(key):
                merged[key] = value
        return merged

    @staticmethod
    def _build_detector(anomaly_cfg: Dict[str, Any]):
        """Create one msProbe ILLDetector with the configured file paths."""
        try:
            import msprobe.response_anomaly as response_anomaly_pkg
            from msprobe.response_anomaly.detector import ILLDetector
        except ImportError:
            return None, (
                "unavailable",
                "mindstudio-probe is required for response anomaly detection. "
                "Install the AISBench response_anomaly extra.",
            )

        base = Path(response_anomaly_pkg.__file__).resolve().parent
        config_path = anomaly_cfg.get("msprobe_config_path") or str(
            base / "configs" / "config.yaml"
        )
        mtype_path = anomaly_cfg.get("msprobe_mtype_path") or str(
            base / "configs" / "mtype_config.json"
        )
        tk2cat_path = anomaly_cfg.get("msprobe_token2category_dir") or str(
            base / "token2category"
        )
        try:
            detector = ILLDetector(config_path, mtype_path, tk2cat_path)
        except Exception as exc:
            return None, (
                "failed",
                f"Failed to initialize msProbe detector: {exc}",
            )
        return detector, None

    def _detect_case(
        self,
        prediction: Dict[str, Any],
        anomaly_cfg: Dict[str, Any],
        detector=None,
        init_error=None,
    ) -> Dict[str, Any]:
        result = {
            "id": prediction.get("id"),
            "uuid": prediction.get("uuid"),
            "is_anomaly": False,
            "anomaly_type": 0,
            "anomaly_type_name": "normal",
        }
        payload = prediction.get("response_anomaly_payload")
        if not isinstance(payload, dict):
            result["detection_status"] = "skipped"
            result["reason"] = "Response does not contain token ids and top-k logprobs."
            result["anomaly_type_name"] = "skipped"
            return result

        tokens = payload.get("tokens")
        topk_logprobs = payload.get("topk_logprobs")
        if (
            not isinstance(tokens, list)
            or not isinstance(topk_logprobs, list)
            or len(tokens) == 0
            or len(tokens) != len(topk_logprobs)
            or any(not isinstance(item, dict) or not item for item in topk_logprobs)
        ):
            result["detection_status"] = "skipped"
            result["reason"] = (
                "tokens and topk_logprobs must be non-empty lists of equal length "
                "with non-empty per-token logprob maps."
            )
            result["anomaly_type_name"] = "skipped"
            return result

        if init_error is not None:
            status, reason = init_error
            result.update(
                detection_status=status,
                reason=reason,
                anomaly_type_name=status,
            )
            return result

        try:
            topk_logprobs = self._normalize_logprobs(topk_logprobs)
            tokens = [int(token) for token in tokens]
            model_name = anomaly_cfg.get("model_name")
            is_anomaly, anomaly_type = detector.run(
                [topk_logprobs], [tokens], [model_name]
            )[0]
            anomaly_type = int(anomaly_type)
            result.update(
                is_anomaly=bool(is_anomaly),
                anomaly_type=anomaly_type,
                anomaly_type_name=_ANOMALY_TYPE_NAMES.get(anomaly_type, "unknown"),
                detection_status="completed",
            )
        except Exception as exc:
            result.update(
                detection_status="failed",
                reason=f"{type(exc).__name__}: {exc}",
                anomaly_type_name="failed",
            )
        return result

    @staticmethod
    def _normalize_logprobs(items: Iterable[Dict[Any, Any]]) -> list[Dict[int, float]]:
        return [
            {int(token_id): float(logprob) for token_id, logprob in item.items()}
            for item in items
        ]

    def _post_status(
        self,
        status_file: Path,
        completed: int,
        total: int,
        counts: Counter[str],
        description: str,
        status: str = "response anomaly",
    ) -> None:
        """Atomically write the latest status.

        The status file is replaced instead of appended, so the coordinator
        writer and TasksMonitor readers never observe partial JSON.
        """
        status_file.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "task_name": self.STATUS_TASK_NAME,
                "process_id": os.getpid(),
                "finish_count": completed,
                "total_count": total,
                "progress_description": description,
                "status": status,
                "other_kwargs": dict(counts),
            }
        ]
        tmp_file = status_file.with_name(status_file.name + ".tmp")
        tmp_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(str(tmp_file), str(status_file))

    def _read_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        records = []
        with path.open(encoding="utf-8") as file:
            for line_no, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    self.logger.warning(
                        "Skip malformed line %s:%s: %s", path, line_no, exc
                    )
        return records
