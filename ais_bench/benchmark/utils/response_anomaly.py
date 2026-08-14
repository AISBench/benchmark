"""Background msProbe response anomaly detection for completed AISBench predictions."""

import json
import logging
import os
import shutil
import threading
import time
import uuid
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


class _ThreadLogFilter(logging.Filter):
    def __init__(self, thread_id: int) -> None:
        super().__init__()
        self.thread_id = thread_id

    def filter(self, record: logging.LogRecord) -> bool:
        return record.thread == self.thread_id


class ResponseAnomalyCoordinator:
    """Run response anomaly detection independently from the evaluation stage."""

    STATUS_TASK_NAME = "ResponseAnomaly"
    STATUS_FILE_NAME = "tmp_ResponseAnomaly.json"

    def __init__(self) -> None:
        self.logger = AISLogger()
        self._thread: Optional[threading.Thread] = None
        self._summary: Dict[str, int] = {}
        self._task_names: List[str] = []
        self._task_statuses: Dict[str, Dict[str, Any]] = {}

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def summary(self) -> Dict[str, int]:
        return dict(self._summary)

    @property
    def task_names(self) -> List[str]:
        return list(self._task_names or [self.STATUS_TASK_NAME])

    @classmethod
    def task_name(cls, model_abbr: str, dataset_abbr: str) -> str:
        return f"{cls.STATUS_TASK_NAME}/{model_abbr}/{dataset_abbr}"

    @staticmethod
    def task_log_path(model_abbr: str, dataset_abbr: str) -> str:
        return (
            Path("logs")
            .joinpath("response_anomaly", model_abbr, f"{dataset_abbr}.out")
            .as_posix()
        )

    @classmethod
    def task_names_from_cfg(cls, cfg: Dict[str, Any]) -> List[str]:
        names = [
            cls.task_name(model["abbr"], dataset["abbr"])
            for model in cfg.get("models", [])
            if model.get("attr", "service") == "service"
            for dataset in cfg.get("datasets", [])
        ]
        return names or [cls.STATUS_TASK_NAME]

    def start(self, cfg: Dict[str, Any]) -> None:
        if self.is_running:
            return
        self._summary = {}
        self._task_names = self.task_names_from_cfg(cfg)
        self._task_statuses = {}
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

    def _open_task_log(
        self, work_dir: str, model_abbr: str, dataset_abbr: str
    ) -> logging.FileHandler:
        log_file = Path(work_dir) / self.task_log_path(model_abbr, dataset_abbr)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        handler.setFormatter(self.logger.formatter)
        handler.addFilter(_ThreadLogFilter(threading.get_ident()))
        self.logger.logger.addHandler(handler)
        return handler

    def _close_task_log(self, handler: Optional[logging.FileHandler]) -> None:
        if handler is None:
            return
        handler.flush()
        self.logger.logger.removeHandler(handler)
        handler.close()

    def _detect(self, cfg: Dict[str, Any]) -> None:
        work_dir = cfg["work_dir"]
        status_dir = Path(work_dir) / "status_tmp"
        status_file = status_dir / self.STATUS_FILE_NAME
        counts: Counter[str] = Counter()
        active_log_handler = None
        current_task_name = self.STATUS_TASK_NAME
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

            self._task_names = [
                self.task_name(model_abbr, dataset_abbr)
                for model_abbr, dataset_abbr, _, _, _ in task_groups
            ] or [self.STATUS_TASK_NAME]
            self._task_statuses = {}
            if not task_groups:
                self.logger.warning(
                    "Response anomaly detection has no service model/dataset "
                    "groups to analyze under %s.",
                    Path(work_dir) / "predictions",
                )
                self._post_status(
                    status_file,
                    0,
                    0,
                    Counter(),
                    "response anomaly finished",
                    "finish",
                )
                return

            for model_abbr, dataset_abbr, _, _, predictions in task_groups:
                self._post_status(
                    status_file,
                    0,
                    len(predictions),
                    Counter(),
                    "waiting for response anomaly detection",
                    "start",
                    self.task_name(model_abbr, dataset_abbr),
                    self.task_log_path(model_abbr, dataset_abbr),
                )

            model_name_warned = False
            # Cache per-model config and detector so that a model with multiple
            # datasets only generates its msProbe config and initializes the
            # ILLDetector once (token2category loading is expensive).
            detector_cache: Dict[str, tuple] = {}
            for model_abbr, dataset_abbr, model_cfg, prediction_file, predictions in task_groups:
                current_task_name = self.task_name(model_abbr, dataset_abbr)
                task_log_path = self.task_log_path(model_abbr, dataset_abbr)
                group_total = len(predictions)
                group_completed = 0
                group_counts: Counter[str] = Counter()
                group_start_time = time.perf_counter()
                active_log_handler = self._open_task_log(
                    work_dir, model_abbr, dataset_abbr
                )
                self.logger.info("Task [%s]", current_task_name)
                self.logger.info("Found %d predictions", group_total)
                if not predictions:
                    self.logger.warning(
                        "No predictions found for model '%s' dataset '%s'; "
                        "response anomaly detection will skip this group.",
                        model_abbr,
                        dataset_abbr,
                    )
                if model_abbr in detector_cache:
                    anomaly_cfg, detector, init_error = detector_cache[model_abbr]
                    self.logger.info(
                        "Reuse response anomaly detector for model [%s]",
                        model_abbr,
                    )
                else:
                    anomaly_cfg = self._merge_model_anomaly_config(
                        model_cfg, cfg["response_anomaly"]
                    )
                    try:
                        self.logger.info(
                            "Preparing response anomaly config for model [%s]",
                            model_abbr,
                        )
                        self._post_status(
                            status_file,
                            group_completed,
                            group_total,
                            group_counts,
                            f"preparing response anomaly config for {model_abbr}",
                            task_name=current_task_name,
                            task_log_path=task_log_path,
                        )
                        anomaly_cfg = self._prepare_model_config(
                            model_abbr, anomaly_cfg, work_dir
                        )
                        self.logger.info(
                            "Loading response anomaly detector for model [%s]",
                            model_abbr,
                        )
                        self._post_status(
                            status_file,
                            group_completed,
                            group_total,
                            group_counts,
                            f"loading response anomaly detector for {model_abbr}",
                            task_name=current_task_name,
                            task_log_path=task_log_path,
                        )
                        detector, init_error = self._build_detector(anomaly_cfg)
                        if detector is not None:
                            self._cache_detector_token_categories(detector)
                            self.logger.info(
                                "Response anomaly detector initialized for model [%s]",
                                model_abbr,
                            )
                        elif init_error:
                            self.logger.warning(
                                "Response anomaly detector is %s: %s",
                                init_error[0],
                                init_error[1],
                            )
                    except Exception as exc:
                        self.logger.logger.error(
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
                    detector_cache[model_abbr] = (anomaly_cfg, detector, init_error)
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
                inherited_names = [
                    item.get("anomaly_type_name", "unknown")
                    for item in inherited.values()
                ]
                group_completed += len(inherited)
                group_counts.update(inherited_names)
                counts.update(inherited_names)
                if inherited:
                    self.logger.info(
                        "Found %d completed response anomaly results in cache",
                        len(inherited),
                    )

                if not model_name_warned and not anomaly_cfg.get("model_name"):
                    self.logger.warning(
                        "response_anomaly.model_name is not set; falling back to model "
                        "abbr '%s'. msProbe model matching may be degraded.",
                        model_cfg.get("abbr"),
                    )
                    model_name_warned = True

                # Ensure the result directory exists once per group instead of
                # once per prediction (mkdir with exist_ok=True is idempotent).
                result_file.parent.mkdir(parents=True, exist_ok=True)
                retention = cfg["response_anomaly"].get(
                    "payload_retention", "anomalies"
                )
                storage_cfg = cfg["response_anomaly"].get(
                    "payload_storage", {}
                )
                payload_dir = (
                    Path(work_dir)
                    / "response_anomaly"
                    / model_abbr
                    / "payload"
                    / dataset_abbr
                )
                source_dir = (
                    Path(work_dir)
                    / "response_anomaly"
                    / model_abbr
                    / "payload_staging"
                    / dataset_abbr
                )
                staging_dir = payload_dir.with_name(
                    f".{dataset_abbr}.payload-build-{uuid.uuid4().hex[:8]}"
                )
                payload_writer = None
                if payload_dir.exists():
                    old_manifest_path = payload_dir / "payload_manifest.json"
                    if not old_manifest_path.exists():
                        raise RuntimeError(
                            "Existing response anomaly payload archive has no "
                            "manifest. Use a new work directory."
                        )
                    old_manifest = json.loads(
                        old_manifest_path.read_text(encoding="utf-8")
                    )
                    old_retention = old_manifest.get(
                        "payload_retention", "all"
                    )
                    if old_retention != retention:
                        raise RuntimeError(
                            "Cannot change response anomaly payload_retention "
                            "while reusing an existing payload archive. Use a "
                            "new work directory."
                        )
                    if retention == "all":
                        source_dir.parent.mkdir(parents=True, exist_ok=True)
                        if source_dir.exists():
                            for item in payload_dir.glob("part-*.jsonl.zst"):
                                shutil.copy2(
                                    item,
                                    source_dir / item.name,
                                )
                        else:
                            shutil.copytree(payload_dir, source_dir)
                        old_source_manifest = source_dir / "payload_manifest.json"
                        if old_source_manifest.exists():
                            old_source_manifest.unlink()
                if retention == "anomalies":
                    from ais_bench.benchmark.utils.response_anomaly_jsonl import (
                        ResponseAnomalyJsonlWriter,
                    )

                    if payload_dir.exists():
                        shutil.copytree(payload_dir, staging_dir)
                    payload_writer = ResponseAnomalyJsonlWriter(
                        staging_dir,
                        compression_level=storage_cfg.get(
                            "compression_level", 3
                        ),
                        rows_per_shard=storage_cfg.get(
                            "rows_per_shard", 2000
                        ),
                    )
                from ais_bench.benchmark.utils.response_anomaly_jsonl import (
                    iter_jsonl_zstd_records,
                )

                self.logger.info(
                    "Response anomaly detecting %s/%s from %s",
                    model_abbr,
                    dataset_abbr,
                    source_dir,
                )
                self.logger.info(
                    "Start detecting %d response anomaly payloads",
                    max(0, group_total - group_completed),
                )
                self._post_status(
                    status_file,
                    group_completed,
                    group_total,
                    group_counts,
                    f"streaming response anomaly payloads for {dataset_abbr}",
                    task_name=current_task_name,
                    task_log_path=task_log_path,
                )
                result_batch = {}
                detected_keys = set(inherited)
                last_status_time = time.monotonic()
                shard_rows: Counter[str] = Counter()
                for payload_record in iter_jsonl_zstd_records(source_dir):
                    shard_rows[payload_record["payload_shard"]] += 1
                    case_key = (
                        f"{payload_record.get('id')}:"
                        f"{payload_record.get('uuid')}"
                    )
                    if case_key not in prediction_keys or case_key in detected_keys:
                        continue
                    result = self._detect_case(
                        payload_record, anomaly_cfg, detector, init_error
                    )
                    result["payload_shard"] = payload_record["payload_shard"]
                    result["payload_row"] = payload_record["payload_row"]
                    result_batch[case_key] = result
                    if payload_writer and self._should_retain_payload(
                        retention, result
                    ):
                        payload_writer.write(payload_record)
                    detected_keys.add(case_key)
                    group_completed += 1
                    group_counts[result["anomaly_type_name"]] += 1
                    counts[result["anomaly_type_name"]] += 1
                    now = time.monotonic()
                    if len(result_batch) >= 100:
                        safe_write(result_batch, result_file)
                        result_batch = {}
                    if now - last_status_time >= 1.0:
                        self._post_status(
                            status_file,
                            group_completed,
                            group_total,
                            group_counts,
                            "response anomaly detecting",
                            task_name=current_task_name,
                            task_log_path=task_log_path,
                        )
                        last_status_time = now
                if result_batch:
                    safe_write(result_batch, result_file)
                legacy_writer = None
                for prediction in predictions:
                    case_key = f"{prediction.get('id')}:{prediction.get('uuid')}"
                    if case_key in detected_keys:
                        continue
                    result = self._detect_case(
                        prediction, anomaly_cfg, detector, init_error
                    )
                    safe_write({case_key: result}, result_file)
                    if payload_writer and self._should_retain_payload(
                        retention, result
                    ):
                        payload_writer.write(prediction)
                    elif (
                        retention == "all"
                        and isinstance(
                            prediction.get("response_anomaly_payload"), dict
                        )
                    ):
                        from ais_bench.benchmark.utils.response_anomaly_jsonl import (
                            ResponseAnomalyJsonlWriter,
                        )

                        if legacy_writer is None:
                            legacy_writer = ResponseAnomalyJsonlWriter(
                                source_dir,
                                storage_cfg.get("compression_level", 3),
                                storage_cfg.get("rows_per_shard", 2000),
                            )
                        legacy_writer.write(prediction)
                    group_completed += 1
                    group_counts[result["anomaly_type_name"]] += 1
                    counts[result["anomaly_type_name"]] += 1
                if legacy_writer is not None:
                    legacy_writer.close(write_manifest=False)
                self._post_status(
                    status_file,
                    group_completed,
                    group_total,
                    group_counts,
                    f"finalizing response anomaly payloads for {dataset_abbr}",
                    task_name=current_task_name,
                    task_log_path=task_log_path,
                )
                if retention == "all":
                    from ais_bench.benchmark.utils.response_anomaly_jsonl import (
                        build_jsonl_zstd_manifest,
                    )

                    build_jsonl_zstd_manifest(
                        source_dir,
                        storage_cfg.get("compression_level", 3),
                        retention,
                        dict(shard_rows),
                    )
                    self._replace_payload_archive(source_dir, payload_dir)
                elif payload_writer is not None:
                    manifest = payload_writer.close(retention)
                    if manifest["total_rows"] or not payload_dir.exists():
                        self._replace_payload_archive(staging_dir, payload_dir)
                    else:
                        shutil.rmtree(staging_dir)
                elif payload_dir.exists():
                    shutil.rmtree(payload_dir)
                if source_dir.exists():
                    shutil.rmtree(source_dir)
                if any(
                    "response_anomaly_payload" in prediction
                    for prediction in predictions
                ):
                    self._strip_payloads_from_predictions(
                        prediction_file, predictions
                    )

                elapsed = time.perf_counter() - group_start_time
                self.logger.info(
                    "Response anomaly detection completed: %s",
                    dict(group_counts),
                )
                self.logger.info(
                    "Response anomaly task time elapsed: %.2fs", elapsed
                )
                self.logger.info("Task state is finish, exit loop")
                self._post_status(
                    status_file,
                    group_completed,
                    group_total,
                    group_counts,
                    "response anomaly finished",
                    "finish",
                    current_task_name,
                    task_log_path,
                )
                self._close_task_log(active_log_handler)
                active_log_handler = None

            self._summary = dict(counts)
        except Exception as exc:
            self.logger.logger.error("Response anomaly detection failed: %s", exc)
            self._summary = dict(counts)
            for task_name in self.task_names:
                state = self._task_statuses.get(task_name, {})
                if state.get("status") in ("finish", "error"):
                    continue
                self._post_status(
                    status_file,
                    state.get("finish_count", 0),
                    state.get("total_count", 0),
                    Counter(state.get("other_kwargs", {})),
                    f"response anomaly failed: {exc}",
                    "error",
                    task_name,
                    state.get("task_log_path"),
                )
        finally:
            self._close_task_log(active_log_handler)

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
    def _should_retain_payload(
        retention: str, result: Dict[str, Any]
    ) -> bool:
        if retention == "all":
            return True
        if retention == "none":
            return False
        return bool(result.get("is_anomaly")) or result.get(
            "detection_status"
        ) in ("failed", "unavailable")

    @staticmethod
    def _replace_payload_archive(staging_dir: Path, payload_dir: Path) -> None:
        payload_dir.parent.mkdir(parents=True, exist_ok=True)
        backup_dir = payload_dir.with_name(payload_dir.name + ".old")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        if payload_dir.exists():
            os.replace(str(payload_dir), str(backup_dir))
        try:
            os.replace(str(staging_dir), str(payload_dir))
        except Exception:
            if backup_dir.exists() and not payload_dir.exists():
                os.replace(str(backup_dir), str(payload_dir))
            raise
        if backup_dir.exists():
            shutil.rmtree(backup_dir)

    @staticmethod
    def _strip_payloads_from_predictions(
        prediction_file: Path, predictions: List[Dict[str, Any]]
    ) -> None:
        if not prediction_file.exists():
            return
        tmp_file = prediction_file.with_name(prediction_file.name + ".tmp")
        with tmp_file.open("w", encoding="utf-8") as file:
            for prediction in predictions:
                prediction.pop("response_anomaly_payload", None)
                file.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        os.replace(str(tmp_file), str(prediction_file))

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

    @staticmethod
    def _cache_detector_token_categories(detector) -> None:
        """Cache msProbe token-category maps instead of loading them per case."""
        get_tk2cat = getattr(detector, "get_tk2cat", None)
        if not callable(get_tk2cat):
            return
        cache = {}

        def cached_get_tk2cat(eos_token, model_config=None):
            try:
                model_key = json.dumps(
                    model_config, ensure_ascii=False, sort_keys=True
                )
            except (TypeError, ValueError):
                model_key = repr(model_config)
            key = (int(eos_token), model_key)
            if key not in cache:
                cache[key] = get_tk2cat(eos_token, model_config)
            return cache[key]

        detector.get_tk2cat = cached_get_tk2cat

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
        task_name: Optional[str] = None,
        task_log_path: Optional[str] = None,
    ) -> None:
        """Atomically write the latest status.

        The status file is replaced instead of appended, so the coordinator
        writer and TasksMonitor readers never observe partial JSON.
        """
        status_file.parent.mkdir(parents=True, exist_ok=True)
        task_name = task_name or self.STATUS_TASK_NAME
        state = {
            "task_name": task_name,
            "process_id": os.getpid(),
            "finish_count": completed,
            "total_count": total,
            "progress_description": description,
            "status": status,
            "other_kwargs": dict(counts),
        }
        if task_log_path:
            state["task_log_path"] = task_log_path
        self._task_statuses[task_name] = state
        payload = list(self._task_statuses.values())
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
