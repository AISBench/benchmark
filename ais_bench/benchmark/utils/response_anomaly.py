"""msProbe response anomaly detection for completed AISBench predictions."""

import json
import logging
import os
import shutil
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
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

# anomaly_type_name values that indicate a detected response anomaly
# (as opposed to non-detection statuses such as skipped/failed/unavailable).
ANOMALY_RESULT_NAMES = frozenset(
    {"rare_character", "garbled", "repetition", "nan_value", "unknown"}
)


class _ThreadLogFilter(logging.Filter):
    def __init__(self, thread_id: int) -> None:
        super().__init__()
        self.thread_id = thread_id

    def filter(self, record: logging.LogRecord) -> bool:
        return record.thread == self.thread_id


@dataclass
class _DetectionTask:
    model_abbr: str
    dataset_abbr: str
    model_cfg: Dict[str, Any]
    prediction_file: Path
    predictions: List[Dict[str, Any]]


@dataclass
class _GroupProgress:
    total: int
    completed: int = 0
    counts: Counter[str] = field(default_factory=Counter)


@dataclass
class _PayloadState:
    retention: str
    storage_cfg: Dict[str, Any]
    payload_dir: Path
    source_dir: Path
    staging_dir: Path
    archive_is_current: bool
    writer: Any = None
    retained_keys: set[str] = field(default_factory=set)


@dataclass
class _DetectionContext:
    task_name: str
    task_log_path: str
    progress: _GroupProgress
    started_at: float
    anomaly_cfg: Dict[str, Any]
    detector: Any
    init_error: Any
    result_file: Path
    prediction_keys: set[str]
    inherited: Dict[str, Dict[str, Any]]
    payload: _PayloadState


@dataclass
class _ActiveResources:
    log_handler: Optional[logging.FileHandler] = None
    payload_build_dir: Optional[Path] = None
    payload_writers: List[Any] = field(default_factory=list)


class ResponseAnomalyCoordinator:
    """Run detection serially in Infer while a status board refreshes."""

    STATUS_TASK_NAME = "ResponseAnomaly"
    STATUS_FILE_NAME = "tmp_ResponseAnomaly.json"

    def __init__(self) -> None:
        self.logger = AISLogger()
        self._thread: Optional[threading.Thread] = None
        self._summary: Dict[str, int] = {}
        self._task_names: List[str] = []
        self._task_statuses: Dict[str, Dict[str, Any]] = {}
        self._anomaly_report: Dict[str, Dict[str, Any]] = {}

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def summary(self) -> Dict[str, int]:
        return dict(self._summary)

    @property
    def anomaly_report(self) -> Dict[str, Dict[str, Any]]:
        """Per-task anomaly counts and on-disk locations for user guidance."""
        return {name: dict(info) for name, info in self._anomaly_report.items()}

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
        resources = _ActiveResources()
        try:
            task_groups = self._build_detection_tasks(cfg, work_dir)
            if not self._initialize_detection_tasks(
                task_groups, status_file, work_dir
            ):
                return

            model_name_warned = False
            # Cache per-model config and detector so that a model with multiple
            # datasets only generates its msProbe config and initializes the
            # ILLDetector once (token2category loading is expensive).
            detector_cache: Dict[str, tuple] = {}
            for task in task_groups:
                model_name_warned = self._detect_task(
                    task,
                    cfg,
                    work_dir,
                    status_file,
                    counts,
                    detector_cache,
                    resources,
                    model_name_warned,
                )

            self._summary = dict(counts)
        except Exception as exc:
            self._handle_detection_failure(status_file, counts, exc)
        finally:
            self._cleanup_detection_resources(
                resources.log_handler,
                resources.payload_build_dir,
                resources.payload_writers,
            )

    def _build_detection_tasks(
        self, cfg: Dict[str, Any], work_dir: str
    ) -> List[_DetectionTask]:
        """Build model/dataset tasks and load their prediction records."""
        tasks = []
        for model in cfg.get("models", []):
            if model.get("attr", "service") != "service":
                continue
            for dataset in cfg.get("datasets", []):
                model_abbr = model["abbr"]
                dataset_abbr = dataset["abbr"]
                prediction_file = (
                    Path(work_dir)
                    / "predictions"
                    / model_abbr
                    / f"{dataset_abbr}.jsonl"
                )
                tasks.append(
                    _DetectionTask(
                        model_abbr=model_abbr,
                        dataset_abbr=dataset_abbr,
                        model_cfg=model,
                        prediction_file=prediction_file,
                        predictions=self._read_jsonl(prediction_file),
                    )
                )
        return tasks

    def _initialize_detection_tasks(
        self,
        tasks: List[_DetectionTask],
        status_file: Path,
        work_dir: str,
    ) -> bool:
        """Initialize status entries and report whether work is available."""
        self._task_names = [
            self.task_name(task.model_abbr, task.dataset_abbr) for task in tasks
        ] or [self.STATUS_TASK_NAME]
        self._task_statuses = {}
        if not tasks:
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
            return False
        for task in tasks:
            self._post_status(
                status_file,
                0,
                len(task.predictions),
                Counter(),
                "waiting for response anomaly detection",
                "start",
                self.task_name(task.model_abbr, task.dataset_abbr),
                self.task_log_path(task.model_abbr, task.dataset_abbr),
            )
        return True

    def _detect_task(
        self,
        task: _DetectionTask,
        cfg: Dict[str, Any],
        work_dir: str,
        status_file: Path,
        counts: Counter[str],
        detector_cache: Dict[str, tuple],
        resources: _ActiveResources,
        model_name_warned: bool,
    ) -> bool:
        """Detect anomalies and publish results for one model/dataset pair."""
        context, model_name_warned = self._prepare_detection_context(
            task,
            cfg,
            work_dir,
            status_file,
            counts,
            detector_cache,
            resources,
            model_name_warned,
        )
        self.logger.info(
            "Response anomaly detecting %s/%s from %s",
            task.model_abbr,
            task.dataset_abbr,
            context.payload.source_dir,
        )
        self.logger.info(
            "Start detecting %d response anomaly payloads",
            max(0, context.progress.total - context.progress.completed),
        )
        self._post_status(
            status_file,
            context.progress.completed,
            context.progress.total,
            context.progress.counts,
            f"streaming response anomaly payloads for {task.dataset_abbr}",
            task_name=context.task_name,
            task_log_path=context.task_log_path,
        )
        detected_keys, shard_rows, context.progress.completed = (
            self._process_staged_payloads(
                context.payload,
                context.prediction_keys,
                context.inherited,
                context.anomaly_cfg,
                context.detector,
                context.init_error,
                context.result_file,
                status_file,
                context.task_name,
                context.task_log_path,
                context.progress.total,
                context.progress.completed,
                context.progress.counts,
                counts,
            )
        )
        context.progress.completed = self._process_prediction_payloads(
            task,
            context.payload,
            context.inherited,
            detected_keys,
            context.anomaly_cfg,
            context.detector,
            context.init_error,
            context.result_file,
            context.progress.completed,
            context.progress.counts,
            counts,
            shard_rows,
            resources.payload_writers,
        )
        self._finalize_group_payloads(
            task,
            context.payload,
            shard_rows,
            status_file,
            context.task_name,
            context.task_log_path,
            context.progress.total,
            context.progress.completed,
            context.progress.counts,
            resources.payload_writers,
        )
        resources.payload_build_dir = None
        self._finish_detection_task(
            context.task_name,
            context.task_log_path,
            work_dir,
            context.result_file,
            context.payload.payload_dir,
            context.progress.counts,
            context.progress.completed,
            context.progress.total,
            status_file,
            context.started_at,
        )
        self._close_task_log(resources.log_handler)
        resources.log_handler = None
        return model_name_warned

    def _prepare_detection_context(
        self,
        task: _DetectionTask,
        cfg: Dict[str, Any],
        work_dir: str,
        status_file: Path,
        counts: Counter[str],
        detector_cache: Dict[str, tuple],
        resources: _ActiveResources,
        model_name_warned: bool,
    ) -> tuple[_DetectionContext, bool]:
        """Prepare detector, resume state, and payload storage for one task."""
        task_name = self.task_name(task.model_abbr, task.dataset_abbr)
        task_log_path = self.task_log_path(task.model_abbr, task.dataset_abbr)
        progress = _GroupProgress(total=len(task.predictions))
        started_at = time.perf_counter()
        resources.log_handler = self._open_task_log(
            work_dir, task.model_abbr, task.dataset_abbr
        )
        self.logger.info("Task [%s]", task_name)
        self.logger.info("Found %d predictions", progress.total)
        if not task.predictions:
            self.logger.warning(
                "No predictions found for model '%s' dataset '%s'; "
                "response anomaly detection will skip this group.",
                task.model_abbr,
                task.dataset_abbr,
            )

        anomaly_cfg, detector, init_error = self._get_model_detector(
            task,
            cfg["response_anomaly"],
            work_dir,
            status_file,
            progress.completed,
            progress.total,
            progress.counts,
            task_name,
            task_log_path,
            detector_cache,
        )
        result_file = (
            Path(work_dir)
            / "response_anomaly"
            / task.model_abbr
            / f"{task.dataset_abbr}.jsonl"
        )
        prediction_keys = {
            f"{item.get('id')}:{item.get('uuid')}" for item in task.predictions
        }
        inherited = self._load_inherited_results(result_file, prediction_keys)
        inherited_names = [
            item.get("anomaly_type_name", "unknown")
            for item in inherited.values()
        ]
        progress.completed += len(inherited)
        progress.counts.update(inherited_names)
        counts.update(inherited_names)
        if inherited:
            self.logger.info(
                "Found %d completed response anomaly results in cache",
                len(inherited),
            )
        if not model_name_warned and not anomaly_cfg.get("model_name"):
            # ConfigManager normally resolves model_name (explicit value, or
            # the model_path basename) before the task runs, so this branch
            # only triggers on manually built configs. The detector receives
            # None as the model name in that case; warn truthfully instead
            # of claiming an abbr fallback that never happens.
            self.logger.warning(
                "response_anomaly.model_name is not set; msProbe will be "
                "called without a model name and model matching may be "
                "degraded. Set response_anomaly.model_name for model '%s'.",
                task.model_cfg.get("abbr"),
            )
            model_name_warned = True

        result_file.parent.mkdir(parents=True, exist_ok=True)
        payload = self._prepare_payload_state(
            task,
            cfg["response_anomaly"],
            work_dir,
            prediction_keys,
            inherited,
            resources.payload_writers,
        )
        resources.payload_build_dir = payload.staging_dir
        context = _DetectionContext(
            task_name=task_name,
            task_log_path=task_log_path,
            progress=progress,
            started_at=started_at,
            anomaly_cfg=anomaly_cfg,
            detector=detector,
            init_error=init_error,
            result_file=result_file,
            prediction_keys=prediction_keys,
            inherited=inherited,
            payload=payload,
        )
        return context, model_name_warned

    def _get_model_detector(
        self,
        task: _DetectionTask,
        global_cfg: Dict[str, Any],
        work_dir: str,
        status_file: Path,
        completed: int,
        total: int,
        counts: Counter[str],
        task_name: str,
        task_log_path: str,
        detector_cache: Dict[str, tuple],
    ) -> tuple:
        """Return a cached detector or prepare one for the task model."""
        if task.model_abbr in detector_cache:
            self.logger.info(
                "Reuse response anomaly detector for model [%s]",
                task.model_abbr,
            )
            return detector_cache[task.model_abbr]

        anomaly_cfg = self._merge_model_anomaly_config(task.model_cfg, global_cfg)
        try:
            self.logger.info(
                "Preparing response anomaly config for model [%s]",
                task.model_abbr,
            )
            self._post_status(
                status_file,
                completed,
                total,
                counts,
                f"preparing response anomaly config for {task.model_abbr}",
                task_name=task_name,
                task_log_path=task_log_path,
            )
            anomaly_cfg = self._prepare_model_config(
                task.model_abbr, anomaly_cfg, work_dir
            )
            self.logger.info(
                "Loading response anomaly detector for model [%s]",
                task.model_abbr,
            )
            self._post_status(
                status_file,
                completed,
                total,
                counts,
                f"loading response anomaly detector for {task.model_abbr}",
                task_name=task_name,
                task_log_path=task_log_path,
            )
            detector, init_error = self._build_detector(anomaly_cfg)
            if detector is not None:
                self._cache_detector_token_categories(detector)
                self.logger.info(
                    "Response anomaly detector initialized for model [%s]",
                    task.model_abbr,
                )
            elif init_error:
                self.logger.warning(
                    "Response anomaly detector is %s: %s",
                    init_error[0],
                    init_error[1],
                )
        except Exception as exc:
            self.logger.logger.error(
                "Failed to prepare response anomaly detection for model %s: %s",
                task.model_abbr,
                exc,
            )
            detector = None
            init_error = (
                "failed",
                f"Failed to prepare msProbe configuration: {exc}",
            )
        detector_cache[task.model_abbr] = (anomaly_cfg, detector, init_error)
        return detector_cache[task.model_abbr]

    def _prepare_payload_state(
        self,
        task: _DetectionTask,
        anomaly_cfg: Dict[str, Any],
        work_dir: str,
        prediction_keys: set[str],
        inherited: Dict[str, Dict[str, Any]],
        active_writers: List[Any],
    ) -> _PayloadState:
        """Validate the existing archive and prepare the next payload build."""
        retention = anomaly_cfg.get("payload_retention", "anomalies")
        storage_cfg = anomaly_cfg.get("payload_storage", {})
        payload_dir = (
            Path(work_dir)
            / "response_anomaly"
            / task.model_abbr
            / "payload"
            / task.dataset_abbr
        )
        source_dir = (
            Path(work_dir)
            / "response_anomaly"
            / task.model_abbr
            / "payload_staging"
            / task.dataset_abbr
        )
        staging_dir = payload_dir.with_name(
            f".{task.dataset_abbr}.payload-build-{uuid.uuid4().hex[:8]}"
        )
        self._cleanup_stale_payload_build_dirs(payload_dir)
        if payload_dir.exists():
            manifest_path = payload_dir / "payload_manifest.json"
            if not manifest_path.exists():
                raise RuntimeError(
                    "Existing response anomaly payload archive has no "
                    "manifest. Use a new work directory."
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("payload_retention", "all") != retention:
                raise RuntimeError(
                    "Cannot change response anomaly payload_retention while "
                    "reusing an existing payload archive. Use a new work directory."
                )
        pending_keys = prediction_keys.difference(inherited)
        archive_is_current = (
            not pending_keys
            and not source_dir.exists()
            and (
                payload_dir.exists()
                if retention != "none"
                else not payload_dir.exists()
            )
        )
        if archive_is_current:
            self.logger.info(
                "No new response anomaly payloads for %s/%s; "
                "keeping the existing payload archive unchanged",
                task.model_abbr,
                task.dataset_abbr,
            )
        elif payload_dir.exists() and retention == "all":
            self._seed_payload_directory(payload_dir, source_dir)

        state = _PayloadState(
            retention=retention,
            storage_cfg=storage_cfg,
            payload_dir=payload_dir,
            source_dir=source_dir,
            staging_dir=staging_dir,
            archive_is_current=archive_is_current,
        )
        if retention != "anomalies" or archive_is_current:
            return state

        from ais_bench.benchmark.utils.response_anomaly_jsonl import (
            ResponseAnomalyJsonlWriter,
            iter_jsonl_zstd_records,
        )

        if payload_dir.exists():
            state.retained_keys = {
                f"{record.get('id')}:{record.get('uuid')}"
                for record in iter_jsonl_zstd_records(payload_dir)
            }
            self._seed_payload_directory(
                payload_dir, staging_dir, include_manifest=True
            )
        state.writer = ResponseAnomalyJsonlWriter(
            staging_dir,
            compression_level=storage_cfg.get("compression_level", 3),
            rows_per_shard=storage_cfg.get("rows_per_shard", 2000),
        )
        active_writers.append(state.writer)
        return state

    def _process_staged_payloads(
        self,
        payload: _PayloadState,
        prediction_keys: set[str],
        inherited: Dict[str, Dict[str, Any]],
        anomaly_cfg: Dict[str, Any],
        detector,
        init_error,
        result_file: Path,
        status_file: Path,
        task_name: str,
        task_log_path: str,
        total: int,
        completed: int,
        group_counts: Counter[str],
        counts: Counter[str],
    ) -> tuple:
        """Detect payloads from compressed staging shards."""
        from ais_bench.benchmark.utils.response_anomaly_jsonl import (
            iter_jsonl_zstd_records,
        )

        result_batch = {}
        detected_keys = set(inherited)
        last_status_time = time.monotonic()
        shard_rows: Counter[str] = Counter()
        for record in iter_jsonl_zstd_records(payload.source_dir):
            shard_rows[record["payload_shard"]] += 1
            case_key = f"{record.get('id')}:{record.get('uuid')}"
            if payload.retention == "all":
                payload.retained_keys.add(case_key)
            if case_key not in prediction_keys:
                continue
            if case_key in inherited:
                self._write_retained_payload(
                    payload.writer,
                    payload.retention,
                    inherited[case_key],
                    record,
                    case_key,
                    payload.retained_keys,
                )
                continue
            if case_key in detected_keys:
                continue
            result = self._detect_case(record, anomaly_cfg, detector, init_error)
            result_batch[case_key] = result
            self._write_retained_payload(
                payload.writer,
                payload.retention,
                result,
                record,
                case_key,
                payload.retained_keys,
            )
            detected_keys.add(case_key)
            completed += 1
            result_name = result["anomaly_type_name"]
            group_counts[result_name] += 1
            counts[result_name] += 1
            now = time.monotonic()
            if len(result_batch) >= 100:
                safe_write(result_batch, result_file)
                result_batch = {}
            if now - last_status_time >= 1.0:
                self._post_status(
                    status_file,
                    completed,
                    total,
                    group_counts,
                    "response anomaly detecting",
                    task_name=task_name,
                    task_log_path=task_log_path,
                )
                last_status_time = now
        if result_batch:
            safe_write(result_batch, result_file)
        return detected_keys, shard_rows, completed

    def _process_prediction_payloads(
        self,
        task: _DetectionTask,
        payload: _PayloadState,
        inherited: Dict[str, Dict[str, Any]],
        detected_keys: set[str],
        anomaly_cfg: Dict[str, Any],
        detector,
        init_error,
        result_file: Path,
        completed: int,
        group_counts: Counter[str],
        counts: Counter[str],
        shard_rows: Counter[str],
        active_writers: List[Any],
    ) -> int:
        """Process legacy payloads still embedded in prediction records."""
        legacy_writer = None
        for prediction in task.predictions:
            case_key = f"{prediction.get('id')}:{prediction.get('uuid')}"
            if case_key in inherited:
                result = inherited[case_key]
                is_inherited = True
            else:
                if case_key in detected_keys:
                    continue
                result = self._detect_case(
                    prediction, anomaly_cfg, detector, init_error
                )
                safe_write({case_key: result}, result_file)
                is_inherited = False
            self._write_retained_payload(
                payload.writer,
                payload.retention,
                result,
                prediction,
                case_key,
                payload.retained_keys,
            )
            if (
                payload.retention == "all"
                and not payload.archive_is_current
                and case_key not in payload.retained_keys
                and isinstance(prediction.get("response_anomaly_payload"), dict)
            ):
                from ais_bench.benchmark.utils.response_anomaly_jsonl import (
                    ResponseAnomalyJsonlWriter,
                )

                if legacy_writer is None:
                    legacy_writer = ResponseAnomalyJsonlWriter(
                        payload.source_dir,
                        payload.storage_cfg.get("compression_level", 3),
                        payload.storage_cfg.get("rows_per_shard", 2000),
                    )
                    active_writers.append(legacy_writer)
                legacy_writer.write(prediction)
                payload.retained_keys.add(case_key)
            if is_inherited:
                continue
            completed += 1
            result_name = result["anomaly_type_name"]
            group_counts[result_name] += 1
            counts[result_name] += 1
        if legacy_writer is not None:
            manifest = legacy_writer.close(write_manifest=False)
            active_writers.remove(legacy_writer)
            shard_rows.update(
                {shard["file"]: shard["rows"] for shard in manifest["shards"]}
            )
        return completed

    def _finalize_group_payloads(
        self,
        task: _DetectionTask,
        payload: _PayloadState,
        shard_rows: Counter[str],
        status_file: Path,
        task_name: str,
        task_log_path: str,
        total: int,
        completed: int,
        counts: Counter[str],
        active_writers: List[Any],
    ) -> None:
        """Publish the retained archive and remove temporary payloads."""
        self._post_status(
            status_file,
            completed,
            total,
            counts,
            f"finalizing response anomaly payloads for {task.dataset_abbr}",
            task_name=task_name,
            task_log_path=task_log_path,
        )
        if payload.archive_is_current:
            pass
        elif payload.retention == "all":
            from ais_bench.benchmark.utils.response_anomaly_jsonl import (
                build_jsonl_zstd_manifest,
            )

            build_jsonl_zstd_manifest(
                payload.source_dir,
                payload.storage_cfg.get("compression_level", 3),
                payload.retention,
                dict(shard_rows),
            )
            self._replace_payload_archive(
                payload.source_dir, payload.payload_dir
            )
        elif payload.writer is not None:
            manifest = payload.writer.close(payload.retention)
            active_writers.remove(payload.writer)
            if manifest["total_rows"] or not payload.payload_dir.exists():
                self._replace_payload_archive(
                    payload.staging_dir, payload.payload_dir
                )
            else:
                shutil.rmtree(payload.staging_dir)
        elif payload.payload_dir.exists():
            shutil.rmtree(payload.payload_dir)
        if payload.source_dir.exists():
            shutil.rmtree(payload.source_dir)
        if any(
            "response_anomaly_payload" in prediction
            for prediction in task.predictions
        ):
            self._strip_payloads_from_predictions(
                task.prediction_file, task.predictions
            )

    def _finish_detection_task(
        self,
        task_name: str,
        task_log_path: str,
        work_dir: str,
        result_file: Path,
        payload_dir: Path,
        counts: Counter[str],
        completed: int,
        total: int,
        status_file: Path,
        started_at: float,
    ) -> None:
        """Record task outputs, timing, and the final monitor state."""
        self._anomaly_report[task_name] = {
            "counts": dict(counts),
            "result_file": str(result_file),
            "payload_dir": str(payload_dir) if payload_dir.exists() else None,
            "task_log": str(Path(work_dir) / task_log_path),
        }
        self.logger.info("Response anomaly detection completed: %s", dict(counts))
        self.logger.info(
            "Response anomaly task time elapsed: %.2fs",
            time.perf_counter() - started_at,
        )
        self.logger.info("Task state is finish, exit loop")
        self._post_status(
            status_file,
            completed,
            total,
            counts,
            "response anomaly finished",
            "finish",
            task_name,
            task_log_path,
        )

    def _handle_detection_failure(
        self, status_file: Path, counts: Counter[str], exc: Exception
    ) -> None:
        """Mark every unfinished task as failed without masking the cause."""
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

    def _cleanup_detection_resources(
        self,
        log_handler: Optional[logging.FileHandler],
        payload_build_dir: Optional[Path],
        payload_writers: List[Any],
    ) -> None:
        """Best-effort cleanup for resources left by an interrupted task."""
        for writer in payload_writers:
            try:
                writer.close(write_manifest=False)
            except Exception as exc:
                self.logger.warning(
                    "Failed to close response anomaly payload writer: %s", exc
                )
        if payload_build_dir is not None and payload_build_dir.exists():
            self._remove_payload_build_dir(payload_build_dir)
        self._close_task_log(log_handler)

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

    @classmethod
    def _write_retained_payload(
        cls,
        payload_writer,
        retention: str,
        result: Dict[str, Any],
        record: Dict[str, Any],
        case_key: str,
        retained_payload_keys: set,
    ) -> None:
        if (
            payload_writer is None
            or case_key in retained_payload_keys
            or not isinstance(record.get("response_anomaly_payload"), dict)
            or not cls._should_retain_payload(retention, result)
        ):
            return
        payload_writer.write(record)
        retained_payload_keys.add(case_key)

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

    def _cleanup_stale_payload_build_dirs(self, payload_dir: Path) -> None:
        """Remove unpublished payload archives left by interrupted detection."""
        parent = payload_dir.parent
        if not parent.exists():
            return
        prefix = f".{payload_dir.name}.payload-build-"
        for candidate in parent.iterdir():
            if candidate.is_dir() and candidate.name.startswith(prefix):
                self._remove_payload_build_dir(candidate)

    def _remove_payload_build_dir(self, directory: Path) -> None:
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            return
        except OSError as exc:
            self.logger.warning(
                "Failed to clean response anomaly payload build directory %s: %s",
                directory,
                exc,
            )

    @staticmethod
    def _seed_payload_directory(
        source_dir: Path,
        destination_dir: Path,
        include_manifest: bool = False,
    ) -> None:
        """Seed a payload build with hard links, falling back to copies."""
        destination_dir.mkdir(parents=True, exist_ok=True)
        for source in source_dir.glob("part-*.jsonl.zst"):
            destination = destination_dir / source.name
            if destination.exists():
                destination.unlink()
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        destination_manifest = destination_dir / "payload_manifest.json"
        if include_manifest:
            shutil.copy2(
                source_dir / "payload_manifest.json",
                destination_manifest,
            )
        elif destination_manifest.exists():
            destination_manifest.unlink()

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
        if (
            has_mtype
            and has_tk2cat
            and Path(anomaly_cfg["msprobe_mtype_path"]).is_file()
            and Path(anomaly_cfg["msprobe_token2category_dir"]).is_dir()
        ):
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
        # Explicitly configured locations act as generation outputs: place the
        # generated resources there, reusing existing files untouched. An empty
        # msprobe_config_path keeps the msProbe built-in default (config.yaml
        # is model-agnostic and does not need to be generated).
        if not anomaly_cfg.get("msprobe_config_path"):
            merged.pop("msprobe_config_path", None)
        elif merged.get("msprobe_config_path") != generated["msprobe_config_path"]:
            merged["msprobe_config_path"] = self._place_generated_resource(
                Path(generated["msprobe_config_path"]),
                merged["msprobe_config_path"],
                is_dir=False,
            )
        if merged.get("msprobe_mtype_path") != generated["msprobe_mtype_path"]:
            merged["msprobe_mtype_path"] = self._place_generated_resource(
                Path(generated["msprobe_mtype_path"]),
                merged["msprobe_mtype_path"],
                is_dir=False,
            )
        if (
            merged.get("msprobe_token2category_dir")
            != generated["msprobe_token2category_dir"]
        ):
            merged["msprobe_token2category_dir"] = self._place_generated_resource(
                Path(generated["msprobe_token2category_dir"]),
                merged["msprobe_token2category_dir"],
                is_dir=True,
            )
        self.logger.info(
            "Auto-generated msProbe model config for [%s]:\n"
            "  config:         %s\n"
            "  mtype_config:   %s\n"
            "  token2category: %s",
            model_abbr,
            merged.get("msprobe_config_path"),
            merged.get("msprobe_mtype_path"),
            merged.get("msprobe_token2category_dir"),
        )
        return merged

    @staticmethod
    def _place_generated_resource(
        source: Path, target: str, is_dir: bool
    ) -> str:
        """Copy a generated msProbe resource to its configured location.

        Existing targets are reused untouched; missing parent directories are
        created as needed. Returns the configured target path.
        """
        target_path = Path(target)
        if not target_path.exists():
            if is_dir:
                shutil.copytree(source, target_path)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target_path)
        return str(target_path)

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
