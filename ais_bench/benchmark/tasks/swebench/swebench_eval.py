import argparse
import json
import os
import os.path as osp
import platform
import signal
import sys
import threading
import time
from pathlib import Path

from tqdm.auto import tqdm

if platform.system() == "Linux":
    import resource

from mmengine.config import Config, ConfigDict
from mmengine.utils import mkdir_or_exist

from ais_bench.benchmark.registry import TASKS
from ais_bench.benchmark.tasks.base import BaseTask, TaskStateManager
from ais_bench.benchmark.utils.config import build_dataset_from_cfg
from ais_bench.benchmark.utils.core.abbr import (
    dataset_abbr_from_cfg,
    get_infer_output_path,
    model_abbr_from_cfg,
    task_abbr_from_cfg,
)
from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import SWEB_CODES
from ais_bench.benchmark.utils.logging.exceptions import (
    AISBenchDataContentError,
    AISBenchImportError,
    FileOperationError,
)
from ais_bench.benchmark.tasks.swebench.utils import (
    cleanup_swebench_containers,
    ensure_swebench_docker_images,
)


# Swebench harness constants (must match harness)
KEY_INSTANCE_ID = "instance_id"
KEY_MODEL = "model_name_or_path"
KEY_PREDICTION = "model_patch"
LOG_REPORT = "report.json"
LOG_TEST_OUTPUT = "test_output.txt"


class _SWEBenchEvalProgressManager:
    """Progress manager that forwards case-level progress to TaskStateManager."""

    def __init__(self, task_state_manager: TaskStateManager, total: int):
        self._tsm = task_state_manager
        self._total = total
        self._finish_count = 0
        self._resolved_count = 0

    def on_instance_start(self, instance_id: str) -> None:
        self._tsm.update_task_state(
            {
                "status": "eval",
                "finish_count": self._finish_count,
                "total_count": self._total,
                "resolved_count": self._resolved_count,
                "accuracy": (
                    round(self._resolved_count / self._finish_count * 100.0, 2)
                    if self._finish_count else 0.0
                ),
                "progress_description": "SWE-bench harness",
                "other_kwargs": {"current": instance_id},
            }
        )

    def on_instance_end(self, instance_id: str, result: dict) -> None:
        self._finish_count += 1
        if result.get("resolved"):
            self._resolved_count += 1
        accuracy = (
            round(self._resolved_count / self._finish_count * 100.0, 2)
            if self._finish_count else 0.0
        )
        self._tsm.update_task_state(
            {
                "status": "eval",
                "finish_count": self._finish_count,
                "total_count": self._total,
                "resolved_count": self._resolved_count,
                "accuracy": accuracy,
                "progress_description": "SWE-bench harness",
                "other_kwargs": {"current": instance_id},
            }
        )


def _make_run_report_fast(
    predictions: dict,
    full_instances: list,
    run_id: str,
    run_log_dir: Path,
    report_output_path: Path,
    logger: AISLogger,
) -> Path:
    """Lightweight report generator: only iterates over predictions, no Docker/client."""
    dataset_ids = {i[KEY_INSTANCE_ID] for i in full_instances}
    prediction_ids = set(predictions.keys())
    incomplete_ids = dataset_ids - prediction_ids
    empty_patch_ids = {
        k
        for k, v in predictions.items()
        if v.get(KEY_PREDICTION) in ("", None)
    }

    completed_ids = set()
    resolved_ids = set()
    unresolved_ids = set()
    error_ids = set()

    for instance_id in prediction_ids:
        if instance_id in empty_patch_ids:
            continue
        pred = predictions[instance_id]
        report_file = (
            run_log_dir
            / run_id
            / pred[KEY_MODEL].replace("/", "__")
            / instance_id
            / LOG_REPORT
        )
        if not report_file.exists():
            error_ids.add(instance_id)
            continue
        completed_ids.add(instance_id)
        try:
            content = report_file.read_text().strip()
            if not content:
                error_ids.add(instance_id)
                completed_ids.discard(instance_id)
                continue
            report = json.loads(content)
            if report.get(instance_id, {}).get("resolved"):
                resolved_ids.add(instance_id)
            else:
                unresolved_ids.add(instance_id)
        except (json.JSONDecodeError, KeyError):
            error_ids.add(instance_id)
            completed_ids.discard(instance_id)

    report = {
        "total_instances": len(full_instances),
        "submitted_instances": len(predictions),
        "completed_instances": len(completed_ids),
        "resolved_instances": len(resolved_ids),
        "unresolved_instances": len(unresolved_ids),
        "empty_patch_instances": len(empty_patch_ids),
        "error_instances": len(error_ids),
        "completed_ids": sorted(completed_ids),
        "incomplete_ids": sorted(incomplete_ids),
        "empty_patch_ids": sorted(empty_patch_ids),
        "submitted_ids": sorted(prediction_ids),
        "resolved_ids": sorted(resolved_ids),
        "unresolved_ids": sorted(unresolved_ids),
        "error_ids": sorted(error_ids),
        "schema_version": 2,
    }
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(
        "Report: %d resolved, %d unresolved, %d errors (of %d submitted)",
        len(resolved_ids),
        len(unresolved_ids),
        len(error_ids),
        len(predictions),
    )
    return report_output_path


def _filter_instances_from_preds(
    full_instances: list,
    predictions: dict,
    run_id: str,
    run_log_dir: Path,
    rewrite_reports: bool = False,
    exclude_completed: bool = True,
) -> tuple[list, int]:
    """Filter instances to those with predictions, not completed, and non-empty patch."""
    prediction_ids = set(predictions.keys())
    dataset_ids = {i[KEY_INSTANCE_ID] for i in full_instances}

    if prediction_ids - dataset_ids:
        raise AISBenchDataContentError(
            SWEB_CODES.PREDICTION_IDS_NOT_FOUND,
            "Some prediction IDs not found in dataset!"
            f"\nMissing IDs:\n{' '.join(sorted(prediction_ids - dataset_ids))}"
        )

    if rewrite_reports:
        test_output_ids = set()
        for instance in full_instances:
            if instance[KEY_INSTANCE_ID] not in predictions:
                continue
            pred = predictions[instance[KEY_INSTANCE_ID]]
            test_output_file = (
                run_log_dir
                / run_id
                / pred[KEY_MODEL].replace("/", "__")
                / pred[KEY_INSTANCE_ID]
                / LOG_TEST_OUTPUT
            )
            if test_output_file.exists():
                test_output_ids.add(instance[KEY_INSTANCE_ID])
        return (
            [
                i
                for i in full_instances
                if i[KEY_INSTANCE_ID] in prediction_ids
                and i[KEY_INSTANCE_ID] in test_output_ids
            ],
            0,
        )

    completed_ids = set()
    for instance in full_instances:
        if instance[KEY_INSTANCE_ID] not in prediction_ids:
            continue
        pred = predictions[instance[KEY_INSTANCE_ID]]
        report_file = (
            run_log_dir
            / run_id
            / pred[KEY_MODEL].replace("/", "__")
            / pred[KEY_INSTANCE_ID]
            / LOG_REPORT
        )
        if report_file.exists():
            completed_ids.add(instance[KEY_INSTANCE_ID])

    if completed_ids and exclude_completed:
        full_instances = [i for i in full_instances if i[KEY_INSTANCE_ID] not in completed_ids]

    empty_patch_ids = {
        k
        for k, v in predictions.items()
        if v.get(KEY_PREDICTION) in ("", None)
    }

    n_skipped = len(completed_ids) if exclude_completed else 0
    return (
        [
            i
            for i in full_instances
            if i[KEY_INSTANCE_ID] in prediction_ids
            and i[KEY_INSTANCE_ID] not in empty_patch_ids
        ],
        n_skipped,
    )


@TASKS.register_module()
class SWEBenchEvalTask(BaseTask):
    """SWEBench Evaluation Task.

    Evaluates SWE-bench predictions using the official harness and writes
    results to work_dir/results. Uses build_dataset_from_cfg for dataset
    loading (same as infer), calls run_instance directly for case-level
    progress reporting. Instance images are ensured via shared Docker
    inspect/pull helpers (same as infer); the harness does not build images here.
    """

    name_prefix = "SWEBenchEval"
    log_subdir = "logs/eval"
    output_subdir = "results"

    def __init__(self, cfg: ConfigDict):
        super().__init__(cfg)

    def get_command(self, cfg_path: str, template: str) -> str:
        sys.path.append(os.getcwd())
        script_path = __file__
        python = sys.executable
        command = f"{python} {script_path} {cfg_path}"
        return template.format(task_cmd=command)

    def run(self, task_state_manager: TaskStateManager):
        self.task_state_manager = task_state_manager
        self.logger.info("SWEBenchEvalTask %s", task_abbr_from_cfg(self.cfg))

        try:
            import swebench.harness.constants as harness_constants
            import swebench.harness.reporting as harness_reporting
            import swebench.harness.run_evaluation as run_eval_mod
            from swebench.harness.docker_utils import clean_images, list_images, should_remove
            from swebench.harness.run_evaluation import run_instance
            from swebench.harness.test_spec.test_spec import make_test_spec
            from swebench.harness.utils import run_threadpool
        except ImportError as e:
            raise AISBenchImportError(
                SWEB_CODES.SWEBENCH_HARNESS_IMPORT_ERROR,
                "SWEBenchEvalTask requires the SWE-bench harness. "
                "Install from: https://github.com/SWE-bench/SWE-bench"
            ) from e

        dataset_cfg = self.dataset_cfgs[0]
        model_abbr = model_abbr_from_cfg(self.model_cfg)
        dataset_abbr = dataset_abbr_from_cfg(dataset_cfg)

        # 1. Load dataset via build_dataset_from_cfg (same as swebench_infer)
        dataset = build_dataset_from_cfg(
            dataset_cfg, task_state_manager=task_state_manager
        )
        test_data = dataset.test
        if hasattr(test_data, "__iter__") and not isinstance(test_data, (list, dict)):
            full_instances = list(test_data)
        else:
            full_instances = [test_data[i] for i in range(len(test_data))]

        # 2. Load predictions
        pred_path = get_infer_output_path(
            self.model_cfg,
            dataset_cfg,
            osp.join(self.work_dir, "predictions"),
            file_extension="json",
        )
        if not osp.isfile(pred_path):
            pred_path_fallback = osp.join(
                osp.dirname(pred_path),
                osp.splitext(osp.basename(pred_path))[0],
                "preds.json",
            )
            if osp.isfile(pred_path_fallback):
                pred_path = pred_path_fallback
                self.logger.info("Using predictions from %s", pred_path)
            else:
                raise FileOperationError(
                    SWEB_CODES.PREDICTIONS_FILE_NOT_FOUND,
                    f"Predictions file not found: {pred_path} (or {pred_path_fallback}). Run infer first."
                )

        original_pred_path = pred_path
        with open(pred_path) as f:
            raw_preds = json.load(f)
        if isinstance(raw_preds, dict):
            predictions = {p[KEY_INSTANCE_ID]: p for p in raw_preds.values() if isinstance(p, dict)}
        else:
            predictions = {p[KEY_INSTANCE_ID]: p for p in raw_preds if isinstance(p, dict)}

        for p in predictions.values():
            if isinstance(p, dict):
                p[KEY_MODEL] = dataset_abbr

        run_log_dir = Path(self.work_dir) / self.output_subdir
        out_path = get_infer_output_path(
            self.model_cfg,
            dataset_cfg,
            osp.join(self.work_dir, self.output_subdir),
            file_extension="json",
        )
        mkdir_or_exist(osp.dirname(out_path))
        report_dir = osp.dirname(out_path)

        # Set harness log dirs (run_instance reads from these modules)
        run_eval_dir = Path(run_log_dir)
        run_eval_dir.mkdir(parents=True, exist_ok=True)
        harness_constants.RUN_EVALUATION_LOG_DIR = run_eval_dir
        harness_reporting.RUN_EVALUATION_LOG_DIR = run_eval_dir
        run_eval_mod.RUN_EVALUATION_LOG_DIR = run_eval_dir

        run_id = model_abbr
        max_workers = self.model_cfg.get("batch_size", 1)
        timeout = 7200
        namespace = "swebench"
        rewrite_reports = False
        cache_level = "env"
        clean = False
        force_rebuild = False

        # 3. Filter instances (align with get_dataset_from_preds)
        instances, n_skipped = _filter_instances_from_preds(
            full_instances,
            predictions,
            run_id,
            run_eval_dir,
            rewrite_reports=rewrite_reports,
            exclude_completed=True,
        )

        if n_skipped:
            self.logger.info("%d instances already run, skipping", n_skipped)

        task_state_manager.update_task_state(
            {
                "status": "eval",
                "finish_count": 0,
                "total_count": len(instances),
                "resolved_count": 0,
                "accuracy": 0.0,
                "progress_description": "SWE-bench harness",
            }
        )

        def _on_interrupt(signum, frame):
            self.logger.info(
                "Interrupted: cleaning up eval containers (sweb.eval.*)..."
            )
            cleanup_swebench_containers()
            os._exit(128 + (2 if signum == signal.SIGINT else 15))

        old_sigint = signal.signal(signal.SIGINT, _on_interrupt)
        old_sigterm = signal.signal(signal.SIGTERM, _on_interrupt)
        harness_exit = 0

        try:
            if platform.system() == "Linux":
                resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))

            client = __import__("docker").from_env()
            existing_images = list_images(client)

            if not instances:
                self.logger.info("No instances to run.")
            else:
                test_specs = [
                    make_test_spec(inst, namespace=namespace)
                    for inst in instances
                ]
                ensure_swebench_docker_images(
                    test_specs,
                    self.logger,
                    lambda spec: spec.instance_image_key,
                    task_label="eval",
                )
                instance_image_ids = {x.instance_image_key for x in test_specs}
                existing_instance_images = {
                    tag
                    for i in client.images.list(all=True)
                    for tag in (i.tags or [])
                    if tag in instance_image_ids
                }

                payloads = []
                for test_spec in test_specs:
                    payloads.append(
                        (
                            test_spec,
                            predictions[test_spec.instance_id],
                            should_remove(
                                test_spec.instance_image_key,
                                cache_level,
                                clean,
                                existing_instance_images,
                            ),
                            force_rebuild,
                            client,
                            run_id,
                            timeout,
                            rewrite_reports,
                        )
                    )

                progress_manager = _SWEBenchEvalProgressManager(
                    task_state_manager, len(payloads)
                )
                self.logger.info("Running %d instances...", len(payloads))
                lock = threading.Lock()
                stats = {"resolved": 0, "unresolved": 0, "error": 0}
                pbar = tqdm(
                    total=len(payloads),
                    desc="Evaluation",
                    postfix=stats,
                    unit="instance",
                )

                def run_eval_with_progress(*args):
                    test_spec, pred, *_ = args
                    instance_id = test_spec.instance_id
                    progress_manager.on_instance_start(instance_id)
                    result = run_instance(*args)
                    with lock:
                        if result.get("completed"):
                            if result.get("resolved"):
                                stats["resolved"] += 1
                            else:
                                stats["unresolved"] += 1
                        else:
                            stats["error"] += 1
                        pbar.set_postfix(stats)
                        pbar.update()
                        progress_manager.on_instance_end(instance_id, result)
                    return result

                try:
                    run_threadpool(run_eval_with_progress, payloads, max_workers)
                finally:
                    pbar.close()
                self.logger.info("All instances run.")

                clean_images(client, existing_images, cache_level, clean)

            harness_report_path = Path(report_dir) / f"{dataset_abbr}.{run_id}.json"
            _make_run_report_fast(
                predictions,
                full_instances,
                run_id,
                run_eval_dir,
                harness_report_path,
                self.logger,
            )

        except SystemExit as e:
            harness_exit = e.code if e.code is not None else 1
        except Exception as e:
            self.logger.error(SWEB_CODES.HARNESS_RUNTIME_FAILED, "Harness failed: %s", e, exc_info=True)
            harness_exit = 1
        finally:
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)

        harness_report_path = Path(report_dir) / f"{dataset_abbr}.{run_id}.json"
        if harness_report_path.exists():
            try:
                with open(harness_report_path) as f:
                    results = json.load(f)
                total = results.get("total_instances") or 0
                submitted = results.get("submitted_instances") or 0
                resolved = results.get("resolved_instances") or 0
                accuracy = round((resolved / total * 100.0), 2) if total else 0.0
                submitted_accuracy = (
                    round((resolved / submitted * 100.0), 2) if submitted else 0.0
                )
                results["harness_exit_code"] = harness_exit
                results["predictions_path"] = original_pred_path
                results["run_id"] = run_id
                ordered_results = {
                    "accuracy": accuracy,
                    "submitted_accuracy": submitted_accuracy,
                    **results,
                }
                with open(out_path, "w") as f:
                    json.dump(ordered_results, f, indent=2)
                try:
                    os.unlink(harness_report_path)
                except OSError:
                    pass
            except (json.JSONDecodeError, OSError) as e:
                self.logger.warning(
                    "Could not merge harness report into %s: %s", out_path, e
                )
                results = {
                    "harness_exit_code": harness_exit,
                    "predictions_path": original_pred_path,
                    "run_id": run_id,
                }
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)
        else:
            results = {
                "harness_exit_code": harness_exit,
                "predictions_path": original_pred_path,
                "run_id": run_id,
            }
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

        if harness_exit != 0:
            self.logger.warning("Harness exited with code %s", harness_exit)


def parse_args():
    parser = argparse.ArgumentParser(description="SWEBench Eval")
    parser.add_argument("config", help="Config file path")
    return parser.parse_args()


if __name__ == "__main__":
    logger = AISLogger()
    args = parse_args()
    cfg = Config.fromfile(args.config)
    task_state_manager = TaskStateManager(
        tmp_path=os.path.join(cfg["work_dir"], "status_tmp"),
        task_name=task_abbr_from_cfg(cfg),
        is_debug=cfg["cli_args"]["debug"],
    )
    manager_t = threading.Thread(target=task_state_manager.launch, args=())
    manager_t.start()
    task_state_manager.update_task_state(
        {
            "status": "start",
            "task_log_path": os.path.join(
                "logs/eval/", f"{task_abbr_from_cfg(cfg)}.out"
            ),
        }
    )
    start_time = time.perf_counter()
    try:
        task = SWEBenchEvalTask(cfg)
        task.run(task_state_manager)
    except Exception as e:
        task_state_manager.update_task_state({"status": "error"})
        raise
    end_time = time.perf_counter()
    logger.info("SWEBench eval time: %.2fs", end_time - start_time)
    task_state_manager.update_task_state({"status": "finish"})
    manager_t.join()
