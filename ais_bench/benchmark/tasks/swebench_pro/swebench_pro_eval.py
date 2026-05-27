import argparse
import concurrent.futures
import json
import os
import os.path as osp
import sys
import threading
import time
from pathlib import Path

from tqdm.auto import tqdm

from mmengine.config import Config, ConfigDict
from mmengine.utils import mkdir_or_exist

from ais_bench.benchmark.registry import TASKS
from ais_bench.benchmark.tasks.base import BaseTask, TaskStateManager
from ais_bench.benchmark.utils.config import build_dataset_from_cfg
from ais_bench.benchmark.utils.core.abbr import (
    get_infer_output_path,
    model_abbr_from_cfg,
    task_abbr_from_cfg,
    dataset_abbr_from_cfg,
)
from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import SWEBP_CODES
from ais_bench.benchmark.utils.logging.exceptions import (
    AISBenchValueError,
    FileOperationError,
)
from ais_bench.benchmark.tasks.swebench_pro.utils import (
    ensure_swebench_pro_docker_images,
    get_dockerhub_image_uri,
    eval_with_docker,
    list_swebench_pro_images,
    clean_swebench_pro_images,
)

KEY_INSTANCE_ID = "instance_id"
KEY_MODEL = "model_name_or_path"


def _filter_instances_from_preds(
    instances,
    predictions,
    model_abbr: str,
    dataset_abbr: str,
    run_dir,
    *,
    rewrite_reports=False,
    exclude_completed=True,
):
    filtered = []
    skipped = 0
    
    for inst in instances:
        instance_id = inst[KEY_INSTANCE_ID]
        if instance_id not in predictions:
            skipped += 1
            continue
        
        if exclude_completed:
            report_path = run_dir / f"{model_abbr}/{instance_id}/{dataset_abbr}_output.json"
            if report_path.exists() and not rewrite_reports:
                skipped += 1
                continue
        
        filtered.append(inst)
    
    return filtered, skipped


def isSolved(tests: list, fail_to_pass: str, pass_to_pass: str) -> bool:
    """Determine if the problem is solved
    
    Args:
        tests: Test execution results
        fail_to_pass: List of tests that need to be fixed and pass (string format)
        pass_to_pass: List of tests that need to remain passing (string format)
    
    Returns:
        True if problem is solved, False if not solved
    """
    try:
        passed_tests = {x["name"] for x in tests if x.get("status") == "PASSED"}
        
        f2p = set(eval(fail_to_pass)) if fail_to_pass and fail_to_pass.strip() else set()
        p2p = set(eval(pass_to_pass)) if pass_to_pass and pass_to_pass.strip() else set()
        
        return (f2p | p2p) <= passed_tests
    except Exception:
        return False


def _build_eval_report(
    full_instances: list,
    original_pred_path: str,
    run_eval_dir: Path,
    model_abbr: str,
    dataset_abbr: str,
) -> dict:
    """Build custom evaluation result report
    
    Args:
        full_instances: Full dataset instances
        original_pred_path: Inference result file path
        run_eval_dir: Root directory for evaluation results
        model_abbr: Model abbreviation
        dataset_abbr: Dataset abbreviation
    
    Returns:
        report dictionary
    """
    with open(original_pred_path) as f:
        predictions = json.load(f)
    
    build_patch_ids = []
    empty_patch_ids = []
    
    for instance_id, pred in predictions.items():
        if not isinstance(pred, dict):
            continue
        model_patch = pred.get("model_patch", "") or pred.get("patch", "")
        if model_patch and model_patch.strip():
            build_patch_ids.append(instance_id)
        else:
            empty_patch_ids.append(instance_id)
    
    resolved_ids = []
    unresolved_ids = []
    
    for instance in full_instances:
        instance_id = instance["instance_id"]
        
        eval_result_path = run_eval_dir / model_abbr / instance_id / f"{dataset_abbr}_output.json"
        
        if not eval_result_path.exists():
            unresolved_ids.append(instance_id)
            continue
        
        try:
            with open(eval_result_path) as f:
                eval_output = json.load(f)
        except (json.JSONDecodeError, OSError):
            unresolved_ids.append(instance_id)
            continue
        
        fail_to_pass = instance.get("fail_to_pass", "")
        pass_to_pass = instance.get("pass_to_pass", "")
        
        tests = eval_output.get("tests", [])
        if not isinstance(tests, list):
            unresolved_ids.append(instance_id)
            continue
        
        if isSolved(tests, fail_to_pass, pass_to_pass):
            resolved_ids.append(instance_id)
        else:
            unresolved_ids.append(instance_id)
    
    report = {
        "total_instances_num": len(full_instances),
        "total_prediction_num": len(predictions),
        "build_patch_instances_num": len(build_patch_ids),
        "empty_patch_instances_num": len(empty_patch_ids),
        "eval_resolved_instances_num": len(resolved_ids),
        "eval_unresolved_instances_num": len(unresolved_ids),
        "empty_patch_instances_ids": sorted(empty_patch_ids),
        "unresolved_instances_ids": sorted(unresolved_ids),
    }

    total = report.get("total_instances_num") or 0
    resolved = report.get("eval_resolved_instances_num") or 0

    accuracy = round((resolved / total * 100.0), 2) if total else 0.0

    report["accuracy"] = accuracy

    return report


class _SWEBenchProEvalProgressManager:
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
                "progress_description": "SWE-bench Pro eval",
                "other_kwargs": {"current": instance_id},
            }
        )


    def on_instance_end(self, instance_id: str, pass_flag: bool) -> None:
        self._finish_count += 1
        if pass_flag:
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
                "progress_description": "SWE-bench Pro eval",
                "other_kwargs": {"current": instance_id},
            }
        )


@TASKS.register_module()
class SWEBenchProEvalTask(BaseTask):
    name_prefix = "SWEBenchProEval"
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
        self.logger.info("SWEBenchProEvalTask %s", task_abbr_from_cfg(self.cfg))

        dataset_cfg = self.dataset_cfgs[0]
        model_abbr = model_abbr_from_cfg(self.model_cfg)
        dataset_abbr = dataset_abbr_from_cfg(dataset_cfg)

        # Load dataset via build_dataset_from_cfg
        dataset = build_dataset_from_cfg(
            dataset_cfg, task_state_manager=task_state_manager
        )
        test_data = dataset.test
        if hasattr(test_data, "__iter__") and not isinstance(test_data, (list, dict)):
            full_instances = list(test_data)
        else:
            full_instances = [test_data[i] for i in range(len(test_data))]

        # Get scripts_dir config
        scripts_dir = dataset_cfg.get("swebp_scripts_dir", "")
        if not scripts_dir or not os.path.isdir(scripts_dir):
            raise AISBenchValueError(
                SWEBP_CODES.UNKNOWN_ERROR,
                f"scripts_dir must be configured in dataset config: {scripts_dir}"
            )
        scripts_dir_abs = Path(scripts_dir).resolve()
        self.logger.info("scripts_dir_abs: %s", scripts_dir_abs)

        # Get docker_dir config
        docker_dir = dataset_cfg.get("swebp_docker_dir", "")
        if not docker_dir or not os.path.isdir(docker_dir):
            raise AISBenchValueError(
                SWEBP_CODES.UNKNOWN_ERROR,
                f"docker_dir must be configured in dataset config: {docker_dir}"
            )
        docker_dir_abs = Path(docker_dir).resolve()
        self.logger.info("docker_dir_abs: %s", docker_dir_abs)

        # Load predictions
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
                    SWEBP_CODES.PREDICTIONS_FILE_NOT_FOUND,
                    f"Predictions file not found: {pred_path} (or {pred_path_fallback}). Run infer first."
                )

        original_pred_path = pred_path
        with open(pred_path) as f:
            raw_preds = json.load(f)
        
        if isinstance(raw_preds, dict):
            predictions = {p[KEY_INSTANCE_ID]: p for p in raw_preds.values() if isinstance(p, dict) and KEY_INSTANCE_ID in p}
        else:
            predictions = {p[KEY_INSTANCE_ID]: p for p in raw_preds if isinstance(p, dict) and KEY_INSTANCE_ID in p}

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

        run_eval_dir = Path(run_log_dir)
        run_eval_dir.mkdir(parents=True, exist_ok=True)

        max_workers = self.model_cfg.get("batch_size", 1)
        timeout = self.model_cfg.get("timeout", 7200)

        # Filter instances (align with get_dataset_from_preds)
        instances_to_eval, n_skipped = _filter_instances_from_preds(
            full_instances,
            predictions,
            model_abbr,
            dataset_abbr,
            run_eval_dir,
            rewrite_reports=False,
            exclude_completed=True,
        )

        if n_skipped:
            self.logger.info("%d instances already run, skipping", n_skipped)

        if not instances_to_eval:
            self.logger.info("No instances to evaluate")
            return

        # Build instance index
        instance_index = {inst["instance_id"]: inst for inst in instances_to_eval}

        # Record existing Docker images and ensure required images
        try:
            import docker
        except Exception:
            raise RuntimeError("docker SDK is not installed. Install via 'pip install docker' or run without --use_local_docker")
        docker_client = docker.from_env()
        prior_images = list_swebench_pro_images(docker_client)
        
        ensure_swebench_pro_docker_images(
            instances_to_eval,
            self.logger,
            get_dockerhub_image_uri,
            task_label="eval",
        )

        # State management
        task_state_manager.update_task_state(
            {
                "status": "eval",
                "finish_count": 0,
                "total_count": len(instances_to_eval),
                "resolved_count": 0,
                "accuracy": 0.0,
                "progress_description": "SWE-bench Pro eval",
            }
        )

        # Core evaluation with progress tracking
        progress_manager = _SWEBenchProEvalProgressManager(task_state_manager, len(instances_to_eval))
        lock = threading.Lock()
        stats = {"resolved": 0, "unresolved": 0}
    
        def run_eval_with_progress(patch, instance, report_dir, scripts_dir_abs, docker_dir_abs, logger, prefix="", docker_client=None, timeout=7200):
            instance_id = instance[KEY_INSTANCE_ID]
            progress_manager.on_instance_start(instance_id)
            try:
                return eval_with_docker(patch, instance, report_dir, scripts_dir_abs, docker_dir_abs, logger, prefix, docker_client, timeout)
            except Exception as e:
                self.logger.error(
                    SWEBP_CODES.HARNESS_RUNTIME_FAILED,
                    "Error evaluating pred %s: %s",
                    instance_id,
                    e,
                    exc_info=True,
                )
                self.logger.info("Evaluation for %s generated an exception: %s", instance_id, e)
                return None

        self.logger.info("Running %d instances...", len(instances_to_eval))

        pbar = None
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_pred = {
                    executor.submit(
                        run_eval_with_progress,
                        predictions[inst[KEY_INSTANCE_ID]].get("model_patch", predictions[inst[KEY_INSTANCE_ID]].get("patch", "")),
                        inst,
                        report_dir,
                        scripts_dir_abs,
                        docker_dir_abs,
                        self.logger,
                        predictions[inst[KEY_INSTANCE_ID]].get(KEY_MODEL, ""),
                        docker_client=docker_client,
                        timeout=timeout,
                    ): predictions[inst[KEY_INSTANCE_ID]] for inst in instances_to_eval
                }
                self.logger.info("Starting pbar for %d futures", len(future_to_pred))
                pbar = tqdm(concurrent.futures.as_completed(future_to_pred), total=len(future_to_pred), postfix=stats, unit="instance")

                for future in pbar:
                    pred = future_to_pred[future]
                    instance_id = pred[KEY_INSTANCE_ID]
                    instance = instance_index.get(instance_id)
                    self.logger.debug("Processing instance: %s", instance_id)
                    
                    if instance is None:
                        self.logger.error(SWEBP_CODES.HARNESS_RUNTIME_FAILED, "Instance %s not found in instance_index", instance_id)
                        with lock:
                            stats["unresolved"] += 1
                            pbar.set_postfix(stats)
                            pbar.update()
                        progress_manager.on_instance_end(instance_id, False)
                        continue
                    
                    try:
                        output = future.result()  # Get raw result
                        self.logger.debug("Output for %s: %s", instance_id, output)
                        pass_flag = False
                        if output is None:
                            self.logger.info("Evaluation for %s returned None", instance_id)
                        elif output.get("error") == "timeout":
                            self.logger.info("Evaluation for %s timed out after %ds", instance_id, timeout)
                            self.logger.info("Error message: %s", output.get("message", "Unknown"))
                        else:
                            try:
                                tests = output.get("tests", [])
                                if not isinstance(tests, list):
                                    raise ValueError(f"tests is not a list: {type(tests)}")
                                                                
                                fail_to_pass = instance.get("fail_to_pass", "")
                                pass_to_pass = instance.get("pass_to_pass", "")        
                                pass_flag = isSolved(tests, fail_to_pass, pass_to_pass) 
                            except Exception as e:
                                self.logger.info("Data error evaluating %s: %s", instance_id, e)
                                pass_flag = False
                        with lock:
                            if pass_flag:
                                stats["resolved"] += 1
                            else:
                                stats["unresolved"] += 1
                            pbar.set_postfix(stats)
                            pbar.update()
                            
                        progress_manager.on_instance_end(instance_id, pass_flag)
                        
                    except Exception as e:
                        self.logger.info("Exception processing %s: %s", instance_id, e)
                        with lock:
                            stats["unresolved"] += 1
                            pbar.set_postfix(stats)
                            pbar.update()
                        progress_manager.on_instance_end(instance_id, False)
        finally:
            if pbar is not None:
                pbar.close()
        
        self.logger.info("All instances run.")
 
        self.logger.info("Cleaning up SWE-bench Pro images...")
        clean_swebench_pro_images(docker_client, prior_images, self.logger)
        self.logger.info("Image cleanup completed.")

        # 7. Build custom eval report and write to file
        report = _build_eval_report(
            full_instances=full_instances,
            original_pred_path=original_pred_path,
            run_eval_dir=run_eval_dir,
            model_abbr=model_abbr,
            dataset_abbr=dataset_abbr,
        )
        
        self.logger.info("Accuracy: %.2f", report.get("accuracy", 0))

        report_path = run_eval_dir / f"{model_abbr}_{dataset_abbr}_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info("Eval report saved to: %s", report_path)


def parse_args():
    parser = argparse.ArgumentParser(description="SWEBenchPro Eval")
    parser.add_argument("config", help="Config file path")
    return parser.parse_args()


if __name__ == "__main__":
    logger = AISLogger()
    args = parse_args()
    cfg = Config.fromfile(args.config)
    task_state_manager = TaskStateManager(
        tmp_path=os.path.join(cfg["work_dir"], "status_tmp"),
        task_name=task_abbr_from_cfg(cfg),
        is_debug=cfg["cli_args"].get("debug", False),
    )
    manager_t = threading.Thread(target=task_state_manager.launch, args=())
    manager_t.start()
    task_state_manager.update_task_state(
        {
            "status": "start",
            "task_log_path": os.path.join("logs/eval/", f"{task_abbr_from_cfg(cfg)}.out"),
        }
    )
    start_time = time.perf_counter()
    try:
        task = SWEBenchProEvalTask(cfg)
        task.run(task_state_manager)
    except KeyboardInterrupt:
        task_state_manager.update_task_state({"status": "cancelled"})
        logger.info("Evaluation interrupted by user")
        os._exit(130)
    except Exception as e:
        task_state_manager.update_task_state({"status": "error"})
        raise
    end_time = time.perf_counter()
    logger.info("SWEBenchPro eval time: %.2fs", end_time - start_time)
    task_state_manager.update_task_state({"status": "finish"})
    manager_t.join()