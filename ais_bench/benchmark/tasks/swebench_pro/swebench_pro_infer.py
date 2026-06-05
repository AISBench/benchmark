import argparse
import concurrent.futures
import json
import os
import os.path as osp
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional, Tuple
import yaml

from mmengine.config import Config, ConfigDict
from mmengine.utils import mkdir_or_exist

from ais_bench.benchmark.registry import TASKS
from ais_bench.benchmark.tasks.base import BaseTask, TaskStateManager
from ais_bench.benchmark.utils.config import build_dataset_from_cfg
from ais_bench.benchmark.utils.core.abbr import (
    get_infer_output_path,
    model_abbr_from_cfg,
    task_abbr_from_cfg,
)
from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import SWEBP_CODES
from ais_bench.benchmark.utils.logging.exceptions import (
    AISBenchImportError,
    AISBenchValueError,
)
from ais_bench.benchmark.tasks.swebench_pro.utils import (
    cleanup_swebench_pro_containers,
    ensure_swebench_pro_docker_images,
    get_dockerhub_image_uri,
    merge_nested_dicts,
)


def _get_minisweagent_config(model_cfg: ConfigDict) -> dict:
    model_name = model_cfg.get("model") or model_cfg.get("model_name") or ""
    if model_cfg.get("url") and model_name:
        model_name = f"hosted_vllm/{model_name}"
    model_type = (
        getattr(model_cfg.get("type"), "__name__", None)
        or (model_cfg.get("type", "") if isinstance(model_cfg.get("type"), str) else "")
    )
    if isinstance(model_type, str):
        model_type = model_type.split(".")[-1]
    model_kwargs = dict(model_cfg.get("generation_kwargs", {}))
    if model_cfg.get("api_key"):
        model_kwargs["api_key"] = model_cfg["api_key"]
    if model_cfg.get("url"):
        model_kwargs["api_base"] = model_cfg["url"]
    model_class = "litellm"
    if "openrouter" in (model_type or "").lower() or "openrouter" in (str(model_cfg.get("type", ""))).lower():
        model_class = "openrouter"
    model_dict = {
        "model_name": model_name,
        "model_class": model_class,
        "model_kwargs": model_kwargs,
        "cost_tracking": "ignore_errors",
    }
    return {"model": model_dict}


class _AISBenchProgressManager:
    def __init__(self, task_state_manager: TaskStateManager, total: int):
        self._tsm = task_state_manager
        self._total = total
        self._finish_count = 0

    def on_instance_start(self, instance_id: str) -> None:
        self._tsm.update_task_state(
            {
                "status": "inferencing",
                "finish_count": self._finish_count,
                "total_count": self._total,
                "progress_description": "SWEBenchPro infer",
                "other_kwargs": {"current": instance_id},
            }
        )

    def update_instance_status(self, instance_id: str, message: str) -> None:
        self._tsm.update_task_state(
            {
                "status": "inferencing",
                "finish_count": self._finish_count,
                "total_count": self._total,
                "progress_description": "SWEBenchPro infer",
                "other_kwargs": {"current": instance_id, "message": message},
            }
        )

    def on_instance_end(self, instance_id: str, exit_status: str = None) -> None:
        self._finish_count += 1
        self._tsm.update_task_state(
            {
                "status": "inferencing",
                "finish_count": self._finish_count,
                "total_count": self._total,
                "progress_description": "SWEBenchPro infer",
            }
        )

    def on_uncaught_exception(self, instance_id: str, exception: Exception) -> None:
        self.on_instance_end(instance_id, f"Uncaught {type(exception).__name__}")


class _CompositeProgressManager:
    """Forwards progress calls to multiple delegates (e.g. TaskStateManager + Rich dashboard)."""

    def __init__(self, *delegates: Any):
        self._delegates = [d for d in delegates if d is not None]

    @property
    def n_completed(self) -> int:
        # Return n_completed from RunBatchProgressManager
        return self._delegates[1].n_completed

    def on_instance_start(self, instance_id: str) -> None:
        for d in self._delegates:
            d.on_instance_start(instance_id)

    def update_instance_status(self, instance_id: str, message: str) -> None:
        for d in self._delegates:
            d.update_instance_status(instance_id, message)

    def on_instance_end(self, instance_id: str, exit_status: str = None) -> None:
        for d in self._delegates:
            d.on_instance_end(instance_id, exit_status)

    def on_uncaught_exception(self, instance_id: str, exception: Exception) -> None:
        for d in self._delegates:
            d.on_uncaught_exception(instance_id, exception)


def _make_swebench_pro_progress_manager(
    task_state_manager: TaskStateManager,
    num_instances: int,
    out_dir: str,
) -> Tuple[Any, Optional[Any]]:
    """Build progress manager and optional Rich live display.

    Returns:
        (progress_manager, live_render_group or None).
        When live_render_group is not None, caller should wrap execution in
        Live(live_render_group, refresh_per_second=4).
    """
    tsm_manager = _AISBenchProgressManager(task_state_manager, num_instances)
    try:
        from minisweagent.run.extra.utils.batch_progress import (
            RunBatchProgressManager,
        )
        from rich.live import Live

        run_batch_manager = RunBatchProgressManager(num_instances, yaml_report_path=out_dir / "exit_statuses.yaml")
        composite = _CompositeProgressManager(tsm_manager, run_batch_manager)
        return composite, run_batch_manager.render_group
    except ImportError:
        return tsm_manager, None


@TASKS.register_module()
class SWEBenchProInferTask(BaseTask):
    name_prefix = "SWEBenchProInfer"
    log_subdir = "logs/infer"
    output_subdir = "predictions"

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
        self.logger.info("SWEBenchProInferTask %s", task_abbr_from_cfg(self.cfg))

        try:
            from minisweagent.run.extra.run_batch import (
                process_instance,
                RunBatchConfig,
            )
            from minisweagent.config import get_config_path
            from minisweagent.run.utils.batch_instances import BatchInstance
        except ImportError as e:
            raise AISBenchImportError(
                SWEBP_CODES.MINISWEAGENT_IMPORT_ERROR,
                "SWEBenchProInferTask requires mini-swe-agent. "
                "Install with: pip install mini-swe-agent"
            ) from e

        dataset_cfg = self.dataset_cfgs[0]
        dataset = build_dataset_from_cfg(
            dataset_cfg, task_state_manager=task_state_manager
        )
        test_data = dataset.test
        if hasattr(test_data, "__iter__") and not isinstance(test_data, (list, dict)):
            instances = list(test_data)
        else:
            instances = [test_data[i] for i in range(len(test_data))]

        self.logger.info(f"instances len={len(instances)}")

        model_abbr = model_abbr_from_cfg(self.model_cfg)
        pred_root = osp.join(self.work_dir, self.output_subdir, model_abbr)
        mkdir_or_exist(pred_root)
        out_path = get_infer_output_path(
            self.model_cfg,
            dataset_cfg,
            osp.join(self.work_dir, self.output_subdir),
            file_extension="json",
        )

        out_dir = Path(osp.splitext(out_path)[0])
        out_dir.mkdir(parents=True, exist_ok=True)

        existing_preds = {}
        if osp.isfile(out_path):
            try:
                with open(out_path) as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    existing_preds = data
            except (json.JSONDecodeError, OSError):
                pass
        if not existing_preds and (out_dir / "preds.json").exists():
            try:
                with open(out_dir / "preds.json") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    existing_preds = data
            except (json.JSONDecodeError, OSError):
                pass
        existing_ids = set(existing_preds.keys())
        instances = [i for i in instances if i["instance_id"] not in existing_ids]
        if existing_ids:
            self.logger.info("Reuse: skipping %d already-done instances", len(existing_ids))
        if not instances:
            self.logger.info("All instances already done, nothing to run.")
            return

        ensure_swebench_pro_docker_images(
            instances,
            self.logger,
            get_dockerhub_image_uri,
            task_label="infer",
        )

        config_path = get_config_path("config/swebp.yaml")
        self.logger.info(f"Loading agent config from '{config_path}'")
        default_swebench_config = yaml.safe_load(config_path.read_text())

        our_config = _get_minisweagent_config(self.model_cfg)
        model_name = (our_config.get("model") or {}).get("model_name") or ""
        if not (model_name or "").strip():
            raise AISBenchValueError(
                SWEBP_CODES.MODEL_NOT_SET,
                "No model set for SWEBenchPro infer. In your config, set "
                "models[0]['model'], models[0]['url'], and models[0]['api_key']. "
                "Example for local vLLM: model='hosted_vllm/qwen3', url='http://127.0.0.1:2998/v1', api_key='EMPTY'."
            )
        our_config.setdefault("environment", {})["environment_class"] = "docker"
        # SWE-bench Pro images have hardcoded entrypoint=bin/bash which prevents sleep command execution, need to override to empty for default bin/sh
        our_config.setdefault("environment", {})["run_args"] = ["--rm", "--entrypoint="]
        base_config = merge_nested_dicts(default_swebench_config, our_config)
        if dataset_cfg.get("step_limit") is not None:
            base_config.setdefault("agent", {})["step_limit"] = dataset_cfg["step_limit"]
        self.logger.info(f"base_config '{base_config}'")

        progress_manager, live_render_group = _make_swebench_pro_progress_manager(
            task_state_manager, len(instances), out_dir
        )
        task_state_manager.update_task_state(
            {
                "status": "inferencing",
                "total_count": len(instances),
                "finish_count": 0,
                "progress_description": "SWEBenchPro infer",
            }
        )

        def build_instance(raw_instance: dict) -> BatchInstance:
            return BatchInstance(
                        instance_id=raw_instance["instance_id"],
                        problem_statement=raw_instance["problem_statement"],
                        image_name=get_dockerhub_image_uri(raw_instance),
                        repo_name=raw_instance["repo"],
                        base_commit=raw_instance["base_commit"],
                        extra_fields={
                            k: v
                            for k, v in raw_instance.items()
                            if k
                            not in [
                                "instance_id",
                                "problem_statement",
                                "image_name",
                                "repo_name",
                                "base_commit",
                            ]
                        },
                    )

        workers = self.model_cfg.get("batch_size", 1)
        pro_instances = [build_instance(inst) for inst in instances]

        run_config = RunBatchConfig(
            workers=workers,
            redo_existing=False,
            raise_exceptions=True, # raise exceptions in the main thread
        )

        def process_futures(futures):
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except concurrent.futures.CancelledError:
                    pass
                except Exception as e:
                    instance_id = futures[future]
                    self.logger.error(
                        SWEBP_CODES.HARNESS_RUNTIME_FAILED,
                        "Error in future for instance %s: %s",
                        instance_id,
                        e,
                        exc_info=True,
                    )
                    progress_manager.on_uncaught_exception(instance_id, e)

        interrupted = [False]

        def run_executor():
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
            try:
                futures = {
                    executor.submit(
                        process_instance,
                        inst,
                        out_dir,
                        base_config,
                        progress_manager,
                        run_config,
                    ): inst.instance_id
                    for inst in pro_instances
                }
                try:
                    process_futures(futures)
                except KeyboardInterrupt:
                    interrupted[0] = True
                    self.logger.info(
                        "Cancelling all pending jobs. Exiting without waiting for running tasks."
                    )
                    for future in futures:
                        if not future.running() and not future.done():
                            future.cancel()
                    cleanup_swebench_pro_containers()
                    executor.shutdown(wait=False)
                    raise
            finally:
                if not interrupted[0]:
                    executor.shutdown(wait=True)
                cleanup_swebench_pro_containers()

        if live_render_group is not None:
            from rich.live import Live
            with Live(live_render_group, refresh_per_second=4):
                run_executor()
        else:
            run_executor()

        preds_path = out_dir / "preds.json"
        merged = dict(existing_preds)
        if preds_path.exists():
            with open(preds_path) as f:
                new_preds = json.load(f)
            if isinstance(new_preds, dict):
                merged.update(new_preds)
        if merged:
            model_abbr = model_abbr_from_cfg(self.model_cfg)
            for pred in merged.values():
                if isinstance(pred, dict):
                    pred["model_name_or_path"] = model_abbr
            mkdir_or_exist(osp.dirname(out_path))
            with open(out_path, "w") as f:
                json.dump(merged, f, indent=2, ensure_ascii=False)
                f.write("\n")
        if preds_path.exists():
            try:
                preds_path.unlink()
            except OSError:
                pass


def parse_args():
    parser = argparse.ArgumentParser(description="SWEBenchPro Infer")
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
                "logs/infer/", f"{task_abbr_from_cfg(cfg)}.out"
            ),
        }
    )
    start_time = time.perf_counter()
    try:
        task = SWEBenchProInferTask(cfg)
        task.run(task_state_manager)
    except KeyboardInterrupt:
        task_state_manager.update_task_state({"status": "cancelled"})
        logger.info("Inference interrupted by user")
        os._exit(130)
    except Exception as e:
        task_state_manager.update_task_state({"status": "error"})
        raise
    end_time = time.perf_counter()
    logger.info("SWEBenchPro infer time: %.2fs", end_time - start_time)
    task_state_manager.update_task_state({"status": "finish"})
    manager_t.join()
