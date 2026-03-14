import argparse
import json
import os
import os.path as osp
import sys
import threading
import time
from pathlib import Path
from typing import List

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


def _get_minisweagent_config(model_cfg: ConfigDict) -> dict:
    """Build mini-swe-agent model config from ais_bench model_cfg (e.g. LiteLLMChat)."""
    model_name = model_cfg.get("model") or model_cfg.get("model_name") or ""
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
    return {
        "model": {
            "model_name": model_name,
            "model_class": model_class,
            "model_kwargs": model_kwargs,
        }
    }


class _AISBenchProgressManager:
    """Minimal progress manager that forwards to TaskStateManager for process_instance."""

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
                "progress_description": "SWEBench infer",
                "other_kwargs": {"current": instance_id},
            }
        )

    def update_instance_status(self, instance_id: str, message: str) -> None:
        self._tsm.update_task_state(
            {
                "status": "inferencing",
                "finish_count": self._finish_count,
                "total_count": self._total,
                "progress_description": "SWEBench infer",
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
                "progress_description": "SWEBench infer",
            }
        )


@TASKS.register_module()
class SWEBenchInferTask(BaseTask):
    """SWEBench Inference Task.

    Runs mini-swe-agent on SWE-bench instances and writes predictions as JSONL.
    """

    name_prefix = "SWEBenchInfer"
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

    def get_output_paths(self, file_extension: str = "jsonl") -> List[str]:
        paths = []
        for dataset_cfg in self.dataset_cfgs:
            paths.append(
                get_infer_output_path(
                    self.model_cfg,
                    dataset_cfg,
                    os.path.join(self.work_dir, self.output_subdir),
                    file_extension=file_extension,
                )
            )
        return paths

    def run(self, task_state_manager: TaskStateManager):
        self.task_state_manager = task_state_manager
        self.logger.info("SWEBenchInferTask %s", task_abbr_from_cfg(self.cfg))

        try:
            from minisweagent.run.benchmarks.swebench import process_instance
        except ImportError as e:
            raise ImportError(
                "SWEBenchInferTask requires mini-swe-agent. "
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

        model_abbr = model_abbr_from_cfg(self.model_cfg)
        pred_root = osp.join(self.work_dir, self.output_subdir, model_abbr)
        mkdir_or_exist(pred_root)
        out_path = get_infer_output_path(
            self.model_cfg,
            dataset_cfg,
            osp.join(self.work_dir, self.output_subdir),
            file_extension="jsonl",
        )
        out_dir = Path(osp.splitext(out_path)[0] + "_tmp")
        out_dir.mkdir(parents=True, exist_ok=True)

        base_config = _get_minisweagent_config(self.model_cfg)
        base_config.setdefault("environment", {})["environment_class"] = "docker"
        base_config.setdefault("agent", {})

        progress_manager = _AISBenchProgressManager(
            task_state_manager, len(instances)
        )
        task_state_manager.update_task_state(
            {
                "status": "inferencing",
                "total_count": len(instances),
                "finish_count": 0,
                "progress_description": "SWEBench infer",
            }
        )

        for instance in instances:
            process_instance(
                instance,
                out_dir,
                base_config,
                progress_manager,
            )

        preds_path = out_dir / "preds.json"
        preds = {}
        if preds_path.exists():
            with open(preds_path) as f:
                preds = json.load(f)

        mkdir_or_exist(osp.dirname(out_path))
        with open(out_path, "w") as f:
            for instance_id, rec in preds.items():
                line = json.dumps(
                    {
                        "instance_id": instance_id,
                        "model_name_or_path": rec.get("model_name_or_path", model_abbr),
                        "model_patch": rec.get("model_patch", ""),
                    },
                    ensure_ascii=False,
                )
                f.write(line + "\n")

        if out_dir.exists():
            import shutil
            try:
                shutil.rmtree(out_dir)
            except OSError:
                pass


def parse_args():
    parser = argparse.ArgumentParser(description="SWEBench Infer")
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
        task = SWEBenchInferTask(cfg)
        task.run(task_state_manager)
    except Exception as e:
        task_state_manager.update_task_state({"status": "error"})
        raise
    end_time = time.perf_counter()
    logger.info("SWEBench infer time: %.2fs", end_time - start_time)
    task_state_manager.update_task_state({"status": "finish"})
    manager_t.join()
