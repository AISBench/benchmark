import argparse
import json
import os
import os.path as osp
import sys
import threading
import time

from mmengine.config import Config, ConfigDict
from mmengine.utils import mkdir_or_exist

from ais_bench.benchmark.registry import TASKS
from ais_bench.benchmark.tasks.base import BaseTask, TaskStateManager
from ais_bench.benchmark.utils.core.abbr import (
    get_infer_output_path,
    task_abbr_from_cfg,
)
from ais_bench.benchmark.utils.logging import AISLogger


@TASKS.register_module()
class SWEBenchEvalTask(BaseTask):
    """SWEBench Evaluation Task.

    Evaluates SWE-bench predictions using the official harness and writes
    results to work_dir/results.
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

        dataset_cfg = self.dataset_cfgs[0]
        dataset_name = dataset_cfg.get("name", "lite")

        pred_path = get_infer_output_path(
            self.model_cfg,
            dataset_cfg,
            osp.join(self.work_dir, "predictions"),
            file_extension="jsonl",
        )
        if not osp.isfile(pred_path):
            raise FileNotFoundError(
                f"Predictions file not found: {pred_path}. Run infer first."
            )

        out_path = get_infer_output_path(
            self.model_cfg,
            dataset_cfg,
            osp.join(self.work_dir, self.output_subdir),
            file_extension="json",
        )
        mkdir_or_exist(osp.dirname(out_path))

        task_state_manager.update_task_state(
            {"status": "eval", "progress_description": "SWE-bench harness"}
        )

        try:
            import swebench.harness.run_evaluation as run_eval
        except ImportError as e:
            raise ImportError(
                "SWEBenchEvalTask requires the SWE-bench harness. "
                "Install from: https://github.com/princeton-nlp/SWE-bench"
            ) from e

        run_id = task_abbr_from_cfg(self.cfg).replace("/", "_")
        eval_runner = self.cfg.get("eval", {}).get("runner", {})
        max_workers = eval_runner.get("max_num_workers", 4)
        report_dir = osp.dirname(out_path)

        try:
            run_eval.main(
                dataset_name=dataset_name,
                split="test",
                instance_ids=[],
                predictions_path=pred_path,
                max_workers=max_workers,
                force_rebuild=False,
                cache_level="env",
                clean=False,
                open_file_limit=4096,
                run_id=run_id,
                timeout=1800,
                namespace=None,
                rewrite_reports=False,
                modal=False,
                report_dir=report_dir,
            )
            harness_exit = 0
        except SystemExit as e:
            harness_exit = e.code if e.code is not None else 1
        except Exception as e:
            self.logger.exception("Harness failed: %s", e)
            harness_exit = 1

        results = {
            "harness_exit_code": harness_exit,
            "dataset_name": dataset_name,
            "predictions_path": pred_path,
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
