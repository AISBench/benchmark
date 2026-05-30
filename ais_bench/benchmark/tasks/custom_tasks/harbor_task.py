import argparse
import copy
import json
import os
import os.path as osp
import re
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

from mmengine.config import Config, ConfigDict
from mmengine.utils import mkdir_or_exist
from tqdm import tqdm

from ais_bench.benchmark.registry import TASKS
from ais_bench.benchmark.tasks.base import BaseTask, TaskStateManager
from ais_bench.benchmark.utils.core.abbr import task_abbr_from_cfg
from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES

DEFAULT_FAKE_API_KEY = "fake_api_key"


@TASKS.register_module()
class HarborTask(BaseTask):
    name_prefix = "HarborTask"
    log_subdir = "logs/eval"
    output_subdir = "results"

    def __init__(self, cfg: ConfigDict) -> None:
        super().__init__(cfg)
        self.captured_metrics = None
        self.job_dir = None
        self.job_result = None
        self.job = None

    def get_command(self, cfg_path, template) -> str:
        sys.path.append(os.getcwd())
        script_path = __file__
        python = sys.executable
        return f"{python} {script_path} {cfg_path}"

    def run(self, task_state_manager: TaskStateManager):
        self.logger.info(f"Task {task_abbr_from_cfg(self.cfg)}")
        self.task_state_manager = task_state_manager

        self._set_api_key()
        self._prepare_out_dir()

        job, job_result = self._run_harbor_job()

        self._dump_eval_results(job, job_result)

    def _set_api_key(self):
        api_key = self.model_cfg.get("api_key")
        if api_key is None:
            api_key = DEFAULT_FAKE_API_KEY
        os.environ["OPENAI_API_KEY"] = api_key

    def _prepare_out_dir(self):
        dataset_cfg = self.dataset_cfgs[0]
        self.out_dir = osp.join(
            self.work_dir, self.output_subdir, self.model_cfg["abbr"]
        )
        mkdir_or_exist(osp.join(self.out_dir, dataset_cfg["abbr"]))
        self.out_detail_dir = osp.join(
            self.out_dir,
            dataset_cfg["abbr"],
        )
        mkdir_or_exist(Path(self.out_detail_dir))

    def _run_harbor_job(self):
        from harbor.cli.utils import run_async
        from harbor.job import Job
        from harbor.models.job.config import (
            AgentConfig,
            DatasetConfig,
            JobConfig,
        )
        from harbor.models.agent.name import AgentName
        from harbor.models.environment_type import EnvironmentType

        dataset_cfg = self.dataset_cfgs[0]
        args = dataset_cfg.get("args") or {}

        config = JobConfig()

        config.job_name = "details"
        config.jobs_dir = Path(self.out_detail_dir)

        if args.get("n_attempts"):
            config.n_attempts = args["n_attempts"]
        if args.get("timeout_multiplier"):
            config.timeout_multiplier = args["timeout_multiplier"]
        if args.get("agent_timeout_multiplier"):
            config.agent_timeout_multiplier = args["agent_timeout_multiplier"]
        if args.get("verifier_timeout_multiplier"):
            config.verifier_timeout_multiplier = args["verifier_timeout_multiplier"]
        if args.get("agent_setup_timeout_multiplier"):
            config.agent_setup_timeout_multiplier = args["agent_setup_timeout_multiplier"]
        if args.get("environment_build_timeout_multiplier"):
            config.environment_build_timeout_multiplier = args["environment_build_timeout_multiplier"]
        if args.get("debug"):
            config.debug = args["debug"]

        if args.get("n_concurrent_trials"):
            config.n_concurrent_trials = args["n_concurrent_trials"]
        if args.get("quiet"):
            config.quiet = args["quiet"]
        if args.get("max_retries"):
            config.retry.max_retries = args["max_retries"]
        if args.get("retry_include_exceptions"):
            config.retry.include_exceptions = set(args["retry_include_exceptions"])
        if args.get("retry_exclude_exceptions"):
            config.retry.exclude_exceptions = set(args["retry_exclude_exceptions"])

        agent_kwargs = self.model_cfg.get("agent_kwargs") or {}
        agent_env = self.model_cfg.get("agent_env") or {}

        agent_name = AgentName(self.model_cfg.get("agent_name", "oracle"))
        model_names = self.model_cfg.get("model_names")
        if model_names:
            config.agents = [
                AgentConfig(
                    name=agent_name,
                    model_name=model_name,
                    kwargs=agent_kwargs,
                    env=agent_env,
                )
                for model_name in model_names
            ]
        else:
            config.agents = [
                AgentConfig(
                    name=agent_name,
                    kwargs=agent_kwargs,
                    env=agent_env,
                )
            ]

        if args.get("environment_type"):
            config.environment.type = EnvironmentType(args["environment_type"])
        if args.get("environment_force_build") is not None:
            config.environment.force_build = args["environment_force_build"]
        if args.get("environment_delete") is not None:
            config.environment.delete = args["environment_delete"]

        if args.get("disable_verification"):
            config.verifier.disable = True
        if args.get("verifier_env"):
            env_list = args["verifier_env"]
            if isinstance(env_list, list):
                config.verifier.env.update({k: v for k, v in (e.split("=", 1) for e in env_list if "=" in e)})

        existing_job_dir = Path(self.out_detail_dir) / config.job_name
        config_path = existing_job_dir / "config.json"
        if config_path.exists():
            return self._resume_job(existing_job_dir)

        if args.get("path"):
            config.datasets = [
                DatasetConfig(
                    path=Path(args["path"]),
                    task_names=args.get("task_names"),
                    exclude_task_names=args.get("exclude_task_names"),
                    n_tasks=args.get("n_tasks"),
                )
            ]
        elif args.get("dataset_name_version"):
            name = args["dataset_name_version"]
            version = None
            if "@" in name:
                name, version = name.split("@", 1)
            config.datasets = [
                DatasetConfig(
                    name=name,
                    version=version,
                    task_names=args.get("task_names"),
                    exclude_task_names=args.get("exclude_task_names"),
                    n_tasks=args.get("n_tasks"),
                )
            ]

        self.logger.info(f"Harbor Job Config: {config}")

        total_tasks = self._get_task_count(config)
        if args.get("n_attempts", 1) > 1:
            total_tasks *= args["n_attempts"]

        return self._run_with_tqdm(config, total_tasks)

    def _get_task_count(self, config) -> int:
        from harbor.cli.utils import run_async

        async def _count():
            count = 0
            for dataset_config in config.datasets:
                task_configs = await dataset_config.get_task_configs(
                    disable_verification=config.verifier.disable
                )
                count += len(task_configs)
            return count

        return run_async(_count())

    def _resume_job(self, job_path):
        from harbor.cli.utils import run_async
        from harbor.job import Job

        async def _resume():
            job_dir = Path(job_path)
            config_path = job_dir / "config.json"
            if not config_path.exists():
                raise ValueError(f"Config file not found: {config_path}")
            from harbor.models.job.config import JobConfig
            config = JobConfig.model_validate_json(config_path.read_text())
            self.logger.info(f"Resuming job from {job_dir}")
            self.logger.info(f"Config jobs_dir: {config.jobs_dir}, job_name: {config.job_name}")
            self.logger.info(f"Expected job_dir: {config.jobs_dir / config.job_name}")
            job = await Job.create(config)
            return job, await job.run()

        return run_async(_resume())

    def _run_with_tqdm(self, config, total_tasks):
        from harbor.cli.utils import run_async
        from harbor.job import Job

        pbar = tqdm(total=total_tasks, desc="Running Harbor Job", unit="task")
        completed = 0
        stop_event = threading.Event()

        if self.task_state_manager:
            self.task_state_manager.update_task_state(
                {
                    "status": "running",
                    "total_count": total_tasks,
                    "progress_description": "Running Harbor Job",
                    "finish_count": 0,
                }
            )

        def monitor_progress():
            nonlocal completed
            while not stop_event.is_set():
                if self.job and self.job.job_dir:
                    trial_count = len(list(self.job.job_dir.glob("trial_*")))
                    if trial_count > completed:
                        pbar.update(trial_count - completed)
                        completed = trial_count
                        if self.task_state_manager:
                            self.task_state_manager.update_task_state(
                                {"finish_count": completed}
                            )
                stop_event.wait(0.5)
            pbar.close()

        monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
        monitor_thread.start()

        def _handle_sigterm(signum, frame):
            stop_event.set()
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, _handle_sigterm)

        try:

            async def _run_job():
                job = await Job.create(config)
                return job, await job.run()

            self.job, self.job_result = run_async(_run_job())
        finally:
            stop_event.set()
            monitor_thread.join(timeout=5)
            pbar.close()
            if self.task_state_manager:
                self.task_state_manager.update_task_state(
                    {"finish_count": total_tasks}
                )

        return self.job, self.job_result

    def _dump_eval_results(self, job, job_result):
        dataset_cfg = self.dataset_cfgs[0]
        task_abbr = dataset_cfg["abbr"]

        if job_result is None:
            self.logger.error(UTILS_CODES.UNKNOWN_ERROR, "No job result captured.")
            return

        out_json = osp.join(self.out_dir, f"{task_abbr}.json")

        total_count = self._get_task_count(job.config)
        all_rewards = []
        n_errors = 0
        reward_distribution = {}
        exception_distribution = {}

        for trial_result in job_result.trial_results or []:
            if trial_result.exception_info is not None:
                n_errors += 1
                exc_type = trial_result.exception_info.exception_type
                exception_distribution[exc_type] = exception_distribution.get(exc_type, 0) + 1
            elif trial_result.verifier_result and trial_result.verifier_result.rewards:
                for key, value in trial_result.verifier_result.rewards.items():
                    all_rewards.append(value)
                    score_key = str(value)
                    reward_distribution[score_key] = reward_distribution.get(score_key, 0) + 1

        total_reward = sum(all_rewards) if all_rewards else 0.0
        avg_reward = (total_reward / job_result.n_total_trials) if job_result.n_total_trials > 0 else 0.0

        pass_at_k = {}
        if job_result.stats and job_result.stats.evals:
            for evals_key, eval_stats in job_result.stats.evals.items():
                if eval_stats.pass_at_k:
                    pass_at_k = eval_stats.pass_at_k
                    break

        results = {
            "total_count": total_count,
            "n_errors": n_errors,
            "avg_score": round(avg_reward, 4),
            "reward_distribution": [
                {"score": float(k), "count": v}
                for k, v in sorted(
                    reward_distribution.items(), key=lambda x: float(x[0]), reverse=True
                )
            ],
            "exception_distribution": [
                {"exception_type": k, "count": v}
                for k, v in sorted(
                    exception_distribution.items(), key=lambda x: x[1], reverse=True
                )
            ],
            "n_total_trials": job_result.n_total_trials,
            "pass_at_k": pass_at_k,
        }

        with open(out_json, "w") as f:
            json.dump(results, f, indent=4)

        self.logger.info(f"Evaluation results saved to {out_json}")


def parse_args():
    parser = argparse.ArgumentParser(description="Harbor Benchmark Task")
    parser.add_argument("config", help="Config file path")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logger = AISLogger(__name__)
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
                HarborTask.log_subdir, f"{task_abbr_from_cfg(cfg)}.out"
            ),
        }
    )

    start_time = time.perf_counter()
    try:
        inferencer = HarborTask(cfg)
        inferencer.run(task_state_manager)
    except Exception as e:
        task_state_manager.update_task_state({"status": "error"})
        raise e

    end_time = time.perf_counter()
    logger.info(f"Harbor benchmark task time elapsed: {end_time - start_time:.2f}s")
    task_state_manager.update_task_state({"status": "finish"})
    manager_t.join()
