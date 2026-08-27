import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

from mmengine.config import Config

from ais_bench.benchmark.registry import TASKS
from ais_bench.benchmark.tasks.base import TaskStateManager
from ais_bench.benchmark.tasks.custom_tasks.harbor_task import HarborTask
from ais_bench.benchmark.utils.agent_params import (
    AgentParamAdapter,
    parse_env_strings,
    parse_kwarg_strings,
)
from ais_bench.benchmark.utils.core.abbr import task_abbr_from_cfg
from ais_bench.benchmark.utils.logging import AISLogger


@TASKS.register_module()
class HarborAgentTask(HarborTask):
    """Harbor agent evaluation task supporting all harbor agents.

    Reuses :class:`HarborTask`'s outer flow (run / _run_with_tqdm /
    _resume_job / _dump_eval_results) so result on-disk format and resume
    behavior stay identical. Only the JobConfig construction is overridden
    to:

      - accept every harbor ``AgentName`` as a plain string, plus custom
        ``module:ClassName`` agents via ``import_path``;
      - translate unified user-facing parameters (api_base / api_key /
        llm_kwargs / model_info) per agent via :class:`AgentParamAdapter`;
      - support all harbor dataset sources (local path incl. single task
        dir, registry ``name@version``, package ``org/name@ref``).

    All harbor APIs follow the current harbor 0.21.0 definitions and are
    imported lazily so this module is safe in non-agent environments.
    """

    name_prefix = "HarborAgentTask"

    def get_command(self, cfg_path, template) -> str:
        sys.path.append(os.getcwd())
        script_path = __file__
        python = sys.executable
        return f"{python} {script_path} {cfg_path}"

    def _run_harbor_job(self):
        dataset_cfg = self.dataset_cfgs[0]
        args = dataset_cfg.get("args") or {}

        # resume check keeps HarborTask semantics: an existing job config
        # means the previous run is resumed instead of re-created.
        existing_job_dir = Path(self.out_detail_dir) / "details"
        if (existing_job_dir / "config.json").exists():
            return self._resume_job(existing_job_dir)

        config = self._build_job_config(args)
        self._dump_job_config_debug(config)
        self.logger.info(f"Harbor Job Config: {config}")

        total_tasks = self._get_task_count(config)
        if config.n_attempts > 1:
            total_tasks *= config.n_attempts
        return self._run_with_tqdm(config, total_tasks)

    # ------------------------------------------------------------------
    # live progress metrics for the state board
    # ------------------------------------------------------------------

    _PROGRESS_METRICS_INTERVAL = 2.0

    def _refresh_progress_metrics(self):
        """Periodically push live 正确/错误/异常/平均分 into the task state so
        the TasksMonitor board can show them under "Extend Parameters"."""
        if not self.task_state_manager:
            return
        job_dir = getattr(self, "_progress_job_dir", None)
        if job_dir is None:
            # fallback for paths that don't go through _run_with_tqdm
            if self.job and self.job.job_dir:
                job_dir = Path(self.job.job_dir)
            else:
                return
        now = time.time()
        if now - getattr(self, "_last_metrics_ts", 0.0) < self._PROGRESS_METRICS_INTERVAL:
            return
        self._last_metrics_ts = now
        metrics = self._job_metrics(job_dir)
        if not getattr(self, "_metrics_reported", False):
            self._metrics_reported = True
            self.logger.info(
                f"[board-metrics] job_dir={job_dir} result_json_exists="
                f"{(job_dir / 'result.json').exists()} metrics={metrics}"
            )
        if metrics:
            self.task_state_manager.update_task_state({"other_kwargs": metrics})

    def _job_metrics(self, job_dir: Path) -> dict | None:
        """Real-time aggregate of the running harbor job, read from
        ``job_dir/result.json``:

        - correct:   completed trials with reward >= 1.0
        - wrong:     completed trials with reward < 1.0 (no exception)
        - exception: trials with an exception
        - avg_score: mean reward over the completed trials
        """
        result_path = job_dir / "result.json"
        if not result_path.exists():
            return None
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return None
        if not isinstance(data, dict):
            return None

        n_correct = n_wrong = n_exception = 0
        rewards: list[float] = []
        for tr in data.get("trial_results") or []:
            if not isinstance(tr, dict):
                continue
            if tr.get("exception_info") is not None:
                n_exception += 1
                continue
            verifier = tr.get("verifier_result")
            if not isinstance(verifier, dict):
                continue
            rewards_map = verifier.get("rewards")
            if not isinstance(rewards_map, dict):
                continue
            reward = rewards_map.get("reward")
            if reward is None:
                continue
            try:
                value = float(reward)
            except (TypeError, ValueError):
                continue
            rewards.append(value)
            if value >= 1.0:
                n_correct += 1
            else:
                n_wrong += 1

        return {
            "correct": n_correct,
            "wrong": n_wrong,
            "exception": n_exception,
            "avg_score": round(sum(rewards) / len(rewards), 4) if rewards else None,
        }

    def _dump_job_config_debug(self, config) -> None:
        """Dump the resolved JobConfig for external replay/debugging.

        Enabled by setting ``AISBENCH_DUMP_JOB_CONFIG=1``. The dumped config
        can be replayed with the native CLI: ``harbor run -c <path> -y``, to
        tell whether a problem comes from the generated config or from the
        AISBench execution context.
        """
        if not os.environ.get("AISBENCH_DUMP_JOB_CONFIG"):
            return
        dump_path = Path(self.out_detail_dir) / "aisbench_job_config.json"
        try:
            dump_path.write_text(
                config.model_dump_json(
                    indent=4,
                    exclude_defaults=True,
                    context={"redact_sensitive_env": False},
                )
            )
            self.logger.info(f"Dumped AISBench job config to {dump_path}")
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"Failed to dump job config: {e}")

    # ------------------------------------------------------------------
    # JobConfig construction (harbor 0.21.0)
    # ------------------------------------------------------------------

    def _build_job_config(self, args: dict):
        from harbor.models.job.config import JobConfig

        config = JobConfig()
        config.job_name = "details"
        config.jobs_dir = Path(self.out_detail_dir)

        self._apply_job_settings(config, args)
        config.agents = self._build_agents()
        self._apply_environment(config, args)
        self._apply_verifier(config, args)
        self._apply_dataset_source(config, args)
        return config

    def _apply_job_settings(self, config, args: dict) -> None:
        for field in (
            "n_attempts",
            "timeout_multiplier",
            "agent_timeout_multiplier",
            "verifier_timeout_multiplier",
            "agent_setup_timeout_multiplier",
            "environment_build_timeout_multiplier",
            "n_concurrent_trials",
            "quiet",
            "debug",
            "install_only",
        ):
            if args.get(field) is not None:
                setattr(config, field, args[field])
        if args.get("max_retries") is not None:
            config.retry.max_retries = args["max_retries"]
        if args.get("retry_include_exceptions") is not None:
            config.retry.include_exceptions = set(args["retry_include_exceptions"])
        if args.get("retry_exclude_exceptions") is not None:
            config.retry.exclude_exceptions = set(args["retry_exclude_exceptions"])
        if args.get("metrics") is not None:
            config.metrics = args["metrics"]
        if args.get("artifacts") is not None:
            config.artifacts = list(args["artifacts"])
        if args.get("extra_instruction_paths") is not None:
            config.extra_instruction_paths = [
                Path(p) for p in args["extra_instruction_paths"]
            ]

    def _build_agents(self):
        from harbor.models.agent.name import AgentName
        from harbor.models.trial.config import AgentConfig

        model_cfg = self.model_cfg
        agent_name = model_cfg.get("agent_name")
        agent_import_path = model_cfg.get("agent_import_path")
        if agent_name is None and agent_import_path is None:
            agent_name = "oracle"
        # Validate built-in agent names; import-path style values ("module:Class")
        # pass through untouched and are resolved by harbor's AgentFactory.
        if agent_name and ":" not in agent_name and agent_name not in AgentName.values():
            raise ValueError(
                f"Unknown agent name {agent_name!r}. Valid agents: "
                f"{sorted(AgentName.values())}, or pass a custom agent import "
                "path 'module.path:ClassName' via agent_import_path."
            )

        translated = AgentParamAdapter.translate(agent_name, model_cfg)
        raw_kwargs = dict(model_cfg.get("agent_kwargs") or {})
        raw_env = dict(model_cfg.get("agent_env") or {})
        # CLI-provided agent kwargs/env are merged directly from cli_args so
        # they always reach the AgentConfig even if the config dump/reload
        # round-trip dropped the in-model dicts. Merging is idempotent and
        # CLI values win over config-file values.
        cli_kwargs = parse_kwarg_strings(self.cli_args.get("agent_kwarg"))
        cli_env = parse_env_strings(self.cli_args.get("agent_env"))
        # explicit user-provided kwargs / env win over translated values
        kwargs = {**translated["kwargs"], **raw_kwargs, **cli_kwargs}
        env = {**translated["env"], **raw_env, **cli_env}
        # inherit proxy env vars from the host process when not explicitly set,
        # so agents can reach model services through the same proxy as the CLI
        for var in ("http_proxy", "https_proxy", "no_proxy",
                    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
            if var not in env and os.environ.get(var):
                env[var] = os.environ[var]
        self.logger.info(
            f"Agent '{agent_name}' built env keys: {sorted(env.keys())} | "
            f"cli_args.agent_env present: {bool(self.cli_args.get('agent_env'))} | "
            f"model_cfg.agent_env present: {bool(model_cfg.get('agent_env'))}"
        )

        model_names = model_cfg.get("model_names")
        deps_path = model_cfg.get("deps_path")
        if deps_path is not None:
            # resolve now so a stored job config replays from any cwd (parity
            # with harbor's CLI --agent-deps handling)
            deps_path = str(Path(deps_path).expanduser().resolve())
        common = {
            "name": agent_name,
            "import_path": agent_import_path,
            "kwargs": kwargs,
            "env": env,
            "skills": list(model_cfg.get("skills") or []),
            "mcp_servers": list(model_cfg.get("mcp_servers") or []),
            "include_logs": list(model_cfg.get("include_logs") or []),
            "exclude_logs": list(model_cfg.get("exclude_logs") or []),
            "extra_allowed_hosts": list(model_cfg.get("extra_allowed_hosts") or []),
            "n_concurrent": model_cfg.get("n_concurrent"),
            "concurrency_group": model_cfg.get("concurrency_group"),
            "resume_trajectory": model_cfg.get("resume_trajectory"),
            "load_trajectory": model_cfg.get("load_trajectory"),
            "deps_path": deps_path,
            "override_timeout_sec": model_cfg.get("override_timeout_sec"),
            "override_setup_timeout_sec": model_cfg.get("override_setup_timeout_sec"),
            "max_timeout_sec": model_cfg.get("max_timeout_sec"),
        }
        common = {k: v for k, v in common.items() if v is not None}

        if model_names:
            return [
                AgentConfig(model_name=model_name, **common)
                for model_name in model_names
            ]
        return [AgentConfig(**common)]

    def _apply_environment(self, config, args: dict) -> None:
        from harbor.models.environment_type import EnvironmentType

        if args.get("environment_type"):
            config.environment.type = EnvironmentType(args["environment_type"])
        if args.get("environment_force_build") is not None:
            config.environment.force_build = args["environment_force_build"]
        if args.get("environment_delete") is not None:
            config.environment.delete = args["environment_delete"]
        if args.get("environment_env"):
            env = args["environment_env"]
            if isinstance(env, dict):
                config.environment.env.update(env)
        if args.get("environment_kwargs"):
            kwargs = args["environment_kwargs"]
            if isinstance(kwargs, dict):
                config.environment.kwargs.update(kwargs)
        for field in (
            "override_cpus",
            "override_memory_mb",
            "override_storage_mb",
            "override_gpus",
        ):
            if args.get(field) is not None:
                setattr(config.environment, field, args[field])

    def _apply_verifier(self, config, args: dict) -> None:
        if args.get("disable_verification"):
            config.verifier.disable = True
        if args.get("verifier_env"):
            env_list = args["verifier_env"]
            if isinstance(env_list, list):
                config.verifier.env.update(
                    {k: v for k, v in (e.split("=", 1) for e in env_list if "=" in e)}
                )
        if args.get("verifier_import_path"):
            config.verifier.import_path = args["verifier_import_path"]
        if args.get("verifier_kwargs"):
            kwargs = args["verifier_kwargs"]
            if isinstance(kwargs, dict):
                config.verifier.kwargs.update(kwargs)

    def _apply_dataset_source(self, config, args: dict) -> None:
        """Resolve datasets / tasks from the dataset args.

        Mirrors harbor's CLI dataset resolution: a local path is a single
        task directory when ``Task.is_valid_dir`` holds, otherwise it is a
        dataset directory. ``name@version`` resolves to a registry dataset;
        ``org/name@ref`` resolves to a package dataset.
        """
        from harbor.models.job.config import DatasetConfig, TaskConfig
        from harbor.models.task.task import Task as TaskModel

        task_names = args.get("task_names")
        exclude_task_names = args.get("exclude_task_names")
        n_tasks = args.get("n_tasks")

        path = args.get("path")
        if path is not None:
            path_obj = Path(path)
            if path_obj.is_dir() and TaskModel.is_valid_dir(
                path_obj, disable_verification=config.verifier.disable
            ):
                config.datasets = []
                config.tasks = [TaskConfig(path=path_obj)]
            else:
                config.datasets = [
                    DatasetConfig(
                        path=path_obj,
                        task_names=task_names,
                        exclude_task_names=exclude_task_names,
                        n_tasks=n_tasks,
                    )
                ]
                config.tasks = []
            return

        dataset_name_version = args.get("dataset_name_version")
        if dataset_name_version:
            name = dataset_name_version
            version = None
            if "@" in name:
                name, version = name.split("@", 1)
            if "/" in name:
                # package reference (org/name)
                config.datasets = [
                    DatasetConfig(
                        name=name,
                        ref=version or "latest",
                        task_names=task_names,
                        exclude_task_names=exclude_task_names,
                        n_tasks=n_tasks,
                    )
                ]
            else:
                config.datasets = [
                    DatasetConfig(
                        name=name,
                        version=version,
                        registry_url=args.get("registry_url"),
                        registry_path=args.get("registry_path"),
                        task_names=task_names,
                        exclude_task_names=exclude_task_names,
                        n_tasks=n_tasks,
                    )
                ]
            config.tasks = []
            return

        raise ValueError(
            "Dataset is not configured: provide 'path' or 'dataset_name_version' "
            "in the dataset args."
        )

    def _get_task_count(self, config) -> int:
        from harbor.cli.utils import run_async

        async def _count():
            count = len(config.tasks)
            for dataset_config in config.datasets:
                task_configs = await dataset_config.get_task_configs(
                    disable_verification=config.verifier.disable
                )
                count += len(task_configs)
            return count

        return run_async(_count())


def parse_args():
    parser = argparse.ArgumentParser(description="Harbor Agent Benchmark Task")
    parser.add_argument("config", help="Config file path")
    return parser.parse_args()


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
                HarborAgentTask.log_subdir, f"{task_abbr_from_cfg(cfg)}.out"
            ),
        }
    )

    start_time = time.perf_counter()
    try:
        inferencer = HarborAgentTask(cfg)
        inferencer.run(task_state_manager)
    except BaseException as e:
        # BaseException (not just Exception): on Ctrl+C (KeyboardInterrupt)
        # the task state must still be flipped to "error" so the non-daemon
        # TaskStateManager thread exits and the process can terminate after
        # harbor recycles its containers.
        task_state_manager.update_task_state({"status": "error"})
        raise e

    end_time = time.perf_counter()
    logger.info(
        f"Harbor agent benchmark task time elapsed: {end_time - start_time:.2f}s"
    )
    task_state_manager.update_task_state({"status": "finish"})
    manager_t.join()
