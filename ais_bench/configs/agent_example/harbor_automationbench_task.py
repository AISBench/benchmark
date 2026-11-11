"""Set the dataset path and model connection before running agent evaluation."""

from ais_bench.benchmark.runners.harbor_runner import HarborRunner
from ais_bench.benchmark.summarizers.harbor import HarborSummarizer
from ais_bench.benchmark.tasks.base import EmptyTask
from ais_bench.benchmark.tasks.custom_tasks.harbor_agent_task import HarborAgentTask

# Replace this placeholder with your extracted dataset directory.
dataset_path = "/path/to/automationbench-public"
models = [
    {
        "abbr": "automationbench-official",
        "agent_import_path": "automationbench_adapter.agent:AutomationBenchOfficialAgent",
        "model_names": ["deepseek-flash"],
        # Replace the placeholder or override it with --api-key.
        "api_key": "sk-xxx",
        # Custom Agent options; do not use Terminus-2 llm_kwargs here.
        "agent_kwargs": {
            "base_url": "https://api.deepseek.com/v1",
            "sampling_args": None,  # Preserve official scaffold defaults.
            "pricing": None,  # No assumed price table.
        },
        "agent_env": {},
        "deps_path": None,
        "n_concurrent": None,  # No additional per-Agent concurrency cap.
        "override_timeout_sec": None,  # Use task.toml timeout.
        "override_setup_timeout_sec": None,
        "max_timeout_sec": None,
    }
]
datasets = [
    {
        "abbr": "automationbench",
        # Local generated tasks: no dataset registry or download is used.
        "args": {
            "path": dataset_path,
            "dataset_name_version": None,
            "task_names": None,
            "exclude_task_names": None,
            "n_tasks": None,  # All 600 tasks; set a small limit for smoke runs.
            "environment_type": "docker",
            "n_attempts": 1,
            "n_concurrent_trials": 4,
            "timeout_multiplier": 1.0,
            "agent_timeout_multiplier": None,
            "verifier_timeout_multiplier": None,
            "agent_setup_timeout_multiplier": None,
            "environment_build_timeout_multiplier": None,
            "max_retries": 0,
            "retry_include_exceptions": None,
            "retry_exclude_exceptions": None,  # Keep Harbor's default exclusions.
            "environment_force_build": False,
            "environment_delete": True,  # Remove containers; retain Job artifacts.
            "environment_env": {},
            "environment_kwargs": {},
            "disable_verification": False,
            "verifier_env": None,
            "debug": False,
            "quiet": False,
            "install_only": False,
        },
    }
]
infer = {"runner": {"task": {"type": EmptyTask}}}
eval = {
    "runner": {
        "type": HarborRunner,
        "task": {"type": HarborAgentTask},
        "max_num_workers": 1,
        "monitor_port": 0,
        "refresh_interval": 0.5,
        "debug": False,
    }
}
summarizer = {"type": HarborSummarizer}
