"""Set the dataset path and model connection before running agent evaluation."""

from ais_bench.benchmark.runners.harbor_runner import HarborRunner
from ais_bench.benchmark.summarizers.harbor import HarborSummarizer
from ais_bench.benchmark.tasks.base import EmptyTask
from ais_bench.benchmark.tasks.custom_tasks.harbor_agent_task import HarborAgentTask

dataset_path = "/data/automationbench-public"
models = [
    {
        "abbr": "automationbench-official",
        "agent_import_path": "automationbench_adapter.agent:AutomationBenchOfficialAgent",
        "model_names": ["deepseek-flash"],
        # Supply the official API key via the AISBench model configuration.
    }
]
datasets = [
    {
        "abbr": "automationbench",
        # Local generated tasks: no dataset registry or download is used.
        "args": {
            "path": dataset_path,
            "environment_type": "docker",
            "n_attempts": 1,
            "environment_force_build": False,
        },
    }
]
infer = {"runner": {"task": {"type": EmptyTask}}}
eval = {"runner": {"type": HarborRunner, "task": {"type": HarborAgentTask}}}
summarizer = {"type": HarborSummarizer}
