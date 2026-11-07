"""Set the dataset path and model connection before running agent evaluation."""
from ais_bench.benchmark.runners.harbor_runner import HarborRunner
from ais_bench.benchmark.summarizers.harbor import HarborSummarizer
from ais_bench.benchmark.tasks.base import EmptyTask
from ais_bench.benchmark.tasks.custom_tasks.harbor_agent_task import HarborAgentTask

dataset_path = "/data/automationbench-public"
models = [dict(
    abbr="automationbench-official",
    agent_import_path="automationbench_adapter.agent:AutomationBenchOfficialAgent",
    model_names=["deepseek-flash"],
    # Supply the official API key via the AISBench model configuration.
)]
datasets = [dict(
    abbr="automationbench",
    args=dict(path=dataset_path, environment_type="docker", n_attempts=1),
    reporting=dict(
        publisher="automationbench_adapter.reporting:publish_job",
        kwargs=dict(manifest=dataset_path + "/manifest.json"),
    ),
)]
infer = dict(runner=dict(task=dict(type=EmptyTask)))
eval = dict(runner=dict(type=HarborRunner, task=dict(type=HarborAgentTask)))
summarizer = dict(attr="accuracy", type=HarborSummarizer)
