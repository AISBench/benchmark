import os.path as osp
import copy
from abc import ABC, abstractmethod
from collections import defaultdict

from mmengine.config import ConfigDict

from ais_bench.benchmark.registry import PARTITIONERS, RUNNERS, build_from_cfg
from ais_bench.benchmark.utils.config.run import get_config_type
from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners import LocalRunner
from ais_bench.benchmark.tasks import (
    OpenICLEvalTask,
    OpenICLApiInferTask,
    OpenICLInferTask,
    VBenchEvalTask,
)
from ais_bench.benchmark.summarizers import DefaultSummarizer, DefaultPerfSummarizer
from ais_bench.benchmark.calculators import DefaultPerfMetricCalculator
from ais_bench.benchmark.cli.utils import fill_model_path_if_datasets_need

logger = AISLogger()


class BaseWorker(ABC):
    def __init__(self, args) -> None:
        self.args = args

    @abstractmethod
    def update_cfg(self, cfg: ConfigDict) -> None:
        # update major cfg content according to worker kind
        pass

    @abstractmethod
    def do_work(self, cfg: ConfigDict):
        # run partitioner and launch runner
        pass


class Infer(BaseWorker):
    def update_cfg(self, cfg: ConfigDict) -> None:
        def get_task_type() -> str:
            if cfg["models"][0]["attr"] == "service":
                return get_config_type(OpenICLApiInferTask)
            else:
                return get_config_type(OpenICLInferTask)

        new_cfg = dict(
            infer=dict(
                partitioner=dict(type=get_config_type(NaivePartitioner)),
                runner=dict(
                    max_num_workers=self.args.max_num_workers,
                    max_workers_per_gpu=self.args.max_workers_per_gpu,
                    debug=self.args.debug,
                    task=dict(type=get_task_type()),
                    type=get_config_type(LocalRunner),
                ),
            ),
        )

        cfg.merge_from_dict(new_cfg)
        if cfg.cli_args.debug:
            cfg.infer.runner.debug = True
        cfg.infer.partitioner["out_dir"] = osp.join(cfg["work_dir"], "predictions/")
        return cfg

    def do_work(self, cfg: ConfigDict):
        partitioner = PARTITIONERS.build(cfg.infer.partitioner)
        logger.info("Starting inference tasks...")
        tasks = partitioner(cfg)

        # update tasks cfg before run
        self._update_tasks_cfg(tasks, cfg)

        if (
            cfg.get("cli_args", {}).get("merge_ds", False)
            or cfg.get("cli_args", {}).get("mode") == "perf" # performance mode will enable merge datasets by default
        ):
            logger.info("Merging datasets with the same model and inferencer...")
            tasks = self._merge_datasets(tasks)

        runner = RUNNERS.build(cfg.infer.runner)
        runner(tasks)
        logger.info("Inference tasks completed.")

    def _merge_datasets(self, tasks):
        # merge datasets with the same model, dataset type and inferencer
        task_groups = defaultdict(list)
        for task in tasks:
            key = (
                task["models"][0]["abbr"] # same model
                + "_"
                + str(task['datasets'][0][0]['type']) # same dataset type
                + "_"
                + str(task["datasets"][0][0]["infer_cfg"]["inferencer"]) # same inferencer with the same args
            )
            task_groups[key].append(task)
        new_tasks = []
        for key, task_group in task_groups.items():
            new_task = copy.deepcopy(task_group[0])
            if len(task_group) > 1:
                for t in task_group[1:]:
                    new_task["datasets"][0].extend(t["datasets"][0])
            new_tasks.append(new_task)
        return new_tasks

    def _update_tasks_cfg(self, tasks, cfg: ConfigDict):
        # update parameters to correct sub cfg
        if hasattr(cfg, "attack"):
            for task in tasks:
                cfg.attack.dataset = task.datasets[0][0].abbr
                task.attack = cfg.attack


def _has_vbench_dataset(cfg: ConfigDict) -> bool:
    """True if any dataset in config is a VBench dataset (use_vbench_task or VBenchDataset type)."""
    for item in cfg.get("datasets", []):
        for ds in (item if isinstance(item, (list, tuple)) else [item]):
            eval_cfg = ds.get("eval_cfg") or {}
            if eval_cfg.get("use_vbench_task") is True:
                return True
            type_str = str(ds.get("type", ""))
            if "VBenchDataset" in type_str or "vbench" in type_str.lower():
                return True
    return False


class Eval(BaseWorker):
    def update_cfg(self, cfg: ConfigDict) -> None:
        eval_task_type = (
            get_config_type(VBenchEvalTask)
            if _has_vbench_dataset(cfg)
            else get_config_type(OpenICLEvalTask)
        )
        new_cfg = dict(
            eval=dict(
                partitioner=dict(type=get_config_type(NaivePartitioner)),
                runner=dict(
                    max_num_workers=self.args.max_num_workers,
                    debug=self.args.debug,
                    task=dict(type=eval_task_type),
                ),
            ),
        )

        new_cfg["eval"]["runner"]["type"] = get_config_type(LocalRunner)
        new_cfg["eval"]["runner"]["max_workers_per_gpu"] = self.args.max_workers_per_gpu
        cfg.merge_from_dict(new_cfg)
        if cfg.cli_args.dump_eval_details:
            cfg.eval.runner.task.dump_details = True
        if cfg.cli_args.dump_extract_rate:
            cfg.eval.runner.task.cal_extract_rate = True
        if cfg.cli_args.debug:
            cfg.eval.runner.debug = True
        cfg.eval.partitioner["out_dir"] = osp.join(cfg["work_dir"], "results/")
        return cfg

    def do_work(self, cfg: ConfigDict):
        partitioner = PARTITIONERS.build(cfg.eval.partitioner)
        logger.info("Starting evaluation tasks...")
        tasks = partitioner(cfg)

        # update tasks cfg before run
        self._update_tasks_cfg(tasks, cfg)

        runner = RUNNERS.build(cfg.eval.runner)
        # For meta-review-judge in subjective evaluation
        if isinstance(tasks, list) and len(tasks) != 0 and isinstance(tasks[0], list):
            for task_part in tasks:
                runner(task_part)
        else:
            runner(tasks)
        logger.info("Evaluation tasks completed.")

    def _update_tasks_cfg(self, tasks, cfg: ConfigDict):
        # update parameters to correct sub cfg
        pass


class AccViz(BaseWorker):
    def update_cfg(self, cfg: ConfigDict) -> None:
        summarizer_cfg = cfg.get("summarizer", {})
        if (
            not summarizer_cfg
            or summarizer_cfg.get("type", None) is None
            or summarizer_cfg.get("attr", None) != "accuracy"
        ):
            summarizer_cfg["type"] = get_config_type(DefaultSummarizer)
        summarizer_cfg.pop("attr", None)
        cfg["summarizer"] = summarizer_cfg
        return cfg

    def do_work(self, cfg: ConfigDict) -> int:
        logger.info("Summarizing evaluation results...")
        summarizer_cfg = cfg.get("summarizer", {})

        # For subjective summarizer
        if summarizer_cfg.get("function", None):
            main_summarizer_cfg = copy.deepcopy(summarizer_cfg)
            grouped_datasets = {}
            for dataset in cfg.datasets:
                prefix = dataset["abbr"].split("_")[0]
                if prefix not in grouped_datasets:
                    grouped_datasets[prefix] = []
                grouped_datasets[prefix].append(dataset)
            dataset_score_container = []
            for dataset in grouped_datasets.values():
                temp_cfg = copy.deepcopy(cfg)
                temp_cfg.datasets = dataset
                summarizer_cfg = dict(
                    type=dataset[0]["summarizer"]["type"], config=temp_cfg
                )
                summarizer = build_from_cfg(summarizer_cfg)
                dataset_score = summarizer.summarize(time_str=self.args.cfg_time_str)
                if dataset_score:
                    dataset_score_container.append(dataset_score)
            main_summarizer_cfg["config"] = cfg
            main_summarizer = build_from_cfg(main_summarizer_cfg)
            main_summarizer.summarize(
                time_str=self.args.cfg_time_str,
                subjective_scores=dataset_score_container,
            )
        else:
            summarizer_cfg["config"] = cfg
            summarizer = build_from_cfg(summarizer_cfg)
            summarizer.summarize(time_str=self.args.cfg_time_str)


class PerfViz(BaseWorker):
    def update_cfg(self, cfg: ConfigDict) -> None:
        summarizer_cfg = cfg.get("summarizer", {})
        if (
            not summarizer_cfg
            or summarizer_cfg.get("type", None) is None
            or summarizer_cfg.get("attr", None) != "performance"
        ):
            summarizer_cfg["type"] = get_config_type(DefaultPerfSummarizer)
        summarizer_cfg.pop("attr", None)
        if summarizer_cfg.get("calculator") is None:
            summarizer_cfg["calculator"] = dict(
                type=get_config_type(DefaultPerfMetricCalculator)
            )
        summarizer_cfg.pop("dataset_abbrs", None)
        summarizer_cfg.pop("summary_groups", None)
        summarizer_cfg.pop("prompt_db", None)
        cfg["summarizer"] = summarizer_cfg
        return cfg

    def do_work(self, cfg: ConfigDict) -> int:
        summarizer_cfg = cfg.get("summarizer", {})
        summarizer_cfg["config"] = cfg
        summarizer = build_from_cfg(summarizer_cfg)
        logger.info("Summarizing performance results...")
        summarizer.summarize()


WORK_FLOW = dict(
    all=[Infer, Eval, AccViz],
    infer=[Infer],
    eval=[Eval, AccViz],
    viz=[AccViz],
    perf=[Infer, PerfViz],
    perf_viz=[PerfViz],
)


class WorkFlowExecutor:
    def __init__(self, cfg, workflow) -> None:
        self.cfg = cfg
        self.workflow = workflow

    def execute(self) -> None:
        for worker in self.workflow:
            worker.do_work(self.cfg)
