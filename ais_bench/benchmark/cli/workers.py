import os
import os.path as osp
import copy
import shutil
from abc import ABC, abstractmethod
from collections import defaultdict

from mmengine.config import ConfigDict

from ais_bench.benchmark.registry import PARTITIONERS, RUNNERS, build_from_cfg
from ais_bench.benchmark.utils.config.run import get_config_type
from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.logging.exceptions import PredictionInvalidException
from ais_bench.benchmark.utils.logging.error_codes import TMAN_CODES
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners import LocalRunner
from ais_bench.benchmark.tasks import OpenICLEvalTask, OpenICLApiInferTask, OpenICLInferTask
from ais_bench.benchmark.summarizers import DefaultSummarizer, DefaultPerfSummarizer
from ais_bench.benchmark.calculators import DefaultPerfMetricCalculator
from ais_bench.benchmark.cli.utils import fill_model_path_if_datasets_need
from ais_bench.benchmark.utils.file.file import load_jsonl, dump_jsonl

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


class JudgeInfer(BaseWorker):
    def update_cfg(self, cfg: ConfigDict) -> None:
        def get_task_type() -> str:
            if cfg["datasets"][0]["judge_infer_cfg"]["judge_model"]["attr"] == "service":
                return get_config_type(OpenICLApiInferTask)
            else:
                return get_config_type(OpenICLInferTask)

        new_cfg = dict(
            judge_infer=dict(
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
            cfg.judge_infer.runner.debug = True
        cfg.judge_infer.partitioner["out_dir"] = osp.join(cfg["work_dir"], "predictions/")
        return cfg

    def do_work(self, cfg: ConfigDict):
        partitioner = PARTITIONERS.build(cfg.judge_infer.partitioner)
        logger.info("Starting inference tasks...")
        self._cfg_pre_process(cfg)
        tasks = partitioner(cfg)

        # delete the tasks without judge_infer_cfg
        new_tasks = []
        for task in tasks:
            if task["datasets"][0][0].get("judge_infer_cfg"):
                new_tasks.append(task)
        tasks = new_tasks
        if len(tasks) == 0:
            return

        # update tasks cfg before run
        self._update_tasks_cfg(tasks, cfg)

        if (
            cfg.get("cli_args", {}).get("merge_ds", False)
            or cfg.get("cli_args", {}).get("mode") == "perf" # performance mode will enable merge datasets by default
        ):
            logger.info("Merging datasets with the same model and inferencer...")
            tasks = self._merge_datasets(tasks)

        runner = RUNNERS.build(cfg.judge_infer.runner)
        runner(tasks)
        self._result_post_process(tasks, cfg)
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

    def _cfg_pre_process(self, cfg: ConfigDict) -> None:
        self.org_dataset_abbrs = {}
        def change_judge_dataset_abbr(item):
            if item.get("judge_infer_cfg"):
                org_dataset_abbr = item["abbr"]
                new_dataset_abbr = f'{item["abbr"]}-{item["judge_infer_cfg"]["judge_model"]["abbr"]}'
                item["abbr"] = new_dataset_abbr
                self.org_dataset_abbrs[new_dataset_abbr] = org_dataset_abbr
        if cfg.get('model_dataset_combinations', None) is not None:
            for item in cfg.model_dataset_combinations:
                for dataset in item["datasets"]:
                    change_judge_dataset_abbr(dataset)
        for dataset in cfg.datasets:
            change_judge_dataset_abbr(dataset)
        return cfg

    def _update_tasks_cfg(self, tasks, cfg: ConfigDict):
        # update parameters to correct sub cfg
        if hasattr(cfg, "attack"):
            for task in tasks:
                cfg.attack.dataset = task.datasets[0][0].abbr
                task.attack = cfg.attack

        # update judge cfgs to model cfgs and data
        for task in tasks:
            task["datasets"][0][0]["predictions_path"] = osp.join(cfg.judge_infer.partitioner.out_dir, task["models"][0]["abbr"], f'{self.org_dataset_abbrs[task["datasets"][0][0]["abbr"]]}.jsonl')
            if not osp.exists(task["datasets"][0][0]["predictions_path"]):
                raise PredictionInvalidException(TMAN_CODES.UNKNOWN_ERROR, f"Predictions path {task['datasets'][0][0]['predictions_path']} does not exist.")
            model_abbr = task["models"][0]["abbr"]
            task["models"][0] = task["datasets"][0][0]["judge_infer_cfg"].pop("judge_model")
            task["models"][0]["abbr"] = model_abbr
            task["datasets"][0][0]["type"] = task["datasets"][0][0]["judge_infer_cfg"].pop("judge_dataset_type")
            task["datasets"][0][0]["reader_cfg"] = task["datasets"][0][0]["judge_infer_cfg"].pop("judge_reader_cfg")
            task["datasets"][0][0]["infer_cfg"] = task["datasets"][0][0].pop("judge_infer_cfg")

    def _result_post_process(self, tasks, cfg: ConfigDict):
        # Reconstruct the judge infer predictions to normal predictions format
        for task in tasks:
            model_org_prediction_path = task["datasets"][0][0]["predictions_path"]
            model_preds: dict = {item["uuid"]: item for item in load_jsonl(model_org_prediction_path)}
            judge_org_prediction_path = osp.join(cfg.judge_infer.partitioner.out_dir, task["models"][0]["abbr"], f'{task["datasets"][0][0]["abbr"]}.jsonl')
            judge_preds: list = load_jsonl(judge_org_prediction_path)
            for i, pred in enumerate(judge_preds):
                uuid = pred["gold"]
                judge_preds[i]["id"] = model_preds[uuid]["id"]
            os.remove(judge_org_prediction_path)
            dump_jsonl(judge_preds, judge_org_prediction_path)


class Eval(BaseWorker):
    def update_cfg(self, cfg: ConfigDict) -> None:
        new_cfg = dict(
            eval=dict(
                partitioner=dict(type=get_config_type(NaivePartitioner)),
                runner=dict(
                    max_num_workers=self.args.max_num_workers,
                    debug=self.args.debug,
                    task=dict(type=get_config_type(OpenICLEvalTask)),
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
        self._cfg_pre_process(cfg)

        tasks = partitioner(cfg)

        # Update tasks cfg before run
        self._update_tasks_cfg(tasks, cfg)

        runner = RUNNERS.build(cfg.eval.runner)
        # For meta-review-judge in subjective evaluation
        if isinstance(tasks, list) and len(tasks) != 0 and isinstance(tasks[0], list):
            for task_part in tasks:
                runner(task_part)
        else:
            runner(tasks)
        self._result_post_process(tasks, cfg)
        logger.info("Evaluation tasks completed.")

    def _cfg_pre_process(self, cfg: ConfigDict) -> None:
        self.org_dataset_abbrs = {}
        def change_eval_dataset_abbr(item):
            if item.get("judge_infer_cfg"):
                org_dataset_abbr = item["abbr"]
                new_dataset_abbr = f'{item["abbr"]}-{item["judge_infer_cfg"]["judge_model"]["abbr"]}'
                item["abbr"] = new_dataset_abbr
                self.org_dataset_abbrs[new_dataset_abbr] = org_dataset_abbr
        if cfg.get('model_dataset_combinations', None) is not None:
            for item in cfg.model_dataset_combinations:
                for dataset in item["datasets"]:
                    change_eval_dataset_abbr(dataset)
        for dataset in cfg.datasets:
            change_eval_dataset_abbr(dataset)
        return cfg

    def _update_tasks_cfg(self, tasks, cfg: ConfigDict):
        # Replace default model config to judge model config
        for task in tasks:
            if task["datasets"][0][0].get("judge_infer_cfg"):
                task["datasets"][0][0].pop("judge_infer_cfg")

    def _result_post_process(self, tasks, cfg: ConfigDict):
        # Copy judge infer result to normal name

        for task in tasks:
            if task["datasets"][0][0]["abbr"] in self.org_dataset_abbrs.keys():
                cur_results_path = osp.join(cfg.eval.partitioner.out_dir, task["models"][0]["abbr"], f'{task["datasets"][0][0]["abbr"]}.jsonl')
                final_org_results_path = osp.join(cfg.eval.partitioner.out_dir, task["models"][0]["abbr"], f'{self.org_dataset_abbrs[task["datasets"][0][0]["abbr"]]}.jsonl')
                if os.path.exists(final_org_results_path):
                    os.remove(final_org_results_path)

                if os.path.exists(cur_results_path):
                    # 基于cur_results_path的文件复制一份final_org_results_path
                    shutil.copy(cur_results_path, final_org_results_path)


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
        cfg = self._cfg_pre_process(cfg)

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

    def _cfg_pre_process(self, cfg: ConfigDict) -> None:
        for i, dataset in enumerate(cfg.datasets):
            if dataset.get("judge_infer_cfg"):
                cfg.datasets[i]["abbr"] = f'{cfg.datasets[i]["abbr"]}-{cfg.datasets[i]["judge_infer_cfg"]["judge_model"]["abbr"]}'
                cfg.datasets[i].pop("judge_infer_cfg")
        return cfg


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
    all=[Infer, JudgeInfer, Eval, AccViz],
    infer=[Infer],
    eval=[JudgeInfer, Eval, AccViz],
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
