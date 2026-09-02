import glob
import os
import os.path as osp
import copy
import shutil
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from mmengine.config import ConfigDict

from ais_bench.benchmark.registry import PARTITIONERS, RUNNERS, build_from_cfg
from ais_bench.benchmark.utils.config.run import get_config_type
from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.logging.exceptions import PredictionInvalidException
from ais_bench.benchmark.utils.logging.error_codes import TMAN_CODES
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners import LocalRunner, TasksMonitor
from ais_bench.benchmark.tasks import OpenICLEvalTask, OpenICLApiInferTask, OpenICLInferTask
from ais_bench.benchmark.tasks.base import EmptyTask
from ais_bench.benchmark.summarizers import DefaultSummarizer, DefaultPerfSummarizer
from ais_bench.benchmark.calculators import DefaultPerfMetricCalculator
from ais_bench.benchmark.cli.utils import clear_repeat_tasks
from ais_bench.benchmark.utils.file.file import load_jsonl, dump_jsonl
from ais_bench.benchmark.utils.response_anomaly import (
    ANOMALY_RESULT_NAMES,
    ResponseAnomalyCoordinator,
)

logger = AISLogger()


class _URLSnapshotEntry(NamedTuple):
    """A single URL's before-snapshot state."""
    before: Any = None   # SpecDecodeSnapshot | None
    error: str | None = None


@dataclass
class _SpecDecodeContext:
    """Carries state between before/after spec decode Prometheus snapshots.

    Each unique metrics URL gets its own entry so that spec decode metrics
    from different servers are collected and reported independently.
    """
    enabled: bool = False
    entries: dict[str, _URLSnapshotEntry] = field(default_factory=dict)


def _run_response_anomaly_monitor(
    task_names: list, work_dir: str, is_debug: bool
) -> None:
    """Run a dedicated status board for the response anomaly task."""
    tasks_monitor = TasksMonitor(
        task_names,
        work_dir,
        is_debug,
        include_anomaly_status=True,
    )
    tasks_monitor.launch_state_board()


class BaseWorker(ABC):
    def __init__(self, args) -> None:
        self.args = args
        self.skip = False

    @abstractmethod
    def update_cfg(self, cfg: ConfigDict) -> None:
        # update major cfg content according to worker kind
        pass

    @abstractmethod
    def do_work(self, cfg: ConfigDict):
        # run partitioner and launch runner
        pass


class Infer(BaseWorker):
    def update_cfg(self, cfg: ConfigDict) -> ConfigDict:
        def get_task_type() -> str:
            if cfg["models"][0]["attr"] == "service":
                return OpenICLApiInferTask
            else:
                return OpenICLInferTask

        custom_infer = cfg.get("infer")
        custom_task = None
        if custom_infer:
            custom_task = custom_infer.get("runner", {}).get("task", {}).get("type")
            if custom_task == EmptyTask:
                self.skip = True
                return cfg

        def update_new_infer_cfg(new_cfg: ConfigDict) -> None:
            runner_cfg = new_cfg['infer']['runner']
            runner_cfg['max_num_workers'] = self.args.max_num_workers
            runner_cfg['max_workers_per_gpu'] = self.args.max_workers_per_gpu
            runner_cfg['debug'] = self.args.debug or cfg.cli_args.debug

        if cfg.get('infer'):
            new_cfg = dict(infer=cfg.infer)
            if not new_cfg["infer"].get("partitioner"):
                new_cfg["infer"]["partitioner"] = dict(type=NaivePartitioner)
            if new_cfg["infer"].get("runner") and new_cfg["infer"]["runner"].get("type") is None:
                new_cfg["infer"]["runner"]["type"] = LocalRunner
        else:
            new_cfg = dict(
                infer=dict(
                    partitioner=dict(type=NaivePartitioner),
                    runner=dict(
                        task=dict(type=custom_task if custom_task else get_task_type()),
                        type=LocalRunner,
                    ),
                ),
            )
        update_new_infer_cfg(new_cfg)
        cfg.merge_from_dict(new_cfg)
        cfg.infer.partitioner["out_dir"] = osp.join(cfg["work_dir"], "predictions/")
        return cfg

    def do_work(self, cfg: ConfigDict):
        if self.skip:
            logger.info("EmptyTask is selected, skip inference.")
            return
        partitioner = PARTITIONERS.build(cfg.infer.partitioner)
        logger.info("Starting inference tasks...")
        tasks = partitioner(cfg)
        tasks = clear_repeat_tasks(tasks)

        # update tasks cfg before run
        self._update_tasks_cfg(tasks, cfg)

        if (
            cfg.get("cli_args", {}).get("merge_ds", False)
            or cfg.get("cli_args", {}).get("mode") == "perf" # performance mode will enable merge datasets by default
        ):
            logger.info("Merging datasets with the same model and inferencer...")
            tasks = self._merge_datasets(tasks)

        spec_ctx = self._spec_decode_before_snapshot(cfg)

        if cfg.get('response_anomaly', {}).get('enabled', False):
            # Remove a stale status left by a previous interrupted run so the
            # inference board does not wait on an outdated ResponseAnomaly state.
            stale_status = osp.join(
                cfg['work_dir'],
                'status_tmp',
                ResponseAnomalyCoordinator.STATUS_FILE_NAME,
            )
            try:
                if os.path.isfile(stale_status):
                    os.remove(stale_status)
            except OSError as exc:
                logger.warning(
                    "Failed to remove stale response anomaly status file %s: %s",
                    stale_status,
                    exc,
                )

        runner = RUNNERS.build(cfg.infer.runner)
        runner(tasks)

        self._spec_decode_finalize(cfg, spec_ctx)

        if cfg.get('response_anomaly', {}).get('enabled', False):
            logger.info(
                "Inference finished; starting response anomaly detection "
                "(bound to the inference stage)..."
            )
            self.response_anomaly_coordinator.start(cfg)
            # Detection runs serially inside the inference stage: wait for
            # it to finish (its status board prints between the inference
            # board and any evaluation board) before the workflow continues.
            _finalize_response_anomaly_detection(
                self.response_anomaly_coordinator,
                cfg['work_dir'],
                cfg.get('cli_args', {}).get('debug', False),
            )
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

    # ------------------------------------------------------------------
    #  Speculative Decoding — per-URL before/after snapshot methods
    # ------------------------------------------------------------------

    def _spec_decode_before_snapshot(self, cfg: ConfigDict) -> _SpecDecodeContext:
        """Fetch before-snapshots for every unique metrics URL."""
        cli_args = cfg.get("cli_args", {})
        ctx = _SpecDecodeContext()

        ctx.enabled = (
            cli_args.get("spec_decode", False)
            and cli_args.get("mode") == "perf"
        )
        if cli_args.get("spec_decode") and cli_args.get("mode") != "perf":
            logger.warning(
                "--spec-decode is only effective in --mode perf. "
                "Ignoring spec decode for current mode '%s'.",
                cli_args.get("mode"),
            )

        if not ctx.enabled:
            return ctx

        from ais_bench.benchmark.spec_decode.urls import resolve_metrics_urls

        urls = resolve_metrics_urls(cfg.get("models", []))
        if not urls:
            ctx.enabled = False
            logger.info("Spec decode before-snapshot failed: no metrics URLs found.")
            return ctx

        from ais_bench.benchmark.spec_decode.fetcher import (
            fetch_spec_decode_metrics_with_error,
        )
        for url in urls:
            snapshot, error = fetch_spec_decode_metrics_with_error(url)
            ctx.entries[url] = _URLSnapshotEntry(before=snapshot, error=error)
            if error:
                logger.info(
                    "Spec decode [%s] before-snapshot failed: %s", url, error
                )
            else:
                logger.info(
                    "Spec decode [%s] before-snapshot captured successfully.", url
                )
        return ctx

    def _spec_decode_finalize(
        self, cfg: ConfigDict, ctx: _SpecDecodeContext
    ) -> None:
        """Finalize: collect after-snapshots, compute deltas, save results."""
        if not ctx.enabled:
            return

        for url, entry in ctx.entries.items():
            try:
                self._process_spec_decode_url(cfg, url, entry)
            except Exception:
                logger.warning(
                    "Spec decode [%s] after-snapshot failed unexpectedly, skipping",
                    url, exc_info=True,
                )

    def _process_spec_decode_url(
        self, cfg: ConfigDict, url: str, entry: _URLSnapshotEntry
    ) -> None:
        """After-snapshot → compute delta → save for a single URL."""
        from ais_bench.benchmark.spec_decode.fetcher import (
            fetch_spec_decode_metrics_with_error,
        )
        from ais_bench.benchmark.spec_decode.calculator import (
            compute_spec_decode_stats,
        )
        from ais_bench.benchmark.spec_decode.reporter import (
            save_spec_decode_result,
        )

        after_snapshot, after_error = fetch_spec_decode_metrics_with_error(url)
        error = self._merge_spec_decode_errors(entry.error, after_error)

        spec_stats = None
        if entry.before is not None and after_snapshot is not None:
            spec_stats = compute_spec_decode_stats(entry.before, after_snapshot)
            if spec_stats is None:
                error = "No spec decode activity detected during benchmark window"

        self._log_spec_decode_result(url, spec_stats, error)

        save_spec_decode_result(
            spec_stats, error, cfg["work_dir"], url,
            before_snapshot=entry.before,
            after_snapshot=after_snapshot,
        )

    @staticmethod
    def _merge_spec_decode_errors(before_error: str | None, after_error: str | None) -> str | None:
        """Merge before/after error messages, preserving both when possible."""
        if not after_error:
            return before_error
        if before_error:
            return f"{before_error}; {after_error}"
        return after_error

    @staticmethod
    def _log_spec_decode_result(url: str, spec_stats: dict | None, error: str | None) -> None:
        """Log the outcome of spec decode collection for a single URL."""
        if spec_stats:
            logger.info(
                "Spec decode [%s] collected: acceptance_rate=%.2f%%, "
                "acceptance_length=%.2f",
                url,
                spec_stats["acceptance_rate"],
                spec_stats["acceptance_length"],
            )
        else:
            logger.info(
                "Spec decode [%s] unavailable: %s",
                url, error or "unknown reason",
            )



class JudgeInfer(BaseWorker):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.judge_model_type = None

    def update_cfg(self, cfg: ConfigDict) -> ConfigDict:
        for dataset_cfg in cfg["datasets"]:
            judge_infer_cfg = dataset_cfg.get("judge_infer_cfg")
            if judge_infer_cfg:
                self.judge_model_type = judge_infer_cfg["judge_model"]["attr"]

        if self.judge_model_type is None:
            logger.debug("Skip Judge Infer")
            return cfg

        def get_task_type() -> str:
            if self.judge_model_type == "service":
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
        if self.judge_model_type is None:
            logger.debug("Skip Judge Infer")
            return

        partitioner = PARTITIONERS.build(cfg.judge_infer.partitioner)
        logger.info("Starting inference tasks...")
        self._cfg_pre_process(cfg)
        tasks = partitioner(cfg)
        tasks = clear_repeat_tasks(tasks)

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
        self._results_pre_process(tasks, cfg)
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
            task["datasets"] = copy.deepcopy(task["datasets"])
            task["models"] = copy.deepcopy(task["models"])
            task["datasets"][0][0]["predictions_path"] = osp.join(cfg.judge_infer.partitioner.out_dir, task["models"][0]["abbr"], f'{self.org_dataset_abbrs[task["datasets"][0][0]["abbr"]]}.jsonl')
            if not osp.exists(task["datasets"][0][0]["predictions_path"]):
                raise PredictionInvalidException(TMAN_CODES.UNKNOWN_ERROR, f"Predictions path {task['datasets'][0][0]['predictions_path']} does not exist.")
            model_abbr = task["models"][0]["abbr"]
            task["models"][0] = task["datasets"][0][0]["judge_infer_cfg"].pop("judge_model")
            task["models"][0]["abbr"] = model_abbr
            task["datasets"][0][0]["type"] = task["datasets"][0][0]["judge_infer_cfg"].pop("judge_dataset_type")
            task["datasets"][0][0]["reader_cfg"] = task["datasets"][0][0]["judge_infer_cfg"].pop("judge_reader_cfg")
            task["datasets"][0][0]["infer_cfg"] = task["datasets"][0][0].pop("judge_infer_cfg")

    def _results_pre_process(self, tasks, cfg: ConfigDict):
        # Copy the original judge infer predictions to cached predictions
        for task in tasks:
            judge_org_prediction_path = osp.join(cfg.judge_infer.partitioner.out_dir, task["models"][0]["abbr"], f'{task["datasets"][0][0]["abbr"]}.jsonl')
            cache_model_org_prediction_path = osp.join(cfg.judge_infer.partitioner.out_dir, task["models"][0]["abbr"], f'{self.org_dataset_abbrs[task["datasets"][0][0]["abbr"]]}-cached.jsonl')
            if osp.exists(judge_org_prediction_path):
                os.remove(judge_org_prediction_path)
            if osp.exists(cache_model_org_prediction_path):
                shutil.copy(cache_model_org_prediction_path, judge_org_prediction_path)
                os.remove(cache_model_org_prediction_path)

    def _result_post_process(self, tasks, cfg: ConfigDict):
        # Reconstruct the judge infer predictions to normal predictions format
        for task in tasks:
            model_org_prediction_path = task["datasets"][0][0]["predictions_path"]
            model_preds: dict = {item["uuid"]: item for item in load_jsonl(model_org_prediction_path)}
            judge_org_prediction_path = osp.join(cfg.judge_infer.partitioner.out_dir, task["models"][0]["abbr"], f'{task["datasets"][0][0]["abbr"]}.jsonl')
            judge_preds: list = load_jsonl(judge_org_prediction_path)
            cache_judge_org_preds_path = osp.join(cfg.judge_infer.partitioner.out_dir, task["models"][0]["abbr"], f'{task["datasets"][0][0]["abbr"]}-cached.jsonl')
            shutil.copy(judge_org_prediction_path, cache_judge_org_preds_path)
            for i, pred in enumerate(judge_preds):
                uuid = pred["gold"]
                judge_preds[i]["id"] = model_preds[uuid]["id"]
            os.remove(judge_org_prediction_path)
            dump_jsonl(judge_preds, judge_org_prediction_path)


class Eval(BaseWorker):
    def update_cfg(self, cfg: ConfigDict) -> ConfigDict:
        custom_eval = cfg.get("eval")
        custom_task = None
        if custom_eval:
            custom_task = custom_eval.get("runner", {}).get("task", {}).get("type")
            if custom_task == EmptyTask:
                self.skip = True
                return cfg

        def update_eval_cfg(new_cfg: ConfigDict) -> None:
            runner_cfg = new_cfg['eval']['runner']
            runner_cfg['max_num_workers'] = self.args.max_num_workers
            runner_cfg['max_workers_per_gpu'] = self.args.max_workers_per_gpu
            runner_cfg['debug'] = self.args.debug or cfg.cli_args.debug
            runner_cfg['task']['dump_details'] = cfg.cli_args.dump_eval_details
            runner_cfg['task']['cal_extract_rate'] = cfg.cli_args.dump_extract_rate

        if cfg.get('eval'):
            new_cfg = dict(eval=cfg.eval)
            if not new_cfg["eval"].get("partitioner"):
                new_cfg["eval"]["partitioner"] = dict(type=NaivePartitioner)
            if new_cfg["eval"].get("runner") and new_cfg["eval"]["runner"].get("type") is None:
                new_cfg["eval"]["runner"]["type"] = LocalRunner
        else:
            new_cfg = dict(
                eval=dict(
                    partitioner=dict(type=NaivePartitioner),
                    runner=dict(
                        type=LocalRunner,
                        task=dict(type=custom_task if custom_task else OpenICLEvalTask),
                    ),
                )
            )

        update_eval_cfg(new_cfg)
        cfg.merge_from_dict(new_cfg)
        cfg.eval.partitioner["out_dir"] = osp.join(cfg["work_dir"], "results/")
        return cfg

    def do_work(self, cfg: ConfigDict):
        if self.skip:
            logger.info("EmptyTask is selected, skip evaluation.")
            return
        partitioner = PARTITIONERS.build(cfg.eval.partitioner)
        logger.info("Starting evaluation tasks...")
        self._cfg_pre_process(cfg)

        tasks = partitioner(cfg)
        tasks = clear_repeat_tasks(tasks)

        # Update tasks cfg before run
        self._update_tasks_cfg(tasks, cfg)

        runner = RUNNERS.build(cfg.eval.runner)
        # For meta-review-judge in subjective evaluation
        if isinstance(tasks, list) and len(tasks) != 0 and isinstance(tasks[0], list):
            for task_part in tasks:
                runner(task_part)
        else:
            runner(tasks)
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
        self.judge_result_paths = {}
        for task in tasks:
            task["datasets"] = copy.deepcopy(task["datasets"])
            if task["datasets"][0][0].get("judge_infer_cfg"):
                task["datasets"][0][0].pop("judge_infer_cfg")


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

        # ========== Speculative Decoding Results ==========
        if cfg.get("cli_args", {}).get("spec_decode", False):
            self._output_spec_decode_results(cfg)

    @staticmethod
    def _output_spec_decode_results(cfg: ConfigDict) -> None:
        """Read all per-URL spec_decode_*.json files and print results."""
        from ais_bench.benchmark.spec_decode.reporter import (
            format_spec_decode_console,
            format_spec_decode_na,
        )

        pattern = osp.join(cfg["work_dir"], "performances", "spec_decode_*.json")
        spec_files = sorted(glob.glob(pattern))

        if not spec_files:
            logger.warning(
                "Spec decode enabled but no result files found matching %s",
                pattern,
            )
            return

        for spec_file in spec_files:
            try:
                with open(spec_file, "r", encoding="utf-8") as f:
                    result = json.load(f)
            except Exception:
                logger.warning(
                    "Failed to read spec decode result file %s, skipping",
                    spec_file, exc_info=True,
                )
                continue

            url = result.get("url", "")
            if result.get("status") == "ok" and result.get("data"):
                print(format_spec_decode_console(result["data"], url))
            else:
                print(format_spec_decode_na(url, result.get("error")))


def _finalize_response_anomaly_detection(
    coordinator, work_dir: str, is_debug: bool
) -> None:
    """Wait for detection to finish and print its status board and summary.

    Called from the Infer worker so detection is serially bound to the
    inference stage: the dedicated board renders right after the inference
    board and before evaluation starts.
    """
    # The dedicated board is the only place detection status is rendered now
    # that evaluation boards stay separate. Start it whenever detection has
    # produced a status (it may have already finished for tiny datasets), so
    # the final table is always printed. Skip it when no status was ever
    # written to avoid the board waiting forever on tasks that never started.
    anomaly_status_file = osp.join(
        work_dir,
        'status_tmp',
        ResponseAnomalyCoordinator.STATUS_FILE_NAME,
    )
    if coordinator.is_running or osp.isfile(anomaly_status_file):
        _run_response_anomaly_monitor(
            coordinator.task_names,
            work_dir,
            is_debug,
        )
    coordinator.join()
    TasksMonitor.rm_tmp_files(work_dir)
    if coordinator.summary:
        logger.info(
            "Response anomaly detection summary across %d task(s): %s",
            len(coordinator.anomaly_report),
            coordinator.summary,
        )
    for task_name, info in coordinator.anomaly_report.items():
        counts = info.get("counts", {})
        anomalies = {
            name: count
            for name, count in counts.items()
            if name in ANOMALY_RESULT_NAMES and count
        }
        if anomalies:
            logger.warning(
                "Response anomalies detected for %s: %s",
                task_name,
                anomalies,
            )
            logger.warning("  detection results: %s", info.get("result_file"))
            if info.get("payload_dir"):
                logger.warning("  payload archive:   %s", info["payload_dir"])
            logger.warning("  task log:          %s", info.get("task_log"))
        undetected = {
            name: count
            for name, count in counts.items()
            if name in ("failed", "unavailable") and count
        }
        if undetected:
            logger.warning(
                "Response anomaly detection did not complete for %s: %s. "
                "Check the task log for the root cause.",
                task_name,
                undetected,
            )
            logger.warning("  task log:          %s", info.get("task_log"))


WORK_FLOW = dict(
    all=[Infer, JudgeInfer, Eval, AccViz],
    infer=[Infer],
    judge=[JudgeInfer],
    infer_judge=[Infer, JudgeInfer],
    eval=[JudgeInfer, Eval, AccViz],
    viz=[AccViz],
    perf=[Infer, PerfViz],
    perf_viz=[PerfViz],
)


class WorkFlowExecutor:
    def __init__(self, cfg, workflow) -> None:
        self.cfg = cfg
        self.workflow = workflow
        self.response_anomaly_coordinator = ResponseAnomalyCoordinator()

    def execute(self) -> None:
        for worker in self.workflow:
            worker.response_anomaly_coordinator = self.response_anomaly_coordinator
            cfg = copy.deepcopy(self.cfg)
            worker.do_work(cfg)
