from ais_bench.benchmark.cli.argument_parser import ArgumentParser
from ais_bench.benchmark.utils.logging.logger import AISLogger


class TaskManager:
    def __init__(self) -> None:
        self.logger = AISLogger()
        self.args_parser = ArgumentParser()
        self.args = self.args_parser.parse_args()

        # To accelerate cmd "ais_bench -h", we only import config_manager and work_flow after argparser
        from ais_bench.benchmark.cli.config_manager import ConfigManager


        self.config_manager = ConfigManager(self.args)


    def run(self) -> None:
        # search
        if self.args.search:
            self.config_manager.search_configs_location()
            return

        from ais_bench.benchmark.cli.workers import WORK_FLOW, AgentXViz, WorkFlowExecutor
        run_mode = self.args.mode
        # --reuse 只做一件事：指定 --work-dir 下已有的时间戳目录，继续执行并覆盖原结果。
        # 是否执行、是否只出汇总，由 --mode 决定
        if run_mode == "perf" and self.args.reuse:
            self.logger.warning(
                "Detected --reuse in perf mode. Execution will be skipped and existing "
                "results will be loaded from the reuse work dir."
            )
            run_mode = "perf_viz"
        self.workflow = [worker_class(self.args) for worker_class in WORK_FLOW.get(run_mode)]
        if self.args.mode in ("agentx", "agentx_raw") and self.args.dry_run:
            # dry-run 只输出执行计划，不尝试汇总不存在的正式结果。
            self.workflow = [worker for worker in self.workflow
                             if not isinstance(worker, AgentXViz)]

        # load config
        cfg = self.config_manager.load_config(self.workflow)

        # run
        workflow_executor = WorkFlowExecutor(cfg, self.workflow)
        workflow_executor.execute()