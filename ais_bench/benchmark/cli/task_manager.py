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

        from ais_bench.benchmark.cli.workers import WORK_FLOW, WorkFlowExecutor
        run_mode = self.args.mode
        if run_mode == "perf" and self.args.reuse:
            self.logger.warning(
                "Detected --reuse in performance mode. The inference stage will be skipped, "
                f"and performance metrics will be loaded from the reuse work dir."
            )
            run_mode = "perf_viz"
        if self.args.config and run_mode == "all":
            try:
                from mmengine.config import Config
                peek_cfg = Config.fromfile(self.args.config, format_python_code=False)
                if "infer" not in peek_cfg:
                    run_mode = "eval"
                    self.logger.info(
                        f"Config has no infer section, defaulting to mode '{run_mode}'"
                    )
            except Exception:
                pass
        self.workflow = [worker_class(self.args) for worker_class in WORK_FLOW.get(run_mode)]

        # load config
        cfg = self.config_manager.load_config(self.workflow)

        # run
        workflow_executor = WorkFlowExecutor(cfg, self.workflow)
        workflow_executor.execute()