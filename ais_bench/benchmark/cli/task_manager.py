from ais_bench.benchmark.cli.argument_parser import ArgumentParser


class TaskManager:
    def __init__(self) -> None:
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
        self.workflow = [worker_class(self.args) for worker_class in WORK_FLOW.get(self.args.mode)]

        # load config
        cfg = self.config_manager.load_config(self.workflow)

        # run
        workflow_executor = WorkFlowExecutor(cfg, self.workflow)
        workflow_executor.execute()