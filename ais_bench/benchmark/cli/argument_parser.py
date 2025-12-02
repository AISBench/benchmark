import argparse
from ais_bench.benchmark.cli.utils import (
    get_current_time_str,
    validate_max_workers,
    validate_max_workers_per_gpu,
    validate_num_prompts,
    validate_num_warmups,
    validate_pressure_time
)
from ais_bench.benchmark.cli.utils import DEFAULT_PRESSURE_TIME


class ArgumentParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Run a benchmark task')
        self.parser.add_argument('config', nargs='?', help='Benchmark config file path')
        self._base_parser()
        self._perf_parser()
        self._accuracy_parser()
        self._custom_dataset_parser()

    def parse_args(self):
        args = self.parser.parse_args()
        args.cfg_time_str = args.dir_time_str = get_current_time_str()
        if args.dry_run:
            args.debug = True
        return args

    def _base_parser(self):
        """These args are all for the base configuration."""
        parser = self.parser.add_argument_group('base_args')
        parser.add_argument('--models', nargs='+', help='', default=None)
        parser.add_argument('--datasets', nargs='+', help='', default=None)
        parser.add_argument('--summarizer', help='', default=None)
        parser.add_argument(
            '--debug',
            help='Debug mode, in which scheduler will run tasks '
            'in the single process, and output will not be '
            'redirected to files',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '-s',
            '--search',
            help='Searching for the configs abs paths of --models --datasets and --summarizer',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--dry-run',
            help='Dry run mode, in which the scheduler will not '
            'actually run the tasks, but only print the commands '
            'to run',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '-m',
            '--mode',
            help='Running mode. Choose "perf" for performance evaluation, "infer" to run inference only, '
            '"eval" to evaluate existing inference results, or "viz" to visualize the results. '
            'The default mode is "all", which runs all steps.',
            choices=['all', 'infer', 'eval', 'viz', 'perf', 'perf_viz'],
            default='all',
            type=str
        )
        parser.add_argument(
            '-r',
            '--reuse',
            nargs='?',
            type=str,
            const='latest',
            help='Reuse previous outputs & results, and run any '
            'missing jobs presented in the config. If its '
            'argument is not specified, the latest results in '
            'the work_dir will be reused. The argument should '
            'also be a specific timestamp, e.g. 20230516_144254'
        )
        parser.add_argument(
            '-w',
            '--work-dir',
            help='Work path, all the outputs will be '
            'saved in this path, including the predictions, '
            'the evaluation results, the summary results, etc.'
            'If not specified, the work_dir will be set to '
            'outputs/default.',
            default=None,
            type=str
        )
        parser.add_argument(
            '--config-dir',
            default='configs',
            help='Use the custom config directory instead of config/ to '
            'search the configs for datasets, models and summarizers',
            type=str
        )
        parser.add_argument(
            '--max-num-workers',
            help='Max number of workers to run in parallel. ',
            type=validate_max_workers,
            default=1
        )
        parser.add_argument(
            '--max-workers-per-gpu',
            help='Max task to run in parallel on one GPU. '
            'It will only be used in the local runner.',
            type=validate_max_workers_per_gpu,
            default=1
        )
        parser.add_argument(
            '--num-prompts',
            help='Num Prompts, Specify the number of prompts to infer and evaluate. '
            'Must be integer >= 1, If not provided, all prompts will be inferred and evaluated. ',
            type=validate_num_prompts,
            default=None
        )
        parser.add_argument(
            '--num-warmups',
            help='Number of warmups, Specify the number of warmups. '
            'Must be integer >= 0, use 0 to disable warmups. If not provided, the default is 1. ',
            type=validate_num_warmups,
            default=1
        )

    def _accuracy_parser(self):
        """These args are all for the accuracy evaluation."""
        parser = self.parser.add_argument_group('accuracy_args')
        parser.add_argument(
            '--merge-ds',
            help='Whether to merge dataset with multi files(mmlu, ceval)',
            action='store_true',
        )
        parser.add_argument(
            '--dump-eval-details',
            help='Whether to dump the evaluation details, including the '
            'correctness of each sample, bpb, etc.',
            action='store_true',
        )
        parser.add_argument(
            '--dump-extract-rate',
            help='Whether to dump the extract rate of evaluation (samples per sec)',
            action='store_true',
        )

    def _perf_parser(self):
        """These args are all for the performance benchmark."""
        parser = self.parser.add_argument_group('perf_args')
        parser.add_argument(
            '--pressure',
            help='Whether to enable pressure test in perf mode (only attr service)',
            action='store_true',
        )
        parser.add_argument(
            '--pressure-time',
            help=f'Pressure test time, only valid when --pressure is True.Must be integer >= 1, default is {DEFAULT_PRESSURE_TIME} seconds',
            type=validate_pressure_time,
            default=DEFAULT_PRESSURE_TIME
        )

    def _custom_dataset_parser(self):
        """These args are all for the quick construction of custom datasets."""
        parser = self.parser.add_argument_group('custom_dataset_args')
        parser.add_argument('--custom-dataset-path', type=str)
        parser.add_argument('--custom-dataset-meta-path', type=str)
        parser.add_argument('--custom-dataset-data-type',
                            type=str,
                            choices=['mcq', 'qa'])
        parser.add_argument('--custom-dataset-infer-method',
                            type=str,
                            choices=['gen'])



