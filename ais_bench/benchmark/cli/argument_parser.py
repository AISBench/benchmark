import argparse
import json
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
        self._agent_parser()
        self._api_model_parser()

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
            choices=['all', 'infer', 'eval', 'viz', 'perf', 'perf_viz', 'judge', 'infer_judge',
                     'agent', 'agent_viz'],
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
        parser.add_argument(
            '--response-anomaly',
            action='store_true',
            default=False,
            help='Enable msProbe response anomaly detection. This command-line '
            'switch is the only way to enable the feature; detection is disabled '
            'when the flag is absent (response_anomaly.enabled in the config '
            'file is not supported and is ignored). '
            'Only supported in all/infer/infer_judge modes; perf and Agent modes are unsupported.'
        )
        parser.add_argument(
            '--response-anomaly-payload-retention',
            choices=('all', 'anomalies', 'none'),
            default=None,
            help='Select which response anomaly payloads to retain after detection. '
            'The command-line value overrides response_anomaly.payload_retention '
            'in the config file. Defaults to anomalies when neither is set.'
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
        parser.add_argument(
            '--spec-decode',
            help='Enable speculative decoding metrics collection. Only effective in --mode perf.',
            action='store_true',
            default=False,
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

    def _agent_parser(self):
        """These args are all for the harbor agent evaluation (mode=agent)."""
        parser = self.parser.add_argument_group('agent_args')
        parser.add_argument(
            '-a',
            '--agent',
            help='Agent name (see harbor AgentName) or a custom agent import '
            'path (module.path:ClassName)',
            type=str,
            default=None,
        )
        parser.add_argument(
            '--agent-import-path',
            help='Import path of a custom agent (module.path:ClassName)',
            type=str,
            default=None,
        )
        parser.add_argument(
            '--model',
            help='Model name for the agent (can be used multiple times)',
            nargs='+',
            type=str,
            default=None,
        )
        parser.add_argument(
            '--api-base',
            help='Model service base url (unified semantic parameter)',
            type=str,
            default=None,
        )
        parser.add_argument(
            '--api-key',
            help='API key for the model service',
            type=str,
            default=None,
        )
        parser.add_argument(
            '--ak',
            '--agent-kwarg',
            dest='agent_kwarg',
            help="Additional agent kwarg in 'key=value' format "
            "(can be used multiple times)",
            nargs='+',
            type=str,
            default=None,
        )
        parser.add_argument(
            '--ae',
            '--agent-env',
            dest='agent_env',
            help="Environment variable for the agent in 'KEY=VALUE' format "
            "(can be used multiple times)",
            nargs='+',
            type=str,
            default=None,
        )
        parser.add_argument(
            '--agent-deps',
            help='Path to an offline agent deps bundle (<agent>.tar.gz), or a '
            'directory of per-base-image bundles auto-matched at runtime',
            type=str,
            default=None,
        )
        parser.add_argument(
            '-p',
            '--path',
            help='Local dataset path (or a single task directory)',
            type=str,
            default=None,
        )
        parser.add_argument(
            '-d',
            '--dataset',
            help='Remote dataset name@version (registry or package)',
            type=str,
            default=None,
        )
        parser.add_argument(
            '-n',
            '--n-concurrent',
            help='Number of concurrent trials to run',
            type=int,
            default=None,
        )
        parser.add_argument(
            '-k',
            '--n-attempts',
            help='Number of attempts per trial',
            type=int,
            default=None,
        )
        parser.add_argument(
            '-e',
            '--environment',
            help='Harbor environment type (docker, daytona, e2b, modal, ...)',
            type=str,
            default=None,
        )
        parser.add_argument(
            '--timeout-multiplier',
            help='Multiplier for task timeouts',
            type=float,
            default=None,
        )
        parser.add_argument(
            '--max-retries',
            help='Maximum number of retry attempts',
            type=int,
            default=None,
        )
        parser.add_argument(
            '--include-task-name',
            help='Task name to include (supports glob, can be used multiple times)',
            nargs='+',
            type=str,
            default=None,
        )
        parser.add_argument(
            '--exclude-task-name',
            help='Task name to exclude (supports glob, can be used multiple times)',
            nargs='+',
            type=str,
            default=None,
        )
        parser.add_argument(
            '--n-tasks',
            help='Maximum number of tasks to include from the dataset',
            type=int,
            default=None,
        )
        parser.add_argument(
            '--disable-verification',
            help='Disable the verifier',
            action='store_true',
            default=None,
        )
        parser.add_argument(
            '--force-build/--no-force-build',
            help='Whether to force rebuild the environment',
            action=argparse.BooleanOptionalAction,
            default=None,
        )
        parser.add_argument(
            '--host-network',
            help='Run all task containers sharing the host network '
            '(docker-compose network_mode: host)',
            action='store_true',
            default=None,
        )
        parser.add_argument(
            '--delete/--no-delete',
            help='Whether to delete the environment after completion',
            action=argparse.BooleanOptionalAction,
            default=None,
        )
        parser.add_argument(
            '--purge-exception-cases',
            help='Before execution, delete all case dirs that ended with an '
            'exception (from each job result.json exception_stats), to '
            'automatically retry them. Only effective when --reuse is set.',
            action='store_true',
            default=False,
        )
        parser.add_argument(
            '-q',
            '--quiet',
            help='Suppress individual trial progress displays',
            action='store_true',
            default=None,
        )
        parser.add_argument(
            '-y',
            '--yes',
            help='Automatically confirm environment variable prompts',
            action='store_true',
            default=None,
        )
        parser.add_argument(
            '--env-file',
            help='Path to a .env file',
            type=str,
            default=None,
        )
        parser.add_argument(
            '--monitor-port',
            help='Port of the harbor monitor HTTP service (0 = disabled, default 0)',
            type=int,
            default=0,
        )
    def _api_model_parser(self):
        """API 模型通用参数，覆盖执行的所有模型配置（mindie_api/tgi_api/triton_api/vllm_api）。

        All args default to None so that they only take effect when explicitly
        specified on the command line.
        """
        parser = self.parser.add_argument_group('api_model_args')
        parser.add_argument('--path', type=str, default=None,
                            help='Path override for all API model configs')
        parser.add_argument('--model-name', type=str, default=None,
                            help='Model name override. Written to the `model` or `model_name` '
                            'field depending on the model type. Named --model-name to avoid '
                            'confusion with --models')
        parser.add_argument('--request-rate', type=float, default=None,
                            help='Request rate override in the range [0, 64000]')
        parser.add_argument('--retry', type=int, default=None,
                            help='Retry count override in the range [0, 1000]')
        parser.add_argument('--api-key', type=str, default=None,
                            help='API key override')
        parser.add_argument('--host-ip', type=str, default=None,
                            help='Host ip override')
        parser.add_argument('--host-port', type=int, default=None,
                            help='Host port override in the range (0, 65536)')
        parser.add_argument('--url', type=str, default=None,
                            help='Endpoint url override')
        parser.add_argument('--max-out-len', type=int, default=None,
                            help='Max output length override')
        parser.add_argument('--batch-size', type=int, default=None,
                            help='Batch size override in the range (0, 100000]')
        parser.add_argument(
            '--trust-remote-code',
            action=argparse.BooleanOptionalAction,
            default=None,
            help='Trust remote code override (--trust-remote-code / --no-trust-remote-code)'
        )
        parser.add_argument('--generation-kwargs', type=json.loads, default=None,
                            help='Generation kwargs override as a JSON object, '
                            'e.g. \'{"temperature": 0.01, "ignore_eos": false}\'')

