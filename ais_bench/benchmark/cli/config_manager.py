
import os
import os.path as osp
import tabulate
from mmengine.config import Config

from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import TMAN_CODES
from ais_bench.benchmark.datasets.custom import make_custom_dataset_config
from ais_bench.benchmark.utils.file import match_cfg_file
from ais_bench.benchmark.utils.config.run import try_fill_in_custom_cfgs
from ais_bench.benchmark.utils.logging.exceptions import CommandError, AISBenchConfigError
from ais_bench.benchmark.cli.utils import fill_model_path_if_datasets_need, fill_test_range_use_num_prompts

class CustomConfigChecker:
    MODEL_REQUIRED_FIELDS = ['type', 'abbr', 'attr']
    DATASET_REQUIRED_FIELDS = ['type', 'abbr', 'reader_cfg', 'infer_cfg', 'eval_cfg']
    SUMMARIZER_REQUIRED_FIELDS = ['attr']

    def __init__(self, config, file_path):
        self.config = config
        self.file_path = file_path

    def check(self):
        self._check_models_config()
        self._check_datasets_config()
        self._check_summarizer_config()

    def _check_models_config(self):
        models = self.config.get('models', [])
        if not models:
            raise AISBenchConfigError(TMAN_CODES.CFG_CONTENT_MISS_REQUIRED_PARAM, f"Config file {self.file_path} does not contain 'models' param!")
        if not isinstance(models, list):
            raise AISBenchConfigError(TMAN_CODES.TYPE_ERROR_IN_CFG_PARAM, f"In config file {self.file_path}, 'models' param must be a list!")
        for model in models:
            if not isinstance(model, dict):
                raise AISBenchConfigError(TMAN_CODES.TYPE_ERROR_IN_CFG_PARAM, f"In config file {self.file_path}, " +
                                 "member of 'models' param must be a dict!")
            for param in self.MODEL_REQUIRED_FIELDS:
                if param not in model:
                    raise AISBenchConfigError(TMAN_CODES.CFG_CONTENT_MISS_REQUIRED_PARAM, f"In config file {self.file_path}, " +
                                     f"member of 'models' param must contain '{param}' param!")

    def _check_datasets_config(self):
        datasets = self.config.get('datasets', [])
        if not datasets:
            raise AISBenchConfigError(TMAN_CODES.CFG_CONTENT_MISS_REQUIRED_PARAM, f"Config file {self.file_path} does not contain 'datasets' param!")
        if not isinstance(datasets, list):
            raise AISBenchConfigError(TMAN_CODES.TYPE_ERROR_IN_CFG_PARAM, f"In config file {self.file_path}, 'datasets' param must be a list!")
        for dataset in datasets:
            if not isinstance(dataset, dict):
                raise AISBenchConfigError(TMAN_CODES.TYPE_ERROR_IN_CFG_PARAM, f"In config file {self.file_path}, " +
                                 "member of 'datasets' param must be a dict!")
            for param in self.DATASET_REQUIRED_FIELDS:
                if param not in dataset:
                    raise AISBenchConfigError(TMAN_CODES.CFG_CONTENT_MISS_REQUIRED_PARAM, f"In config file {self.file_path}, " +
                                     f"member of 'datasets' param must contain '{param}' param!")

    def _check_summarizer_config(self):
        summarizer = self.config.get('summarizer', None)
        if not summarizer:
            raise AISBenchConfigError(TMAN_CODES.CFG_CONTENT_MISS_REQUIRED_PARAM, f"Config file {self.file_path} does not contain 'summarizer' param!")
        if not isinstance(summarizer, dict):
            raise AISBenchConfigError(TMAN_CODES.TYPE_ERROR_IN_CFG_PARAM, f"In config file {self.file_path}, " +
                             "'summarizer' param must be a dict!")
        for param in self.SUMMARIZER_REQUIRED_FIELDS:
            if param not in summarizer:
                raise AISBenchConfigError(TMAN_CODES.CFG_CONTENT_MISS_REQUIRED_PARAM, f"In config file {self.file_path}, " +
                                 f"member of 'summarizer' param must contain '{param}' param!")

class ConfigManager:
    def __init__(self, args):
        self.args = args
        self.logger = AISLogger()

    def search_configs_location(self):
        """Get the config object given args.
        """
        self.logger.info('Searching configs...')
        self.table = [["Task Type", "Task Name", "Config File Path"]]
        if self.args.models:
           self._search_models_config()

        if self.args.datasets:
            self._search_datasets_config()

        if self.args.summarizer:
            self._search_summarizers_config()

        print( # origin print
            tabulate.tabulate(
                self.table,
                headers='firstrow',
                tablefmt="fancy_grid",
                stralign="left",
                missingval="N/A",
            )
        )

    def load_config(self, workflow):
        self.cfg = self._get_config_from_arg()
        self._update_and_init_work_dir()
        self._fill_dataset_configs()
        self._update_cfg_of_workflow(workflow)
        self._dump_and_reload_config()
        return self.cfg
    
    def _fill_dataset_configs(self):
        for dataset_cfg in self.cfg["datasets"]:
            fill_test_range_use_num_prompts(self.cfg["cli_args"].get("num_prompts"), dataset_cfg)
            fill_model_path_if_datasets_need(self.cfg["models"][0], dataset_cfg)
            retriever_cfg = dataset_cfg["infer_cfg"]["retriever"]
            infer_cfg = dataset_cfg["infer_cfg"]
            if "prompt_template" in infer_cfg:
                retriever_cfg["prompt_template"] = infer_cfg["prompt_template"]
            if "ice_template" in infer_cfg:
                retriever_cfg["ice_template"] = infer_cfg["ice_template"]

    def _search_models_config(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        default_configs_dir = os.path.join(parent_dir, 'configs')
        models_dir = [
            os.path.join(self.args.config_dir, 'models'),
            os.path.join(default_configs_dir, './models'),
        ]
        for model_arg in self.args.models:
            for model in match_cfg_file(models_dir, [model_arg]):
                self.table.append(["--models", model[0], os.path.abspath(model[1])])

    def _search_datasets_config(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        default_configs_dir = os.path.join(parent_dir, 'configs')
        datasets_dir = [
            os.path.join(self.args.config_dir, 'datasets'),
            os.path.join(self.args.config_dir, 'dataset_collections'),
            os.path.join(default_configs_dir, './datasets'),
            os.path.join(default_configs_dir, './dataset_collections')
        ]
        for dataset_arg in self.args.datasets:
            if '/' in dataset_arg:
                dataset_name, _dataset_suffix = dataset_arg.split('/', 1)
            else:
                dataset_name = dataset_arg

            for dataset in match_cfg_file(datasets_dir, [dataset_name]):
                self.table.append(["--datasets", dataset[0], os.path.abspath(dataset[1])])

    def _search_summarizers_config(self):
        summarizer_arg = self.args.summarizer if self.args.summarizer is not None else 'example'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        default_configs_dir = os.path.join(parent_dir, 'configs')
        summarizers_dir = [
            os.path.join(self.args.config_dir, 'summarizers'),
            os.path.join(default_configs_dir, './summarizers'),
        ]

        # Check if summarizer_arg contains '/'
        if '/' in summarizer_arg:
            # If it contains '/', split the string by '/'
            # and use the second part as the configuration key
            summarizer_file, summarizer_key = summarizer_arg.split('/', 1)
        else:
            # If it does not contain '/', keep the original logic unchanged
            summarizer_file = summarizer_arg

        s = match_cfg_file(summarizers_dir, [summarizer_file])[0]
        self.table.append(["--summarizer", s[0], os.path.abspath(s[1])])

    def _get_config_from_arg(self):
        if self.args.config:
            try:
                config = Config.fromfile(self.args.config, format_python_code=False)
            except BaseException as e:
                raise AISBenchConfigError(TMAN_CODES.INVAILD_SYNTAX_IN_CFG_CONTENT, f'Config file {self.args.config} contain invaild syntax: {e}')
            config = try_fill_in_custom_cfgs(config)
            CustomConfigChecker(config, self.args.config).check()
            config.merge_from_dict(dict(cli_args = vars(self.args)))
            return config

        models = self._load_models_config()
        datasets = self._load_datasets_config()
        summarizer = self._load_summarizers_config()

        return Config(dict(models=models, datasets=datasets, summarizer=summarizer, cli_args=vars(self.args)), format_python_code=False)

    def _load_datasets_config(self):
        datasets = []
        if self.args.datasets:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            default_configs_dir = os.path.join(parent_dir, 'configs')
            datasets_dir = [
                os.path.join(self.args.config_dir, 'datasets'),
                os.path.join(self.args.config_dir, 'dataset_collections'),
                os.path.join(default_configs_dir, './datasets'),
                os.path.join(default_configs_dir, './dataset_collections')
            ]
            for dataset_arg in self.args.datasets:
                if '/' in dataset_arg:
                    dataset_name, dataset_suffix = dataset_arg.split('/', 1)
                    dataset_key_suffix = dataset_suffix
                else:
                    dataset_name = dataset_arg
                    dataset_key_suffix = '_datasets'

                for dataset in match_cfg_file(datasets_dir, [dataset_name]):
                    self.logger.info(f'Loading {dataset[0]}: {dataset[1]}')
                    try:
                        cfg = Config.fromfile(dataset[1])
                    except BaseException as e:
                        raise AISBenchConfigError(TMAN_CODES.INVAILD_SYNTAX_IN_CFG_CONTENT, f'Config file {dataset[1]} contain invaild syntax: {e}')
                    dataset_cfg_exist = False
                    for k in cfg.keys():
                        if k.endswith(dataset_key_suffix):
                            datasets += cfg[k]
                            dataset_cfg_exist = True
                    if not dataset_cfg_exist:
                        raise AISBenchConfigError(TMAN_CODES.CFG_CONTENT_MISS_REQUIRED_PARAM, f"Config file {dataset[1]} does not contain a param end with {dataset_key_suffix}!")
        else:
            if self.args.custom_dataset_path is None:
                raise CommandError(TMAN_CODES.CMD_MISS_REQUIRED_ARG, 'You must specify a custom dataset path, or specify --datasets.')
            dataset = {'path': self.args.custom_dataset_path}
            if self.args.custom_dataset_infer_method is not None:
                dataset['infer_method'] = self.args.custom_dataset_infer_method
            if self.args.custom_dataset_data_type is not None:
                dataset['data_type'] = self.args.custom_dataset_data_type
            if self.args.custom_dataset_meta_path is not None:
                dataset['meta_path'] = self.args.custom_dataset_meta_path
            dataset = make_custom_dataset_config(dataset)
            datasets.append(dataset)
        return datasets

    def _load_models_config(self):
        if not self.args.models:
            raise CommandError(TMAN_CODES.CMD_MISS_REQUIRED_ARG, 'You must specify a config file path, or specify --models and --datasets.')
        models = []
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        default_configs_dir = os.path.join(parent_dir, 'configs')
        models_dir = [
            os.path.join(self.args.config_dir, 'models'),
            os.path.join(default_configs_dir, './models'),

        ]
        if self.args.models:
            for model_arg in self.args.models:
                for model in match_cfg_file(models_dir, [model_arg]):
                    self.logger.info(f'Loading {model[0]}: {model[1]}')
                    try:
                        cfg = Config.fromfile(model[1])
                    except BaseException as e:
                        raise AISBenchConfigError(TMAN_CODES.INVAILD_SYNTAX_IN_CFG_CONTENT, f'Config file {model[1]} contain invaild syntax: {e}')
                    if 'models' not in cfg:
                        raise AISBenchConfigError(TMAN_CODES.CFG_CONTENT_MISS_REQUIRED_PARAM, f"Config file {model[1]} does not contain 'models' param")
                    models += cfg['models']
        return models

    def _load_summarizers_config(self):
        # parse summarizer args
        summarizer_arg = self.args.summarizer if self.args.summarizer is not None else 'example'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        default_configs_dir = os.path.join(parent_dir, 'configs')
        summarizers_dir = [
            os.path.join(self.args.config_dir, 'summarizers'),
            os.path.join(default_configs_dir, './summarizers'),

        ]

        # Check if summarizer_arg contains '/'
        if '/' in summarizer_arg:
            # If it contains '/', split the string by '/'
            # and use the second part as the configuration key
            summarizer_file, summarizer_key = summarizer_arg.split('/', 1)
        else:
            # If it does not contain '/', keep the original logic unchanged
            summarizer_key = 'summarizer'
            summarizer_file = summarizer_arg

        s = match_cfg_file(summarizers_dir, [summarizer_file])[0]
        self.logger.info(f'Loading {s[0]}: {s[1]}')
        try:
            cfg = Config.fromfile(s[1])
        except BaseException as e:
            raise AISBenchConfigError(TMAN_CODES.INVAILD_SYNTAX_IN_CFG_CONTENT, f'Config file {s[1]} contain invaild syntax: {e}')
        # Use summarizer_key to retrieve the summarizer definition
        # from the configuration file
        summarizer = cfg[summarizer_key]
        return summarizer

    def _update_and_init_work_dir(self):
        if self.args.work_dir is not None:
            self.cfg['work_dir'] = self.args.work_dir
        else:
            self.cfg.setdefault('work_dir', os.path.join('outputs', 'default'))

        # cfg_time_str defaults to the current time
        self.cfg_time_str = dir_time_str = self.args.dir_time_str

        if self.args.reuse:
            if self.args.reuse == 'latest':
                if not os.path.exists(self.cfg.work_dir) or not os.listdir(
                        self.cfg.work_dir):
                    self.logger.warning('No previous experiment results found to reuse.')
                else:
                    dirs = os.listdir(self.cfg.work_dir)
                    dir_time_str = sorted(dirs)[-1]
            else:
                dir_time_str = self.args.reuse
            self.args.dir_time_str = dir_time_str
            self.logger.info(f'Reusing experiements from {dir_time_str}')

        # update "actual" work_dir
        self.cfg['work_dir'] = osp.join(self.cfg.work_dir, dir_time_str)
        current_workdir = self.cfg['work_dir']
        self.logger.info(f'Current exp folder: {current_workdir}')

        os.makedirs(osp.join(self.cfg.work_dir, 'configs'), exist_ok=True)

    def _update_cfg_of_workflow(self, workflow):
        for work in workflow:
            self.cfg = work.update_cfg(self.cfg)

    def _dump_and_reload_config(self):
        # dump config
        output_config_path = osp.join(self.cfg.work_dir, 'configs',
                                    f'{self.cfg_time_str}_{os.getpid()}.py')
        self.cfg.dump(output_config_path)
        # eval nums set
        if (self.args.num_prompts and self.args.num_prompts < 0) or self.args.num_prompts == 0:
            raise CommandError(TMAN_CODES.INVALID_ARG_VALUE_IN_CMD, "'--num-prompts' must be a positive integer greater than 0.")
        self.cfg['num_prompts'] = self.args.num_prompts
        # Config is intentally reloaded here to avoid initialized
        # types cannot be serialized
        try:
            self.cfg = Config.fromfile(output_config_path, format_python_code=False)
        except BaseException as e:
            raise AISBenchConfigError(TMAN_CODES.INVAILD_SYNTAX_IN_CFG_CONTENT, f'Config file {output_config_path} contain invaild syntax: {e}')