import os
import os.path as osp
import inspect
import tabulate
from mmengine.config import Config

from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import TMAN_CODES
from ais_bench.benchmark.datasets.custom import make_custom_dataset_config
from ais_bench.benchmark.utils.file import match_cfg_file
from ais_bench.benchmark.utils.config.run import try_fill_in_custom_cfgs
from ais_bench.benchmark.utils.logging.exceptions import CommandError, AISBenchConfigError
from ais_bench.benchmark.cli.utils import fill_model_path_if_datasets_need, fill_test_range_use_num_prompts, recur_convert_config_type
from ais_bench.benchmark.utils.response_anomaly import ResponseAnomalyCoordinator

RESPONSE_ANOMALY_TOP_LOGPROBS = 20

# Backends allowed to enable response anomaly detection. New backends that can
# return token ids + top-k logprobs should subclass VLLMCustomAPIChat (then
# this check passes automatically) or be added to the name tuple below when
# configured by class-name string.
RESPONSE_ANOMALY_SUPPORTED_MODEL_NAMES = (
    'VLLMCustomAPIChat',
    'VLLMCustomAPIChatStream',
    'VllmMultiturnAPIChatStream',
)


class CustomConfigChecker:
    MODEL_REQUIRED_FIELDS = ['abbr']
    DATASET_REQUIRED_FIELDS = ['abbr']
    SUMMARIZER_REQUIRED_FIELDS = ['attr']

    def __init__(self, config, file_path):
        self.config = config
        self.file_path = file_path

    def check(self):
        self._check_models_config()
        self._check_datasets_config()

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
        self._init_response_anomaly_config()
        self._update_and_init_work_dir()
        self._inject_response_anomaly_payload_storage()
        self._fill_dataset_configs()
        self._update_cfg_of_workflow(workflow)
        self._dump_and_reload_config()
        return self.cfg

    def _init_response_anomaly_config(self):
        """Normalize the optional response anomaly detection configuration."""
        global_cfg, configured_top_logprobs = (
            self._normalize_response_anomaly_global_config()
        )
        self.cfg['response_anomaly'] = global_cfg
        if not global_cfg['enabled']:
            return

        self._validate_response_anomaly_top_logprobs(configured_top_logprobs)
        self._validate_response_anomaly_support()
        self._validate_response_anomaly_payload_config(global_cfg)

        service_models = self._get_response_anomaly_service_models()
        if service_models is None:
            return
        self._warn_shared_response_anomaly_model_name(global_cfg, service_models)
        for model_cfg in service_models:
            self._init_response_anomaly_model(model_cfg, global_cfg)

    def _inject_response_anomaly_payload_storage(self):
        """Add runtime payload storage settings to supported model configs."""
        anomaly_cfg = self.cfg.get('response_anomaly') or {}
        if not anomaly_cfg.get('enabled', False):
            return
        models = self.cfg.get('models')
        if not isinstance(models, list):
            return
        storage_cfg = dict(anomaly_cfg.get('payload_storage') or {})
        for model_cfg in models:
            if model_cfg.get('attr', 'service') != 'service':
                continue
            model_cfg['response_anomaly_payload_storage'] = {
                'work_dir': self.cfg['work_dir'],
                'model_abbr': model_cfg['abbr'],
                **storage_cfg,
            }

    def _normalize_response_anomaly_global_config(self):
        """Apply CLI overrides and defaults to the global anomaly config."""
        raw_anomaly_cfg = self.cfg.get('response_anomaly') or {}
        global_cfg = dict(raw_anomaly_cfg) if isinstance(raw_anomaly_cfg, dict) else {}
        # The enabled switch is command-line only (--response-anomaly); an
        # 'enabled' key in the config file is not a supported enable path.
        # Warn and drop it so it can never silently enable detection.
        if 'enabled' in global_cfg:
            self.logger.warning(
                "response_anomaly.enabled in the config file is not "
                "supported; use the --response-anomaly command-line switch "
                "to enable response anomaly detection. The configured "
                "value is ignored."
            )
            global_cfg.pop('enabled')
        # Strictly a real boolean from the CLI; anything else (missing
        # attribute, mocks) means the switch was not passed -> disabled.
        cli_enabled = getattr(self.args, 'response_anomaly', False)
        global_cfg['enabled'] = cli_enabled if isinstance(cli_enabled, bool) else False
        configured_top_logprobs = global_cfg.pop('top_logprobs', None)
        global_cfg.setdefault('msprobe_config_path', None)
        cli_payload_retention = getattr(
            self.args, 'response_anomaly_payload_retention', None
        )
        if isinstance(cli_payload_retention, str):
            global_cfg['payload_retention'] = cli_payload_retention
        global_cfg.setdefault('payload_retention', 'anomalies')
        payload_storage = dict(global_cfg.get('payload_storage') or {})
        payload_storage.setdefault('format', 'jsonl')
        payload_storage.setdefault('compression', 'zstd')
        payload_storage.setdefault('compression_level', 3)
        payload_storage.setdefault('rows_per_shard', 2000)
        global_cfg['payload_storage'] = payload_storage
        return global_cfg, configured_top_logprobs

    @staticmethod
    def _validate_response_anomaly_top_logprobs(configured_top_logprobs):
        """Reject attempts to override the detector's fixed top-k value."""
        if (
            configured_top_logprobs is not None
            and (
                not isinstance(configured_top_logprobs, int)
                or isinstance(configured_top_logprobs, bool)
                or configured_top_logprobs != RESPONSE_ANOMALY_TOP_LOGPROBS
            )
        ):
            raise AISBenchConfigError(
                TMAN_CODES.UNKNOWN_ERROR,
                "response_anomaly.top_logprobs is fixed at "
                f"{RESPONSE_ANOMALY_TOP_LOGPROBS} and cannot be configured.",
            )

    @staticmethod
    def _validate_response_anomaly_payload_config(global_cfg):
        """Validate payload retention and compressed storage settings."""
        if global_cfg['payload_retention'] not in ('all', 'anomalies', 'none'):
            raise AISBenchConfigError(
                TMAN_CODES.UNKNOWN_ERROR,
                "response_anomaly.payload_retention must be one of "
                "'all', 'anomalies' or 'none'.",
            )
        payload_storage = global_cfg['payload_storage']
        if payload_storage['format'] != 'jsonl':
            raise AISBenchConfigError(
                TMAN_CODES.UNKNOWN_ERROR,
                "response_anomaly.payload_storage.format must be 'jsonl'.",
            )
        if payload_storage['compression'] != 'zstd':
            raise AISBenchConfigError(
                TMAN_CODES.UNKNOWN_ERROR,
                "response_anomaly.payload_storage.compression must be 'zstd'.",
            )
        compression_level = payload_storage['compression_level']
        rows_per_shard = payload_storage['rows_per_shard']
        if (
            not isinstance(compression_level, int)
            or isinstance(compression_level, bool)
            or not 1 <= compression_level <= 22
        ):
            raise AISBenchConfigError(
                TMAN_CODES.UNKNOWN_ERROR,
                "response_anomaly.payload_storage.compression_level must be "
                "an integer between 1 and 22.",
            )
        if (
            not isinstance(rows_per_shard, int)
            or isinstance(rows_per_shard, bool)
            or rows_per_shard <= 0
        ):
            raise AISBenchConfigError(
                TMAN_CODES.UNKNOWN_ERROR,
                "response_anomaly.payload_storage.rows_per_shard must be a "
                "positive integer.",
            )

    def _get_response_anomaly_service_models(self):
        """Return configured service models, preserving absent-model behavior."""
        models = self.cfg.get('models')
        if not isinstance(models, list):
            return None
        service_models = [
            model_cfg
            for model_cfg in models
            if model_cfg.get('attr', 'service') == 'service'
        ]
        if not service_models:
            raise AISBenchConfigError(
                TMAN_CODES.UNKNOWN_ERROR,
                "response_anomaly is enabled but no service model is configured. "
                "Response anomaly detection requires service models (attr='service').",
            )
        return service_models

    def _warn_shared_response_anomaly_model_name(
        self, global_cfg, service_models
    ):
        """Warn when one global model name may be applied to multiple models."""
        if (
            len(service_models) > 1
            and global_cfg.get('model_name')
            and any(
                'model_name' not in (model_cfg.get('response_anomaly') or {})
                for model_cfg in service_models
            )
        ):
            self.logger.warning(
                "response_anomaly.model_name is configured globally while multiple "
                "service models are present; prefer setting model_name inside each "
                "model's response_anomaly config."
            )

    def _init_response_anomaly_model(self, model_cfg, global_cfg):
        """Build and validate one model's anomaly detection configuration."""
        model_anomaly_cfg = self._merge_response_anomaly_model_config(
            model_cfg, global_cfg
        )
        self._resolve_response_anomaly_model_path(model_cfg, model_anomaly_cfg)
        self._validate_response_anomaly_model_resources(
            model_cfg, model_anomaly_cfg
        )
        self._validate_response_anomaly_model_name(model_cfg, model_anomaly_cfg)
        self._validate_response_anomaly_resource_paths(
            model_cfg, model_anomaly_cfg
        )
        model_cfg['response_anomaly'] = model_anomaly_cfg
        self._inject_response_anomaly_request_config(model_cfg)

    def _merge_response_anomaly_model_config(self, model_cfg, global_cfg):
        """Merge global model resources with model-level overrides."""
        model_anomaly_cfg = dict(model_cfg.get('response_anomaly') or {})
        configured_top_logprobs = model_anomaly_cfg.pop('top_logprobs', None)
        self._validate_response_anomaly_top_logprobs(configured_top_logprobs)
        # NOTE: model_name intentionally has no model-abbr fallback. The abbr
        # is a task label (e.g. 'vllm-api-general-chat') unrelated to the
        # served model, so it would silently miss the keys in msProbe's
        # mtype_config.json and token2category. When model_name is not set,
        # derive it from the most specific source available, in order (each
        # path resolved to its directory basename, the same default the
        # config generator uses): explicit model_name is handled above by the
        # ``not model_anomaly_cfg.get('model_name')`` guard; then model-level
        # model_path, global model_name, global model_path, and finally the
        # model 'path' field.
        if not model_anomaly_cfg.get('model_name'):
            fallback = (
                self._model_name_from_path(model_anomaly_cfg.get('model_path'))
                or global_cfg.get('model_name')
                or self._model_name_from_path(global_cfg.get('model_path'))
                or self._model_name_from_path(model_cfg.get('path'))
            )
            if fallback:
                model_anomaly_cfg['model_name'] = fallback
        for key in (
            'model_path',
            'msprobe_config_path',
            'msprobe_mtype_path',
            'msprobe_token2category_dir',
        ):
            if key not in model_anomaly_cfg:
                model_anomaly_cfg[key] = global_cfg.get(key)
        return model_anomaly_cfg

    @staticmethod
    def _model_name_from_path(model_path):
        """Derive the de-facto model name from the model directory basename."""
        if not model_path:
            return None
        basename = osp.basename(osp.normpath(str(model_path).strip()))
        return basename or None

    @staticmethod
    def _validate_response_anomaly_model_name(model_cfg, model_anomaly_cfg):
        """Fail fast when no reliable msProbe model name can be determined."""
        if model_anomaly_cfg.get('model_name'):
            return
        has_explicit_resources = bool(
            model_anomaly_cfg.get('msprobe_mtype_path')
            and model_anomaly_cfg.get('msprobe_token2category_dir')
        )
        raise AISBenchConfigError(
            TMAN_CODES.UNKNOWN_ERROR,
            f"response_anomaly is enabled for model "
            f"'{model_cfg.get('abbr', '')}' but response_anomaly.model_name "
            "is not set and cannot be inferred from a model directory. "
            + (
                "Since explicit msProbe resource paths are configured, set "
                "response_anomaly.model_name to the key used in "
                "msprobe_mtype_path and in the token2category file names."
                if has_explicit_resources
                else "Set response_anomaly.model_name explicitly (it must "
                "match the keys in msProbe's mtype_config.json and the "
                "token2category file names), or set model "
                "'path'/model_path to the local model directory so the "
                "name can be derived from its basename."
            ),
        )

    @staticmethod
    def _resolve_response_anomaly_model_path(model_cfg, model_anomaly_cfg):
        """Use the model tokenizer path when no explicit model path is set."""
        if model_anomaly_cfg.get('model_path'):
            return
        model_path = str(model_cfg.get('path') or '').strip()
        if not model_path:
            return
        if not osp.isdir(model_path):
            raise AISBenchConfigError(
                TMAN_CODES.UNKNOWN_ERROR,
                f"response_anomaly is enabled for model "
                f"'{model_cfg.get('abbr', '')}' but its 'path' field "
                f"points to a non-existent directory: {model_path}. "
                "Fix the model 'path' or configure "
                "response_anomaly.model_path / msprobe paths.",
            )
        model_anomaly_cfg['model_path'] = model_path

    @staticmethod
    def _validate_response_anomaly_model_resources(model_cfg, model_anomaly_cfg):
        """Require either a tokenizer path or explicit model resources."""
        if model_anomaly_cfg.get('model_path') or (
            model_anomaly_cfg.get('msprobe_mtype_path')
            and model_anomaly_cfg.get('msprobe_token2category_dir')
        ):
            return
        missing = [
            "response_anomaly.model_path is not set and the model "
            "'path' field (tokenizer directory) is empty"
        ]
        if not model_anomaly_cfg.get('msprobe_mtype_path'):
            missing.append("response_anomaly.msprobe_mtype_path is not set")
        if not model_anomaly_cfg.get('msprobe_token2category_dir'):
            missing.append(
                "response_anomaly.msprobe_token2category_dir is not set"
            )
        raise AISBenchConfigError(
            TMAN_CODES.UNKNOWN_ERROR,
            f"response_anomaly is enabled for model "
            f"'{model_cfg.get('abbr', '')}' but no msProbe model "
            "resources are available: the token2category vocabulary is "
            "model-specific and cannot fall back to msProbe built-in "
            "defaults. Missing:\n"
            + "\n".join(f"  - {item}" for item in missing)
            + "\nProvide one of the following:\n"
            "  1) set the model 'path' to the local tokenizer directory "
            "so the msProbe config files and token2category vocabulary "
            "are auto-generated;\n"
            "  2) set response_anomaly.model_path to the local "
            "model/tokenizer directory;\n"
            "  3) generate them manually with "
            "`ais_bench-gen-response-anomaly-config --model-path <dir>` "
            "and set msprobe_mtype_path together with "
            "msprobe_token2category_dir.",
        )

    @staticmethod
    def _validate_response_anomaly_resource_paths(model_cfg, model_anomaly_cfg):
        """Validate paths that will not be generated from a local model."""
        invalid_paths = []
        model_path = model_anomaly_cfg.get('model_path')
        if model_path and not osp.isdir(model_path):
            invalid_paths.append(f"model_path={model_path} (directory not found)")
        if not model_path:
            path_checks = (
                ('msprobe_config_path', osp.isfile, 'file'),
                ('msprobe_mtype_path', osp.isfile, 'file'),
                ('msprobe_token2category_dir', osp.isdir, 'directory'),
            )
            for key, exists, path_type in path_checks:
                path = model_anomaly_cfg.get(key)
                if path and not exists(path):
                    invalid_paths.append(f"{key}={path} ({path_type} not found)")
        if not invalid_paths:
            return
        raise AISBenchConfigError(
            TMAN_CODES.UNKNOWN_ERROR,
            f"response_anomaly is enabled for model "
            f"'{model_cfg.get('abbr', '')}' but some configured msProbe "
            "resources do not exist:\n"
            + "\n".join(f"  - {item}" for item in invalid_paths)
            + "\nCheck that the paths are mounted/copied to this machine, "
            "or re-generate them with "
            "`ais_bench-gen-response-anomaly-config --model-path <dir>`.",
        )

    @staticmethod
    def _inject_response_anomaly_request_config(model_cfg):
        """Force the service response fields required by anomaly detection."""
        generation_kwargs = model_cfg.setdefault('generation_kwargs', {})
        if not isinstance(generation_kwargs, dict):
            raise AISBenchConfigError(
                TMAN_CODES.UNKNOWN_ERROR,
                "response_anomaly is enabled but "
                f"model '{model_cfg.get('abbr', '')}' has invalid "
                "generation_kwargs; expected a dict.",
            )
        generation_kwargs['logprobs'] = True
        generation_kwargs['top_logprobs'] = RESPONSE_ANOMALY_TOP_LOGPROBS
        generation_kwargs['response_anomaly_enabled'] = True

    @staticmethod
    def _is_supported_response_anomaly_model(model_cfg: dict) -> bool:
        """Only chat backends returning token ids + top-k logprobs are allowed."""
        model_type = model_cfg.get('type')
        if isinstance(model_type, str):
            return model_type in RESPONSE_ANOMALY_SUPPORTED_MODEL_NAMES
        if not isinstance(model_type, type):
            return False
        from ais_bench.benchmark.models.api_models.vllm_custom_api_chat import (
            VLLMCustomAPIChat,
        )
        return issubclass(model_type, VLLMCustomAPIChat)

    def _validate_response_anomaly_support(self):
        """Reject modes/links that are intentionally unsupported."""
        self._validate_response_anomaly_mode()
        self._validate_response_anomaly_infer_task()
        self._validate_response_anomaly_models()
        self._validate_response_anomaly_datasets()

    def _validate_response_anomaly_mode(self):
        """Allow response anomaly detection only in inference workflows."""
        mode = getattr(self.args, 'mode', 'all')
        if isinstance(mode, str) and mode not in ('all', 'infer', 'infer_judge'):
            raise AISBenchConfigError(
                TMAN_CODES.UNKNOWN_ERROR,
                f"response anomaly detection is not supported in mode "
                f"'{mode}'; supported modes are 'all', 'infer' and "
                "'infer_judge'.",
            )

    def _validate_response_anomaly_infer_task(self):
        """Reject custom inference tasks that bypass the supported pipeline."""
        infer_cfg = self.cfg.get('infer')
        if not isinstance(infer_cfg, dict):
            return
        task_type = (infer_cfg.get('runner') or {}).get('task', {}).get('type')
        task_name = self._cfg_type_name(task_type)
        if task_name and task_name not in ('OpenICLInferTask', 'OpenICLApiInferTask'):
            raise AISBenchConfigError(
                TMAN_CODES.UNKNOWN_ERROR,
                f"response anomaly detection is not supported for infer task "
                f"'{task_name}' (Agent/custom tasks are not supported).",
            )

    def _validate_response_anomaly_models(self):
        """Validate every configured model against the supported backends."""
        models = self.cfg.get('models')
        if not isinstance(models, list):
            return
        for model_cfg in models:
            if isinstance(model_cfg, dict):
                self._validate_response_anomaly_model_support(model_cfg)

    def _validate_response_anomaly_model_support(self, model_cfg):
        """Reject Agent and service backends without anomaly payload support."""
        agent_keys = ('agent', 'agent_name', 'llm_agent', 'llm_user')
        if any(key in model_cfg for key in agent_keys):
            raise AISBenchConfigError(
                TMAN_CODES.UNKNOWN_ERROR,
                f"response anomaly detection is not supported for Agent "
                f"models (model abbr='{model_cfg.get('abbr', '')}').",
            )
        if model_cfg.get('attr', 'service') != 'service':
            return
        if self._is_supported_response_anomaly_model(model_cfg):
            return
        model_type = model_cfg.get('type')
        raise AISBenchConfigError(
            TMAN_CODES.UNKNOWN_ERROR,
            f"response anomaly detection is not supported for model "
            f"type '{getattr(model_type, '__name__', model_type)}' "
            f"(model abbr='{model_cfg.get('abbr', '')}'): only "
            "VLLMCustomAPIChat backends return the required token "
            "ids and top-k logprobs. Use the vllm_api_general_chat / "
            "vllm_api_stream_chat / vllm_api_stream_chat_multiturn "
            "model configs instead.",
        )

    def _validate_response_anomaly_datasets(self):
        """Validate every dataset against unsupported Agent-style links."""
        datasets = self.cfg.get('datasets')
        if not isinstance(datasets, list):
            return
        for dataset_cfg in datasets:
            if isinstance(dataset_cfg, dict):
                self._validate_response_anomaly_dataset_support(dataset_cfg)

    def _validate_response_anomaly_dataset_support(self, dataset_cfg):
        """Reject datasets whose inferencer uses an Agent-style protocol."""
        infer_cfg = dataset_cfg.get('infer_cfg') or {}
        inferencer = infer_cfg.get('inferencer') or {}
        inferencer_name = self._cfg_type_name(inferencer.get('type'))
        dataset_name = self._cfg_type_name(dataset_cfg.get('type'))
        haystack = f"{inferencer_name} {dataset_name}".lower()
        unsupported_markers = (
            'swebench',
            'bfcl',
            'agent',
            'function_call',
            'tool_call',
            'harbor',
            'tau2',
        )
        if any(marker in haystack for marker in unsupported_markers):
            raise AISBenchConfigError(
                TMAN_CODES.UNKNOWN_ERROR,
                f"response anomaly detection is not supported for Agent/custom "
                f"evaluation (inferencer='{inferencer_name}', "
                f"dataset='{dataset_name}').",
            )

    @staticmethod
    def _cfg_type_name(value) -> str:
        """Return the short class name of a config type value."""
        if value is None:
            return ''
        if isinstance(value, type):
            return value.__name__
        return str(value).rsplit('.', 1)[-1]

    def _fill_dataset_configs(self):
        for dataset_cfg in self.cfg["datasets"]:
            if dataset_cfg.get("infer_cfg", None) is None:
                continue
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
            if config.get('models'):
                self._apply_cli_api_model_overrides(config['models'])
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
        self._apply_cli_api_model_overrides(models)
        return models

    def _resolve_model_field_name(self, model_cfg):
        """Return the model-name key ('model'/'model_name') accepted by the model
        class, or None if the type accepts neither.

        Which field name a config can carry depends on its `type` (the model
        class), not on the config file itself.
        """
        try:
            params = inspect.signature(model_cfg["type"].__init__).parameters
        except (TypeError, ValueError):
            return None
        for key in ("model", "model_name"):
            if key in params:
                return key
        return None

    def _apply_cli_api_model_overrides(self, models):
        """Override each model config with API model args explicitly given on the CLI.

        Only fields already present in a model config are overwritten (no new
        keys are injected), so that classes not supporting a given param (e.g.
        MindieStreamApi) are not passed unexpected keywords. The model-name field
        is resolved by the model `type` signature.
        """
        fields = ["path", "request_rate", "retry", "api_key", "host_ip",
                  "host_port", "url", "max_out_len", "batch_size",
                  "trust_remote_code", "generation_kwargs"]
        for model_cfg in models:
            # 1) model/model_name depends on the type: overwrite if accepted,
            #    otherwise warn and skip.
            model_val = getattr(self.args, "model_name", None)
            if model_val is not None:
                target_key = self._resolve_model_field_name(model_cfg)
                if target_key is not None:
                    model_cfg[target_key] = model_val
                else:
                    type_name = getattr(model_cfg.get("type"), "__name__", model_cfg.get("type"))
                    self.logger.warning(
                        'CLI --model-name=%s is ignored: model type %s accepts '
                        'neither model nor model_name',
                        model_val, type_name,
                    )
            # 2) Other common fields: only overwrite existing keys.
            for field in fields:
                cli_val = getattr(self.args, field, None)
                if cli_val is None or field not in model_cfg:
                    continue
                model_cfg[field] = cli_val

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
        # Remove a response anomaly status left by a previous interrupted run so
        # stale state never blocks or misleads a new run's task board.
        stale_anomaly_status = osp.join(
            self.cfg.work_dir,
            'status_tmp',
            ResponseAnomalyCoordinator.STATUS_FILE_NAME,
        )
        try:
            if os.path.isfile(stale_anomaly_status):
                os.remove(stale_anomaly_status)
        except OSError:
            # Best-effort cleanup; a concurrent process may have removed it.
            pass

    def _update_cfg_of_workflow(self, workflow):
        for work in workflow:
            self.cfg = work.update_cfg(self.cfg)

    def _dump_and_reload_config(self):
        # dump config
        output_config_path = osp.join(self.cfg.work_dir, 'configs',
                                    f'{self.cfg_time_str}_{os.getpid()}.py')

        recur_convert_config_type(self.cfg)
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
