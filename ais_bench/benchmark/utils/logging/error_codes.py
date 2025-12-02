from enum import Enum, unique
from typing import Dict, Optional

@unique
class ErrorModule(Enum):
    TASK_MANAGER = "TMAN"                        # TaskManager
    PARTITIONER = "PARTI"                        # Partitioner
    SUMMARY = "SUMM"                             # Summary
    RUNNER = "RUNNER"                            # Runner
    TASK = "TASK"                                # Task
    TASK_INFER = "TINFER"                        # inference Task
    TASK_EVALUATE = "TEVAL"                      # evaluate Task
    TASK_MONITOR = "TMON"                        # TaskMonitor
    TASK_STATUS_MANAGER = "TSMAN"                # TaskStateManager
    ICL_INFERENCER = "ICLI"                      # icl_inferencer
    ICL_EVALUATOR = "ICLE"                       # icl_evaluator
    ICL_RETRIEVER = "ICLR"                       # icl_retriever
    DATASET = "DSET"                             # dataset
    MODEL = "MODEL"                              # model
    CALCULATOR = "CALC"                          # calculator
    UTILS = "UTILS"                              # other utils func
    UNKNOWN = "UNK"                              # unknown module


@unique
class ErrorType(Enum):
    UNKNOWN = "UNK"     # unknown error type
    DEPENDENCY = "DEPENDENCY"     # third party error type
    IMPLEMENTATION = "IMPL"     # implementation error type
    COMMAND = "CMD"     # command error type
    CONFIG = "CFG"     # config error type
    MATCH = "MATCH"     # pattern match error type
    FILE = "FILE"     # file error type
    DATA = "DATA"     # data error type
    METRIC = "MTRC"     # metric error type
    TYPE = "TYPE"     # type error type
    MODULE = "MOD"     # module error type
    PARAM = "PARAM"     # parameter error type
    RUNTIME = "RUNTIME"     # runtime error type
    TASK = "TASK"     # task error type

class BaseErrorCode:
    FAQ_BASE_URL = "https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/faqs/error_codes.html#"

    def __init__(self, code_name: str, module: ErrorModule, err_type: ErrorType, code: int,
                 message: str):
        """
        Args:
            code_name (str): error code name (just for developer to check full_code)
            module (ErrorModule): error module
            err_type (ErrorType): error type
            code (int): error code number
            message (str): error message

        """
        self.module = module
        self.err_type = err_type
        self.code = code
        self.message = message
        self.faq_url = self.FAQ_BASE_URL + self.heading_id
        if code_name != self.full_code:
            raise ValueError(f"code_name {code_name} is not equal to full_code {self.full_code}")

    @property
    def full_code(self) -> str:

        return f"{self.module.value}-{self.err_type.value}-{self.code:03d}"
    @property
    def heading_id(self) -> str:
        return self.full_code.lower()

    def __str__(self) -> str:
        return f"{self.full_code}: {self.message}"



class ErrorCodeManager:
    def __init__(self):
        self._error_codes: Dict[str, BaseErrorCode] = {}

    def register(self, ec: BaseErrorCode) -> None:
        if ec.full_code in self._error_codes:
            raise ValueError(f"error code {ec.full_code} is exist!")
        self._error_codes[ec.full_code] = ec

    def get(self, full_code: str) -> Optional[BaseErrorCode]:
        return self._error_codes.get(full_code)

    def list_all(self) -> Dict[str, BaseErrorCode]:
        return self._error_codes.copy()

# error code consts
class TMAN_CODES:
    UNKNOWN_ERROR = BaseErrorCode("TMAN-UNK-001", ErrorModule.TASK_MANAGER, ErrorType.UNKNOWN, 1, "unknown error of task manager")
    CMD_MISS_REQUIRED_ARG = BaseErrorCode("TMAN-CMD-001", ErrorModule.TASK_MANAGER, ErrorType.COMMAND, 1, "command miss required argument") # docs coverd
    INVALID_ARG_VALUE_IN_CMD = BaseErrorCode("TMAN-CMD-002", ErrorModule.TASK_MANAGER, ErrorType.COMMAND, 2, "invalid argument value in command") # docs coverd
    INVAILD_SYNTAX_IN_CFG_CONTENT = BaseErrorCode("TMAN-CFG-001", ErrorModule.TASK_MANAGER, ErrorType.CONFIG, 1, "invaild syntax in config content") # docs coverd
    CFG_CONTENT_MISS_REQUIRED_PARAM = BaseErrorCode("TMAN-CFG-002", ErrorModule.TASK_MANAGER, ErrorType.CONFIG, 2, "config content miss required param") # docs coverd
    TYPE_ERROR_IN_CFG_PARAM = BaseErrorCode("TMAN-CFG-003", ErrorModule.TASK_MANAGER, ErrorType.CONFIG, 3, "type error in config param") # docs coverd


class PARTI_CODES:
    UNKNOWN_ERROR = BaseErrorCode("PARTI-UNK-001", ErrorModule.PARTITIONER, ErrorType.UNKNOWN, 1, "unknown error of partitioner")
    OUT_DIR_PERMISSION_DENIED = BaseErrorCode("PARTI-FILE-001", ErrorModule.PARTITIONER, ErrorType.FILE, 1, "out dir permission denied") # docs coverd


class SUMM_CODES:
    UNKNOWN_ERROR = BaseErrorCode("SUMM-UNK-001", ErrorModule.SUMMARY, ErrorType.UNKNOWN, 1, "unknown error of summary")
    NOT_SUPPORTED_DATASET_TYPES = BaseErrorCode("SUMM-TYPE-001", ErrorModule.SUMMARY, ErrorType.TYPE, 1, "not support mixed dataset_abbr type") # docs coverd
    NO_PERF_DATA_FILE = BaseErrorCode("SUMM-FILE-001", ErrorModule.SUMMARY, ErrorType.FILE, 1, "can't find detail perf data file") # docs coverd
    DIFF_STRUCTURE_OF_PERF_DATA = BaseErrorCode("SUMM-MTRC-001", ErrorModule.SUMMARY, ErrorType.METRIC, 1, "different structure of perf data")  # docs coverd


class RUNNER_CODES:
    UNKNOWN_ERROR = BaseErrorCode("RUNNER-UNK-001", ErrorModule.RUNNER, ErrorType.UNKNOWN, 1, "unknown error of runner")
    TASK_FAILED = BaseErrorCode("RUNNER-TASK-001", ErrorModule.RUNNER, ErrorType.TASK, 1, "task failed")  # docs coverd


class TMON_CODES:
    UNKNOWN_ERROR = BaseErrorCode("TMON-UNK-001", ErrorModule.TASK_MONITOR, ErrorType.UNKNOWN, 1, "unknown error of task monitor")


class TSMAN_CODES:
    UNKNOWN_ERROR = BaseErrorCode("TSMAN-UNK-001", ErrorModule.TASK_STATUS_MANAGER, ErrorType.UNKNOWN, 1, "unknown error of task state manager")


class TASK_CODES:
    UNKNOWN_ERROR = BaseErrorCode("TASK-UNK-001", ErrorModule.TASK, ErrorType.UNKNOWN, 1, "unknown error of task")


class TINFER_CODES:
    UNKNOWN_ERROR = BaseErrorCode("TINFER-UNK-001", ErrorModule.TASK_INFER, ErrorType.UNKNOWN, 1, "unknown error of infer task")
    CONCURRENCY_ERROR = BaseErrorCode("TINFER-PARAM-001", ErrorModule.TASK_INFER, ErrorType.PARAM, 1, "concurrency error of infer task") # docs coverd
    FAILED_TO_START_WORKER = BaseErrorCode("TINFER-IMPL-001", ErrorModule.TASK_INFER, ErrorType.IMPLEMENTATION, 1, "failed to start worker") # docs coverd
    NUM_RETURN_SEQUENCES_NOT_POSITIVE = BaseErrorCode("TINFER-PARAM-002", ErrorModule.TASK_INFER, ErrorType.PARAM, 2, "num_return sequences must be a positive integer") # docs coverd
    INVALID_RAMP_UP_STRATEGY = BaseErrorCode("TINFER-PARAM-004", ErrorModule.TASK_INFER, ErrorType.PARAM, 4, "invalid ramp up strategy") # docs coverd
    VIRTUAL_MEMORY_USAGE_TOO_HIGH = BaseErrorCode("TINFER-PARAM-005", ErrorModule.TASK_INFER, ErrorType.PARAM, 5, "virtual memory usage too high") # docs coverd
    WARMUP_FAILED = BaseErrorCode("TINFER-RUNTIME-001", ErrorModule.TASK_INFER, ErrorType.RUNTIME, 1, "warmup failed")


class TEVAL_CODES:
    UNKNOWN_ERROR = BaseErrorCode("TEVAL-UNK-001", ErrorModule.TASK_EVALUATE, ErrorType.UNKNOWN, 1, "unknown error of evaluate task")
    N_K_ILLEGAL = BaseErrorCode("TEVAL-PARAM-001", ErrorModule.TASK_EVALUATE, ErrorType.PARAM, 1, "n and k parameters illegal") # docs coverd


class ICLI_CODES:
    UNKNOWN_ERROR = BaseErrorCode("ICLI-UNK-001", ErrorModule.ICL_INFERENCER, ErrorType.UNKNOWN, 1, "unknown error of icl inferencer")
    INVALID_PARAM_VALUE = BaseErrorCode("ICLI-PARAM-001", ErrorModule.ICL_INFERENCER, ErrorType.PARAM, 1, "invalid parameter value") # docs coverd
    MULTITRUN_MODE_OUT_OF_RANGE = BaseErrorCode("ICLI-PARAM-002", ErrorModule.ICL_INFERENCER, ErrorType.PARAM, 2, "multiturn mode out of range") # docs coverd
    CONCURRENCY_NOT_SET_IN_PRESSEURE_MODE = BaseErrorCode("ICLI-PARAM-003", ErrorModule.ICL_INFERENCER, ErrorType.PARAM, 3, "concurrency not set in pressure mode") # docs coverd
    BATCH_SIZE_OUT_OF_RANGE = BaseErrorCode("ICLI-PARAM-004", ErrorModule.ICL_INFERENCER, ErrorType.PARAM, 4, "batch size out of range") # docs coverd
    PERF_MODE_NOT_SUPPORTED_FOR_PPL_INFERENCE = BaseErrorCode("ICLI-PARAM-006", ErrorModule.ICL_INFERENCER, ErrorType.PARAM, 6, "perf mode is not supported for ppl inference") # docs coverd
    STREAM_MODE_NOT_SUPPORTED_FOR_PPL_INFERENCE = BaseErrorCode("ICLI-PARAM-007", ErrorModule.ICL_INFERENCER, ErrorType.PARAM, 7, "stream mode is not supported for ppl inference") # docs coverd
    IMPLEMENTATION_ERROR_PPL_METHOD_NOT_IMPLEMENTED = BaseErrorCode("ICLI-IMPL-008", ErrorModule.ICL_INFERENCER, ErrorType.IMPLEMENTATION, 8, "ppl method not implemented") # docs coverd
    PPL_COMPUTE_ERROR_NO_VALID_TOKENS = BaseErrorCode("ICLI-IMPL-010", ErrorModule.ICL_INFERENCER, ErrorType.IMPLEMENTATION, 10, "no valid tokens with log probabilities found for PPL computation") # docs coverd


    WARMUP_GET_RESULT_FAILED = BaseErrorCode("ICLI-RUNTIME-001", ErrorModule.ICL_INFERENCER, ErrorType.RUNTIME, 1, "get result from cache queue failed") # docs coverd
    IMPLEMENTATION_ERROR_BFCL_V3_NOT_SUPPORT_PERF_MODE = BaseErrorCode("ICLI-IMPL-004", ErrorModule.ICL_INFERENCER, ErrorType.IMPLEMENTATION, 4, "bfcl v3 not support perf mode") # docs coverd
    IMPLEMENTATION_ERROR_BFCL_V3_NOT_SUPPORT_STREAM = BaseErrorCode("ICLI-IMPL-006", ErrorModule.ICL_INFERENCER, ErrorType.IMPLEMENTATION, 6, "bfcl v3 not support stream") # docs coverd

    INFER_RESULT_WRITE_ERROR = BaseErrorCode("ICLI-FILE-001", ErrorModule.ICL_INFERENCER, ErrorType.FILE, 1, "failed to write results files") # docs coverd
    SQLITE_WRITE_ERROR = BaseErrorCode("ICLI-FILE-002", ErrorModule.ICL_INFERENCER, ErrorType.FILE, 2, "failed to write results to sqlite database") # docs coverd


class ICLE_CODES:
    UNKNOWN_ERROR = BaseErrorCode("ICLE-UNK-001", ErrorModule.ICL_EVALUATOR, ErrorType.UNKNOWN, 1, "unknown error of icl evaluator")

    REPLICATION_LENGTH_MISMATCH = BaseErrorCode("ICLE-DATA-002", ErrorModule.ICL_EVALUATOR, ErrorType.DATA, 2, "replication length mismatch") # docs coverd


class ICLR_CODES:
    UNKNOWN_ERROR = BaseErrorCode("ICLR-UNK-001", ErrorModule.ICL_RETRIEVER, ErrorType.UNKNOWN, 1, "unknown error of icl retriever")

    TEMPLATE_TYPE_ERROR = BaseErrorCode("ICLR-TYPE-001", ErrorModule.ICL_RETRIEVER, ErrorType.TYPE, 1, "template type error") # docs coverd
    TEMPLATE_VALUE_TYPE_ERROR = BaseErrorCode("ICLR-TYPE-002", ErrorModule.ICL_RETRIEVER, ErrorType.TYPE, 2, "template value type error") # docs coverd

    TEMPLATE_ICE_TOKEN_NOT_IN_VALUE = BaseErrorCode("ICLR-PARAM-001", ErrorModule.ICL_RETRIEVER, ErrorType.PARAM, 1, "ice token not in value of template") # docs coverd
    TEMPLATE_ICE_TOKEN_NOT_IN_TEMPLATE = BaseErrorCode("ICLR-PARAM-002", ErrorModule.ICL_RETRIEVER, ErrorType.PARAM, 2, "ice template not set") # docs coverd
    MULTIMODAL_TEMPLATE_TYPE_ERROR = BaseErrorCode("ICLR-PARAM-003", ErrorModule.ICL_RETRIEVER, ErrorType.PARAM, 3, "multimodal template type error") # docs coverd
    FIX_K_RETRIEVER_INDEX_OUT_OF_RANGE = BaseErrorCode("ICLR-PARAM-004", ErrorModule.ICL_RETRIEVER, ErrorType.PARAM, 4, "fix-k retriever index out of range") # docs coverd

    IMPLEMENTATION_ERROR_ICE_TOKEN_NOT_PROVIDED = BaseErrorCode("ICLR-IMPL-002", ErrorModule.ICL_RETRIEVER, ErrorType.IMPLEMENTATION, 2, "ice token not provided") # docs coverd
    IMPLEMENTATION_ERROR_PROMPT_TEMPLATE_NOT_PROVIDED = BaseErrorCode("ICLR-IMPL-003", ErrorModule.ICL_RETRIEVER, ErrorType.IMPLEMENTATION, 3, "template not provided") # docs coverd


class MODEL_CODES:
    UNKNOWN_ERROR = BaseErrorCode("MODEL-UNK-001", ErrorModule.MODEL, ErrorType.UNKNOWN, 1, "unknown error of model")
    PARSE_TEXT_RSP_NOT_IMPLEMENTED = BaseErrorCode("MODEL-IMPL-001", ErrorModule.MODEL, ErrorType.IMPLEMENTATION, 1, "parse text response not implemented") # docs coverd
    PARSE_STREAM_RSP_NOT_IMPLEMENTED = BaseErrorCode("MODEL-IMPL-002", ErrorModule.MODEL, ErrorType.IMPLEMENTATION, 2, "parse stream response not implemented") # docs coverd

    INVALID_ROLE_IN_PROMPT_TEMPLATE = BaseErrorCode("MODEL-PARAM-002", ErrorModule.MODEL, ErrorType.PARAM, 2, "invalid role in prompt template") # docs coverd
    INVALID_ROLE_IN_CHAT_TEMPLATE = BaseErrorCode("MODEL-PARAM-003", ErrorModule.MODEL, ErrorType.PARAM, 3, "invalid role in chat template") # docs coverd
    MISS_REQUIRED_PARAM_IN_META_TEMPLATE = BaseErrorCode("MODEL-PARAM-004", ErrorModule.MODEL, ErrorType.PARAM, 4, "miss required param in meta template") # user not awared
    ROLE_IN_META_TEMPLATE_IS_NOT_UNIQUE = BaseErrorCode("MODEL-PARAM-005", ErrorModule.MODEL, ErrorType.PARAM, 5, "role in meta prompt must be unique!") # user not awared

    MIX_STR_WITHOUT_EXPLICIT_ROLE = BaseErrorCode("MODEL-TYPE-001", ErrorModule.MODEL, ErrorType.TYPE, 1, "mixing str without explicit role is not allowed") # docs coverd
    PARSE_TEMPLATE_INVALID_TYPE = BaseErrorCode("MODEL-TYPE-002", ErrorModule.MODEL, ErrorType.TYPE, 2, "invalid prompt template type") # user not awared
    PARSE_TEMPLATE_INVALID_MODE = BaseErrorCode("MODEL-TYPE-003", ErrorModule.MODEL, ErrorType.TYPE, 3, "invalid mode in prompt template") # user not awared
    INVALID_TYPE_OF_PARAM_IN_META_TEMPLATE = BaseErrorCode("MODEL-TYPE-004", ErrorModule.MODEL, ErrorType.TYPE, 4, "invalid type of param in meta template") # user not awared

    GET_SERVICE_MODEL_PATH_FAILED = BaseErrorCode("MODEL-DATA-001", ErrorModule.MODEL, ErrorType.DATA, 1, "fail to get service model path") # docs coverd
    INVALID_PROMPT_CONTENT = BaseErrorCode("MODEL-DATA-002", ErrorModule.MODEL, ErrorType.DATA, 2, "invalid prompt content") # docs coverd
    PARSE_TEXT_RSP_INVALID_FORMAT = BaseErrorCode("MODEL-DATA-003", ErrorModule.MODEL, ErrorType.DATA, 3, "parse text response invalid format") # docs coverd

    MAX_SEQ_LEN_NOT_FOUND = BaseErrorCode("MODEL-CFG-001", ErrorModule.MODEL, ErrorType.CONFIG, 1, "max_seq_len is not provided and cannot be inferred from the model config.") # docs coverd
    MODULE_NOT_FOUND = BaseErrorCode("MODEL-MOD-001", ErrorModule.MODEL, ErrorType.MODULE, 1, "module not found") # docs coverd


class UNK_CODES:
    UNKNOWN_ERROR = BaseErrorCode("UNK-UNK-001", ErrorModule.UNKNOWN, ErrorType.UNKNOWN, 1, "unknown error")


class UTILS_CODES:
    UNKNOWN_ERROR = BaseErrorCode("UTILS-UNK-001", ErrorModule.UTILS, ErrorType.UNKNOWN, 1, "unknown error of utils")
    MATCH_CONFIG_FILE_FAILED = BaseErrorCode("UTILS-MATCH-001", ErrorModule.UTILS, ErrorType.MATCH, 1, "match config file failed") # docs coverd
    DEPENDENCY_MODULE_IMPORT_ERROR = BaseErrorCode("UTILS-DEPENDENCY-001", ErrorModule.UTILS, ErrorType.DEPENDENCY, 1, "third party dependency module import error") # docs coverd

    # Type validation errors
    INVALID_TYPE = BaseErrorCode("UTILS-TYPE-001", ErrorModule.UTILS, ErrorType.TYPE, 1, "invalid object type") # user not awared
    INVALID_DATASET_TYPE = BaseErrorCode("UTILS-TYPE-002", ErrorModule.UTILS, ErrorType.TYPE, 2, "invalid dataset type") # user not awared
    INVALID_LIST_TYPE = BaseErrorCode("UTILS-TYPE-003", ErrorModule.UTILS, ErrorType.TYPE, 3, "invalid list type") # user not awared
    INVALID_STRING_TYPE = BaseErrorCode("UTILS-TYPE-004", ErrorModule.UTILS, ErrorType.TYPE, 4, "invalid string type") # user not awared
    INVALID_DICT_TYPE = BaseErrorCode("UTILS-TYPE-005", ErrorModule.UTILS, ErrorType.TYPE, 5, "invalid dict type") # user not awared
    INVALID_TYPE_SPECIFIER = BaseErrorCode("UTILS-TYPE-006", ErrorModule.UTILS, ErrorType.TYPE, 6, "invalid type specifier") # user not awared
    TYPE_MISMATCH = BaseErrorCode("UTILS-TYPE-007", ErrorModule.UTILS, ErrorType.TYPE, 7, "type mismatch in validation") # user not awared
    ARGUMENT_TOO_LARGE = BaseErrorCode("UTILS-TYPE-008", ErrorModule.UTILS, ErrorType.TYPE, 8, "argument value too large") # docs coverd
    INVALID_INTEGER_TYPE = BaseErrorCode("UTILS-TYPE-009", ErrorModule.UTILS, ErrorType.TYPE, 9, "invalid integer type") # docs coverd
    ARGUMENT_TOO_SMALL = BaseErrorCode("UTILS-TYPE-010", ErrorModule.UTILS, ErrorType.TYPE, 10, "argument value too small") # docs coverd

    # Parameter validation errors
    ROOT_PATH_NOT_SET = BaseErrorCode("UTILS-PARAM-001", ErrorModule.UTILS, ErrorType.PARAM, 1, "root_path not set") # user not awared
    INVALID_REQUEST_COUNT = BaseErrorCode("UTILS-PARAM-002", ErrorModule.UTILS, ErrorType.PARAM, 2, "invalid request_count value") # docs coverd
    INVALID_MIN_MAX_VALUE = BaseErrorCode("UTILS-PARAM-003", ErrorModule.UTILS, ErrorType.PARAM, 3, "invalid min_value or max_value") # docs coverd
    MIN_GREATER_THAN_MAX = BaseErrorCode("UTILS-PARAM-004", ErrorModule.UTILS, ErrorType.PARAM, 4, "min_value greater than max_value") # docs coverd
    MISSING_PARAMS = BaseErrorCode("UTILS-PARAM-005", ErrorModule.UTILS, ErrorType.PARAM, 5, "missing required params") # docs coverd
    INVALID_PERCENTAGE_DISTRIBUTE = BaseErrorCode("UTILS-PARAM-006", ErrorModule.UTILS, ErrorType.PARAM, 6, "invalid percentage_distribute format") # docs coverd
    UNSUPPORTED_DISTRIBUTION_METHOD = BaseErrorCode("UTILS-PARAM-007", ErrorModule.UTILS, ErrorType.PARAM, 7, "unsupported distribution method") # docs coverd
    ILLEGAL_KEYS_IN_CONFIG = BaseErrorCode("UTILS-PARAM-008", ErrorModule.UTILS, ErrorType.PARAM, 8, "illegal keys in configuration") # docs coverd

    # File errors
    TOKENIZER_PATH_NOT_FOUND = BaseErrorCode("UTILS-FILE-002", ErrorModule.UTILS, ErrorType.FILE, 2, "tokenizer path not found") # docs coverd
    TOKENIZER_LOAD_FAILED = BaseErrorCode("UTILS-FILE-003", ErrorModule.UTILS, ErrorType.FILE, 3, "tokenizer load failed") # docs coverd
    CHART_FILE_NOT_FOUND = BaseErrorCode("UTILS-FILE-004", ErrorModule.UTILS, ErrorType.FILE, 4, "chart file not found") # user not awared

    # Config validation errors
    SYNTHETIC_DS_MISS_REQUIRED_PARAM = BaseErrorCode("UTILS-CFG-001", ErrorModule.UTILS, ErrorType.CONFIG, 1, "synthetic dataset miss required param") # docs coverd
    MODEL_CONFIG_VALIDATE_FAILED = BaseErrorCode("UTILS-CFG-002", ErrorModule.UTILS, ErrorType.CONFIG, 2, "model config validate failed") # docs coverd
    ILLEGAL_MODEL_ATTR = BaseErrorCode("UTILS-CFG-003", ErrorModule.UTILS, ErrorType.CONFIG, 3, "illegal model attr in config") # docs coverd
    MIXED_MODEL_ATTRS = BaseErrorCode("UTILS-CFG-004", ErrorModule.UTILS, ErrorType.CONFIG, 4, "mixed model attrs in config") # docs coverd
    MM_CUSTOM_DATASET_WRONG_FORMAT = BaseErrorCode("UTILS-CFG-008", ErrorModule.UTILS, ErrorType.CONFIG, 8, "invalid mm custom dataset") # docs coverd


class CALC_CODES:
    UNKNOWN_ERROR = BaseErrorCode("CALC-UNK-001", ErrorModule.CALCULATOR, ErrorType.UNKNOWN, 1, "unknown error of calculator")
    INVALID_METRIC_DATA = BaseErrorCode("CALC-MTRC-001", ErrorModule.CALCULATOR, ErrorType.METRIC, 1, "invalid content of metric data") # docs coverd
    DUMPING_RESULT_FAILED = BaseErrorCode("CALC-FILE-001", ErrorModule.CALCULATOR, ErrorType.FILE, 1, "fail to dump result to file") # docs coverd
    ALL_REQUEST_DATAS_INVALID = BaseErrorCode("CALC-DATA-001", ErrorModule.CALCULATOR, ErrorType.DATA, 1, "all request datas are invalid") # docs coverd
    CAN_NOT_FIND_STABLE_STAGE = BaseErrorCode("CALC-DATA-002", ErrorModule.CALCULATOR, ErrorType.DATA, 2, "invalid response datas") # docs coverd


class DSET_CODES:
    UNKNOWN_ERROR = BaseErrorCode("DSET-UNK-001", ErrorModule.DATASET, ErrorType.UNKNOWN, 1, "unknown error of dataset")

    INVALID_DATASET_CONFIG = BaseErrorCode("DSET-CFG-001", ErrorModule.DATASET, ErrorType.CONFIG, 1, "invalid dataset config") # docs coverd
    # File related errors
    FILE_NOT_FOUND = BaseErrorCode("DSET-FILE-001", ErrorModule.DATASET, ErrorType.FILE, 1, "dataset file not found") # docs coverd

    # Data related errors
    DATA_INVALID_STRUCTURE = BaseErrorCode("DSET-DATA-002", ErrorModule.DATASET, ErrorType.DATA, 2, "dataset has invalid structure") # docs coverd
    DATA_PREPROCESSING_ERROR = BaseErrorCode("DSET-DATA-005", ErrorModule.DATASET, ErrorType.DATA, 5, "data preprocessing or cleaning failed") # user not awared
    INVALID_DATA_TYPE = BaseErrorCode("DSET-DATA-006", ErrorModule.DATASET, ErrorType.DATA, 6, "data type does not match expected type") # user not awared

    # Parameter related errors
    INVALID_REPEAT_FACTOR = BaseErrorCode("DSET-PARAM-002", ErrorModule.DATASET, ErrorType.PARAM, 2, "invalid repeat factor") # docs coverd
    INVALID_PARAM_VALUE = BaseErrorCode("DSET-PARAM-004", ErrorModule.DATASET, ErrorType.PARAM, 4, "invalid parameter value") # docs coverd

    # Dependency related errors
    EVALUATION_LIBRARY_NOT_INSTALLED = BaseErrorCode("DSET-DEPENDENCY-002", ErrorModule.DATASET, ErrorType.DEPENDENCY, 2, "evaluation library not installed") # docs coverd

    # Evaluation related errors
    PREDICTION_LENGTH_MISMATCH = BaseErrorCode("DSET-MTRC-001", ErrorModule.DATASET, ErrorType.METRIC, 1, "prediction and reference have different length") # user not awared
    INVALID_MBPP_METRIC = BaseErrorCode("DSET-MTRC-003", ErrorModule.DATASET, ErrorType.METRIC, 3, "invalid MBPP metric type") # user not awared


ERROR_CODES_CLASSES = [
    TMAN_CODES,
    PARTI_CODES,
    SUMM_CODES,
    RUNNER_CODES,
    TMON_CODES,
    TSMAN_CODES,
    TINFER_CODES,
    TEVAL_CODES,
    ICLI_CODES,
    ICLE_CODES,
    ICLR_CODES,
    MODEL_CODES,
    UNK_CODES,
    UTILS_CODES,
    CALC_CODES,
    DSET_CODES,
]

# init error code manager
error_manager = ErrorCodeManager()

# regist all errors
for error_codes_class in ERROR_CODES_CLASSES:
    for error_code in error_codes_class.__dict__.values():
        if isinstance(error_code, BaseErrorCode):
            error_manager.register(error_code)
