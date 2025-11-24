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
    DATASETS = "DATASETS"                        # datasets
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

    def register(self, error_code: BaseErrorCode) -> None:
        if error_code.full_code in self._error_codes:
            raise ValueError(f"error code {error_code.full_code} is exist!")
        self._error_codes[error_code.full_code] = error_code

    def get(self, full_code: str) -> Optional[BaseErrorCode]:
        return self._error_codes.get(full_code)

    def list_all(self) -> Dict[str, BaseErrorCode]:
        return self._error_codes.copy()

# error code consts
class TMAN_CODES:
    UNKNOWN_ERROR = BaseErrorCode("TMAN-UNK-001", ErrorModule.TASK_MANAGER, ErrorType.UNKNOWN, 1, "unknown error of task manager")
    CMD_MISS_REQUIRED_ARG = BaseErrorCode("TMAN-CMD-001", ErrorModule.TASK_MANAGER, ErrorType.COMMAND, 1, "command miss required argument")
    INVALID_ARG_VALUE_IN_CMD = BaseErrorCode("TMAN-CMD-002", ErrorModule.TASK_MANAGER, ErrorType.COMMAND, 2, "invalid argument value in command")
    INVAILD_SYNTAX_IN_CFG_CONTENT = BaseErrorCode("TMAN-CFG-001", ErrorModule.TASK_MANAGER, ErrorType.CONFIG, 1, "invaild syntax in config content")
    CFG_CONTENT_MISS_REQUIRED_PARAM = BaseErrorCode("TMAN-CFG-002", ErrorModule.TASK_MANAGER, ErrorType.CONFIG, 2, "config content miss required param")
    TYPE_ERROR_IN_CFG_PARAM = BaseErrorCode("TMAN-CFG-003", ErrorModule.TASK_MANAGER, ErrorType.CONFIG, 3, "type error in config param")


class PARTI_CODES:
    UNKNOWN_ERROR = BaseErrorCode("PARTI-UNK-001", ErrorModule.PARTITIONER, ErrorType.UNKNOWN, 1, "unknown error of partitioner")
    OUT_DIR_PERMISSION_DENIED = BaseErrorCode("PARTI-FILE-001", ErrorModule.PARTITIONER, ErrorType.FILE, 1, "out dir permission denied")


class SUMM_CODES:
    UNKNOWN_ERROR = BaseErrorCode("SUMM-UNK-001", ErrorModule.SUMMARY, ErrorType.UNKNOWN, 1, "unknown error of summary")
    NOT_SUPPORTED_DATASET_TYPES = BaseErrorCode("SUMM-TYPE-001", ErrorModule.SUMMARY, ErrorType.TYPE, 1, "not support mixed dataset_abbr type")
    NO_PERF_DATA_FILE = BaseErrorCode("SUMM-FILE-001", ErrorModule.SUMMARY, ErrorType.FILE, 1, "can't find detail perf data file")
    DIFF_STRUCTURE_OF_PERF_DATA = BaseErrorCode("SUMM-MTRC-001", ErrorModule.SUMMARY, ErrorType.METRIC, 1, "different structure of perf data")


class RUNNER_CODES:
    UNKNOWN_ERROR = BaseErrorCode("RUNNER-UNK-001", ErrorModule.RUNNER, ErrorType.UNKNOWN, 1, "unknown error of runner")
    TASK_FAILED = BaseErrorCode("RUNNER-TASK-001", ErrorModule.RUNNER, ErrorType.TASK, 1, "task failed")


class TMON_CODES:
    UNKNOWN_ERROR = BaseErrorCode("TMON-UNK-001", ErrorModule.TASK_MONITOR, ErrorType.UNKNOWN, 1, "unknown error of task monitor")


class TSMAN_CODES:
    UNKNOWN_ERROR = BaseErrorCode("TSMAN-UNK-001", ErrorModule.TASK_STATUS_MANAGER, ErrorType.UNKNOWN, 1, "unknown error of task state manager")

class TASK_CODES:
    UNKNOWN_ERROR = BaseErrorCode("TASK-UNK-001", ErrorModule.TASK, ErrorType.UNKNOWN, 1, "unknown error of task")
    MODEL_MULTIPLE = BaseErrorCode("TASK-PARAM-001", ErrorModule.TASK, ErrorType.PARAM, 1, "task only supports one model")

class TINFER_CODES:
    UNKNOWN_ERROR = BaseErrorCode("TINFER-UNK-001", ErrorModule.TASK_INFER, ErrorType.UNKNOWN, 1, "unknown error of infer task")
    CONCURRENCY_ERROR = BaseErrorCode("TINFER-PARAM-001", ErrorModule.TASK_INFER, ErrorType.PARAM, 1, "concurrency error of infer task")
    FAILED_TO_START_WORKER = BaseErrorCode("TINFER-IMPL-001", ErrorModule.TASK_INFER, ErrorType.IMPLEMENTATION, 1, "failed to start worker")
    NUM_RETURN_SEQUENCES_NOT_POSITIVE = BaseErrorCode("TINFER-PARAM-002", ErrorModule.TASK_INFER, ErrorType.PARAM, 2, "num_return sequences must be a positive integer")
    N_NOT_POSITIVE = BaseErrorCode("TINFER-PARAM-003", ErrorModule.TASK_INFER, ErrorType.PARAM, 3, "n expected a positive integer")
    INVALID_RAMP_UP_STRATEGY = BaseErrorCode("TINFER-PARAM-004", ErrorModule.TASK_INFER, ErrorType.PARAM, 4, "invalid ramp up strategy")
    VIRTUAL_MEMORY_USAGE_TOO_HIGH = BaseErrorCode("TINFER-PARAM-005", ErrorModule.TASK_INFER, ErrorType.PARAM, 5, "virtual memory usage too high")
    
class TEVAL_CODES:
    UNKNOWN_ERROR = BaseErrorCode("TEVAL-UNK-001", ErrorModule.TASK_EVALUATE, ErrorType.UNKNOWN, 1, "unknown error of evaluate task")
    N_K_ILLEGAL = BaseErrorCode("TEVAL-PARAM-001", ErrorModule.TASK_EVALUATE, ErrorType.PARAM, 1, "n and k parameters illegal")
    MODEL_PRED_STRS_EMPTY = BaseErrorCode("TEVAL-PARAM-002", ErrorModule.TASK_EVALUATE, ErrorType.PARAM, 2, "model pred strs empty")


class ICLI_CODES:
    UNKNOWN_ERROR = BaseErrorCode("ICLI-UNK-001", ErrorModule.ICL_INFERENCER, ErrorType.UNKNOWN, 1, "unknown error of icl inferencer")

    INVALID_PARAM_VALUE = BaseErrorCode("ICLI-PARAM-001", ErrorModule.ICL_INFERENCER, ErrorType.PARAM, 1, "invalid parameter value")
    MULTITRUN_MODE_OUT_OF_RANGE = BaseErrorCode("ICLI-PARAM-002", ErrorModule.ICL_INFERENCER, ErrorType.PARAM, 2, "multiturn mode out of range")
    CONCURRENCY_NOT_SET_IN_PRESSEURE_MODE = BaseErrorCode("ICLI-PARAM-003", ErrorModule.ICL_INFERENCER, ErrorType.PARAM, 3, "concurrency not set in pressure mode")
    BATCH_SIZE_OUT_OF_RANGE = BaseErrorCode("ICLI-PARAM-004", ErrorModule.ICL_INFERENCER, ErrorType.PARAM, 4, "batch size out of range")
    INVALID_OUTPUT_FILEPATH = BaseErrorCode("ICLI-PARAM-005", ErrorModule.ICL_INFERENCER, ErrorType.PARAM, 5, "invalid output jsonl filepath")
    PERF_MODE_NOT_SUPPORTED_FOR_PPL_INFERENCE = BaseErrorCode("ICLI-PARAM-006", ErrorModule.ICL_INFERENCER, ErrorType.PARAM, 6, "perf mode is not supported for ppl inference")
    STREAM_MODE_NOT_SUPPORTED_FOR_PPL_INFERENCE = BaseErrorCode("ICLI-PARAM-007", ErrorModule.ICL_INFERENCER, ErrorType.PARAM, 7, "stream mode is not supported for ppl inference")
    IMPLEMENTATION_ERROR_PPL_METHOD_NOT_IMPLEMENTED = BaseErrorCode("ICLI-IMPL-008", ErrorModule.ICL_INFERENCER, ErrorType.IMPLEMENTATION, 8, "ppl method not implemented")
    IMPLEMENTATION_ERROR_OUTPUT_NOT_PPL_RESPONSE_OUTPUT = BaseErrorCode("ICLI-IMPL-009", ErrorModule.ICL_INFERENCER, ErrorType.IMPLEMENTATION, 9, "output is not a PPLResponseOutput")
    PPL_COMPUTE_ERROR_NO_VALID_TOKENS = BaseErrorCode("ICLI-IMPL-010", ErrorModule.ICL_INFERENCER, ErrorType.IMPLEMENTATION, 10, "no valid tokens with log probabilities found for PPL computation")
    
    WARMUP_GET_RESULT_FAILED = BaseErrorCode("ICLI-RUNTIME-001", ErrorModule.ICL_INFERENCER, ErrorType.RUNTIME, 1, "get result from cache queue failed")
    WARMUP_FAILED = BaseErrorCode("ICLI-RUNTIME-002", ErrorModule.ICL_INFERENCER, ErrorType.RUNTIME, 2, "warmup failed")
    WARMUP_EMPTY_RESULT = BaseErrorCode("ICLI-RUNTIME-003", ErrorModule.ICL_INFERENCER, ErrorType.RUNTIME, 3, "empty result from cache queue")

    IMPLEMENTATION_ERROR = BaseErrorCode("ICLI-IMPL-001", ErrorModule.ICL_INFERENCER, ErrorType.IMPLEMENTATION, 1, "not implemented error")
    IMPLEMENTATION_ERROR_DO_REQUEST_METHOD_NOT_IMPLEMENTED = BaseErrorCode("ICLI-IMPL-002", ErrorModule.ICL_INFERENCER, ErrorType.IMPLEMENTATION, 2, "do request method for api inferencer not implemented")
    IMPLEMENTATION_ERROR_BATCH_INFERENCE_METHOD_NOT_IMPLEMENTED = BaseErrorCode("ICLI-IMPL-003", ErrorModule.ICL_INFERENCER, ErrorType.IMPLEMENTATION, 3, "batch inference method for local inferencer not implemented")
    IMPLEMENTATION_ERROR_BFCL_V3_NOT_SUPPORT_PERF_MODE = BaseErrorCode("ICLI-IMPL-004", ErrorModule.ICL_INFERENCER, ErrorType.IMPLEMENTATION, 4, "bfcl v3 not support perf mode")
    IMPLEMENTATION_ERROR_OUTPUT_NOT_FUNCTION_CALL_OUTPUT = BaseErrorCode("ICLI-IMPL-005", ErrorModule.ICL_INFERENCER, ErrorType.IMPLEMENTATION, 5, "output is not correct type")
    IMPLEMENTATION_ERROR_BFCL_V3_NOT_SUPPORT_STREAM = BaseErrorCode("ICLI-IMPL-006", ErrorModule.ICL_INFERENCER, ErrorType.IMPLEMENTATION, 6, "bfcl v3 not support stream")

    INFER_RESULT_WRITE_ERROR = BaseErrorCode("ICLI-FILE-001", ErrorModule.ICL_INFERENCER, ErrorType.FILE, 1, "failed to write results files")
    SQLITE_WRITE_ERROR = BaseErrorCode("ICLI-FILE-002", ErrorModule.ICL_INFERENCER, ErrorType.FILE, 2, "failed to write results to sqlite database")

class ICLE_CODES:
    UNKNOWN_ERROR = BaseErrorCode("ICLE-UNK-001", ErrorModule.ICL_EVALUATOR, ErrorType.UNKNOWN, 1, "unknown error of icl evaluator")

    PREDICTION_LENGTH_MISMATCH = BaseErrorCode("ICLE-DATA-001", ErrorModule.ICL_EVALUATOR, ErrorType.DATA, 1, "prediction result length mismatch")
    REPLICATION_LENGTH_MISMATCH = BaseErrorCode("ICLE-DATA-002", ErrorModule.ICL_EVALUATOR, ErrorType.DATA, 2, "replication length mismatch")

    IMPLEMENTATION_ERROR = BaseErrorCode("ICLE-IMPL-001", ErrorModule.ICL_EVALUATOR, ErrorType.IMPLEMENTATION, 1, "not implemented error")

class ICLR_CODES:
    UNKNOWN_ERROR = BaseErrorCode("ICLR-UNK-001", ErrorModule.ICL_RETRIEVER, ErrorType.UNKNOWN, 1, "unknown error of icl retriever")

    TEMPLATE_TYPE_ERROR = BaseErrorCode("ICLR-TYPE-001", ErrorModule.ICL_RETRIEVER, ErrorType.TYPE, 1, "template type error")
    TEMPLATE_VALUE_TYPE_ERROR = BaseErrorCode("ICLR-TYPE-002", ErrorModule.ICL_RETRIEVER, ErrorType.TYPE, 2, "template value type error")

    TEMPLATE_ICE_TOKEN_NOT_IN_VALUE = BaseErrorCode("ICLR-PARAM-001", ErrorModule.ICL_RETRIEVER, ErrorType.PARAM, 1, "ice token not in value of template")
    TEMPLATE_ICE_TOKEN_NOT_IN_TEMPLATE = BaseErrorCode("ICLR-PARAM-002", ErrorModule.ICL_RETRIEVER, ErrorType.PARAM, 2, "ice template not set")
    MULTIMODAL_TEMPLATE_TYPE_ERROR = BaseErrorCode("ICLR-PARAM-003", ErrorModule.ICL_RETRIEVER, ErrorType.PARAM, 3, "multimodal template type error")
    FIX_K_RETRIEVER_INDEX_OUT_OF_RANGE = BaseErrorCode("ICLR-PARAM-004", ErrorModule.ICL_RETRIEVER, ErrorType.PARAM, 4, "fix-k retriever index out of range")

    IMPLEMENTATION_ERROR = BaseErrorCode("ICLR-IMPL-001", ErrorModule.ICL_RETRIEVER, ErrorType.IMPLEMENTATION, 1, "not implemented error")
    IMPLEMENTATION_ERROR_ICE_TOKEN_NOT_PROVIDED = BaseErrorCode("ICLR-IMPL-002", ErrorModule.ICL_RETRIEVER, ErrorType.IMPLEMENTATION, 2, "ice token not provided")
    IMPLEMENTATION_ERROR_PROMPT_TEMPLATE_NOT_PROVIDED = BaseErrorCode("ICLR-IMPL-003", ErrorModule.ICL_RETRIEVER, ErrorType.IMPLEMENTATION, 3, "template not provided")


class MODEL_CODES:
    UNKNOWN_ERROR = BaseErrorCode("MODEL-UNK-001", ErrorModule.MODEL, ErrorType.UNKNOWN, 1, "unknown error of model")
    PARSE_TEXT_RSP_NOT_IMPLEMENTED = BaseErrorCode("MODEL-IMPL-001", ErrorModule.MODEL, ErrorType.IMPLEMENTATION, 1, "parse text response not implemented")
    PARSE_STREAM_RSP_NOT_IMPLEMENTED = BaseErrorCode("MODEL-IMPL-002", ErrorModule.MODEL, ErrorType.IMPLEMENTATION, 2, "parse stream response not implemented")

    INVALID_POS_IN_PROMPT_TEMPLATE = BaseErrorCode("MODEL-PARAM-001", ErrorModule.MODEL, ErrorType.PARAM, 1, "invalid pos in prompt template")
    INVALID_ROLE_IN_PROMPT_TEMPLATE = BaseErrorCode("MODEL-PARAM-002", ErrorModule.MODEL, ErrorType.PARAM, 2, "invalid role in prompt template")
    INVALID_ROLE_IN_CHAT_TEMPLATE = BaseErrorCode("MODEL-PARAM-003", ErrorModule.MODEL, ErrorType.PARAM, 3, "invalid role in chat template")
    MISS_REQUIRED_PARAM_IN_META_TEMPLATE = BaseErrorCode("MODEL-PARAM-004", ErrorModule.MODEL, ErrorType.PARAM, 4, "miss required param in meta template")
    ROLE_IN_META_TEMPLATE_IS_NOT_UNIQUE = BaseErrorCode("MODEL-PARAM-005", ErrorModule.MODEL, ErrorType.PARAM, 5, "role in meta prompt must be unique!")

    MIX_STR_WITHOUT_EXPLICIT_ROLE = BaseErrorCode("MODEL-TYPE-001", ErrorModule.MODEL, ErrorType.TYPE, 1, "mixing str without explicit role is not allowed")
    PARSE_TEMPLATE_INVALID_TYPE = BaseErrorCode("MODEL-TYPE-002", ErrorModule.MODEL, ErrorType.TYPE, 2, "invalid prompt template type")
    PARSE_TEMPLATE_INVALID_MODE = BaseErrorCode("MODEL-TYPE-003", ErrorModule.MODEL, ErrorType.TYPE, 3, "invalid mode in prompt template")
    INVALID_TYPE_OF_PARAM_IN_META_TEMPLATE = BaseErrorCode("MODEL-TYPE-004", ErrorModule.MODEL, ErrorType.TYPE, 4, "invalid type of param in meta template")

    GET_SERVICE_MODEL_PATH_FAILED = BaseErrorCode("MODEL-DATA-001", ErrorModule.MODEL, ErrorType.DATA, 1, "fail to get service model path")
    INVALID_PROMPT_CONTENT = BaseErrorCode("MODEL-DATA-002", ErrorModule.MODEL, ErrorType.DATA, 2, "invalid prompt content")
    PARSE_TEXT_RSP_INVALID_FORMAT = BaseErrorCode("MODEL-DATA-003", ErrorModule.MODEL, ErrorType.DATA, 3, "parse text response invalid format")

    MAX_SEQ_LEN_NOT_FOUND = BaseErrorCode("MODEL-CFG-001", ErrorModule.MODEL, ErrorType.CONFIG, 1, "max_seq_len is not provided and cannot be inferred from the model config.")
    MODULE_NOT_FOUND = BaseErrorCode("MODEL-MOD-001", ErrorModule.MODEL, ErrorType.MODULE, 1, "module not found")



class UNK_CODES:
    UNKNOWN_ERROR = BaseErrorCode("UNK-UNK-001", ErrorModule.UNKNOWN, ErrorType.UNKNOWN, 1, "unknown error")


class UTILS_CODES:
    UNKNOWN_ERROR = BaseErrorCode("UTILS-UNK-001", ErrorModule.UTILS, ErrorType.UNKNOWN, 1, "unknown error of utils")
    MATCH_CONFIG_FILE_FAILED = BaseErrorCode("UTILS-MATCH-001", ErrorModule.UTILS, ErrorType.MATCH, 1, "match config file failed")
    DEPENDENCY_MODULE_IMPORT_ERROR = BaseErrorCode("UTILS-DEPENDENCY-001", ErrorModule.UTILS, ErrorType.DEPENDENCY, 1, "third party dependency module import error")
    MODEL_CONFIG_VALIDATE_FAILED = BaseErrorCode("UTILS-CFG-002", ErrorModule.UTILS, ErrorType.CONFIG, 2, "model config validate failed")
    ILLEGAL_MODEL_ATTR = BaseErrorCode("UTILS-CFG-003", ErrorModule.UTILS, ErrorType.CONFIG, 3, "illegal model attr in config")
    MIXED_MODEL_ATTRS = BaseErrorCode("UTILS-CFG-004", ErrorModule.UTILS, ErrorType.CONFIG, 4, "mixed model attrs in config")
    NON_FUNCTION_CALL_MODEL = BaseErrorCode("UTILS-CFG-005", ErrorModule.UTILS, ErrorType.CONFIG, 5, "non function call model found for BFCLDataset")
    NON_BFCL_DATASET = BaseErrorCode("UTILS-CFG-006", ErrorModule.UTILS, ErrorType.CONFIG, 6, "non BFCL dataset found for VLLMFunctionCallAPIChat")
    INCOMPATIBLE_MERGE_DS = BaseErrorCode("UTILS-CFG-007", ErrorModule.UTILS, ErrorType.CONFIG, 7, "incompatible --merge-ds option for function call task")
    SYNTHETIC_DS_MISS_REQUIRED_PARAM = BaseErrorCode("UTILS-CFG-001", ErrorModule.UTILS, ErrorType.CONFIG, 1, "synthetic dataset miss required param")
    MM_CUSTOM_DATASET_WRONG_FORMAT = BaseErrorCode("UTILS-CFG-008", ErrorModule.UTILS, ErrorType.CONFIG, 8, "invalid mm custom dataset")

class CALC_CODES:
    UNKNOWN_ERROR = BaseErrorCode("CALC-UNK-001", ErrorModule.CALCULATOR, ErrorType.UNKNOWN, 1, "unknown error of calculator")
    INVALID_METRIC_DATA = BaseErrorCode("CALC-MTRC-001", ErrorModule.CALCULATOR, ErrorType.METRIC, 1, "invalid content of metric data")
    DUMPING_RESULT_FAILED = BaseErrorCode("CALC-FILE-001", ErrorModule.CALCULATOR, ErrorType.FILE, 1, "fail to dump result to file")
    ALL_REQUEST_DATAS_INVALID = BaseErrorCode("CALC-DATA-001", ErrorModule.CALCULATOR, ErrorType.DATA, 1, "all request datas are invalid")
    CAN_NOT_FIND_STABLE_STAGE = BaseErrorCode("CALC-DATA-002", ErrorModule.CALCULATOR, ErrorType.DATA, 2, "invalid response datas")
    OUTPUT_HANDLER_INVALID_OUTPUT = BaseErrorCode("CALC-DATA-003", ErrorModule.CALCULATOR, ErrorType.DATA, 3, "output handler invalid output")

class DATASETS_CODES:
    UNKNOWN_ERROR = BaseErrorCode("DATASETS-UNK-001", ErrorModule.DATASETS, ErrorType.UNKNOWN, 1, "unknown error of datasets")
    INVALID_DATASET_CONFIG = BaseErrorCode("DATASETS-CFG-001", ErrorModule.DATASETS, ErrorType.CONFIG, 1, "invalid dataset config")


class DSET_CODES:
    UNKNOWN_ERROR = BaseErrorCode("DSET-UNK-001", ErrorModule.DATASET, ErrorType.UNKNOWN, 1, "unknown error of dataset")
    
    # File related errors
    FILE_NOT_FOUND = BaseErrorCode("DSET-FILE-001", ErrorModule.DATASET, ErrorType.FILE, 1, "dataset file not found")
    FILE_READ_ERROR = BaseErrorCode("DSET-FILE-002", ErrorModule.DATASET, ErrorType.FILE, 2, "failed to read dataset file")
    FILE_FORMAT_ERROR = BaseErrorCode("DSET-FILE-003", ErrorModule.DATASET, ErrorType.FILE, 3, "invalid dataset file format")
    
    # Data related errors
    DATA_EMPTY = BaseErrorCode("DSET-DATA-001", ErrorModule.DATASET, ErrorType.DATA, 1, "dataset is empty")
    DATA_INVALID_STRUCTURE = BaseErrorCode("DSET-DATA-002", ErrorModule.DATASET, ErrorType.DATA, 2, "dataset has invalid structure")
    DATA_MISSING_REQUIRED_FIELD = BaseErrorCode("DSET-DATA-003", ErrorModule.DATASET, ErrorType.DATA, 3, "dataset missing required field")
    DATA_LABEL_PARSE_ERROR = BaseErrorCode("DSET-DATA-004", ErrorModule.DATASET, ErrorType.DATA, 4, "failed to parse label")
    DATA_PREPROCESSING_ERROR = BaseErrorCode("DSET-DATA-005", ErrorModule.DATASET, ErrorType.DATA, 5, "data preprocessing or cleaning failed")
    INVALID_DATA_TYPE = BaseErrorCode("DSET-DATA-006", ErrorModule.DATASET, ErrorType.DATA, 6, "data type does not match expected type")
    
    # Parameter related errors
    INVALID_SPLIT_NAME = BaseErrorCode("DSET-PARAM-001", ErrorModule.DATASET, ErrorType.PARAM, 1, "invalid split name")
    INVALID_REPEAT_FACTOR = BaseErrorCode("DSET-PARAM-002", ErrorModule.DATASET, ErrorType.PARAM, 2, "invalid repeat factor")
    INVALID_DATASET_NAME = BaseErrorCode("DSET-PARAM-003", ErrorModule.DATASET, ErrorType.PARAM, 3, "invalid dataset name")
    INVALID_PARAM_VALUE = BaseErrorCode("DSET-PARAM-004", ErrorModule.DATASET, ErrorType.PARAM, 4, "invalid parameter value")
    
    # Dependency related errors
    MODELSCOPE_NOT_INSTALLED = BaseErrorCode("DSET-DEPENDENCY-001", ErrorModule.DATASET, ErrorType.DEPENDENCY, 1, "ModelScope library not installed")
    EVALUATION_LIBRARY_NOT_INSTALLED = BaseErrorCode("DSET-DEPENDENCY-002", ErrorModule.DATASET, ErrorType.DEPENDENCY, 2, "evaluation library not installed")
    
    # Evaluation related errors
    PREDICTION_LENGTH_MISMATCH = BaseErrorCode("DSET-MTRC-001", ErrorModule.DATASET, ErrorType.METRIC, 1, "prediction and reference have different length")
    EVALUATION_FAILED = BaseErrorCode("DSET-MTRC-002", ErrorModule.DATASET, ErrorType.METRIC, 2, "evaluation failed")
    INVALID_MBPP_METRIC = BaseErrorCode("DSET-MTRC-003", ErrorModule.DATASET, ErrorType.METRIC, 3, "invalid MBPP metric type")


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
    DATASETS_CODES,
    DSET_CODES,
]

# init error code manager
error_manager = ErrorCodeManager()

# regist all errors
for error_codes_class in ERROR_CODES_CLASSES:
    for error_code in error_codes_class.__dict__.values():
        if isinstance(error_code, BaseErrorCode):
            error_manager.register(error_code)
