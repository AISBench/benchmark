from typing import Optional
from ais_bench.benchmark.utils.logging.error_codes import error_manager
from ais_bench.benchmark.utils.logging import get_formatted_log_content
from ais_bench.benchmark.utils.logging.error_codes import BaseErrorCode


class AISBenchBaseException(Exception):
    def __init__(self, error_code: BaseErrorCode, message: Optional[str] = None):
        """
        Args:
            error_str (str): full code of error code
            message (Optional[str], optional): error message. Defaults to None.

        """
        if not isinstance(error_code, BaseErrorCode):
            raise ValueError(f"error_code {error_code} is not instance of BaseErrorCode!")
        if not error_manager.get(error_code.full_code):
            raise ValueError(f"error_code {error_code.full_code} is not exist!")

        self.error_code_str = error_code.full_code

        super().__init__(get_formatted_log_content(error_code.full_code, message))


class CommandError(AISBenchBaseException):
    pass


class ConfigError(AISBenchBaseException):
    pass


class AISBenchImportError(AISBenchBaseException):
    pass


class AISBenchValueError(AISBenchBaseException):
    pass


class AISBenchImplementationError(AISBenchBaseException):
    pass


class FileMatchError(AISBenchBaseException):
    pass


class FileOperationError(AISBenchBaseException):
    pass


class ParameterValueError(AISBenchBaseException):
    pass


class AISBenchMetricError(AISBenchBaseException):
    pass


class AISBenchDumpError(AISBenchBaseException):
    pass


class AISBenchDataContentError(AISBenchBaseException):
    pass


class PerfResultCalcException(AISBenchBaseException):
    pass


class AISBenchNotImplementedError(AISBenchBaseException):
    pass


class AISBenchKeyError(AISBenchBaseException):
    pass


class AISBenchTypeError(AISBenchBaseException):
    pass


class AISBenchRuntimeError(AISBenchBaseException):
    pass


class AISBenchModuleNotFoundError(AISBenchBaseException):
    pass


class PredictionInvalidException(AISBenchBaseException):
    pass

class AISBenchInvalidTypeException(AISBenchBaseException):
    pass
