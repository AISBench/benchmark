from typing import Any, Dict, List, Union, TypeVar, Type
from decimal import Decimal
from datasets import Dataset, DatasetDict

from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.logging.exceptions import (
    AISBenchConfigError,
    AISBenchInvalidTypeException,
)
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES

logger = AISLogger()


def check_type_list(obj, typelist: List):
    """Check if object matches any type in the type list.

    Args:
        obj: Object to check
        typelist: List of acceptable types

    Returns:
        The object if type check passes

    Raises:
        AISBenchInvalidTypeException: If object doesn't match any type in list
    """
    for _type in typelist:
        if _type is None:
            if obj is None:
                return obj
        elif isinstance(obj, _type):
            return obj

    expected_types = [ty.__name__ if ty is not None else None for ty in typelist]
    actual_type = type(obj).__name__
    raise AISBenchInvalidTypeException(
        UTILS_CODES.INVALID_TYPE,
        f"Expected an object in {expected_types} type, but got {actual_type}: {obj}"
    )


def check_dataset(obj) -> Union[Dataset, DatasetDict]:
    """Check if object is a valid HuggingFace Dataset.

    Args:
        obj: Object to check

    Returns:
        The dataset object if valid

    Raises:
        AISBenchInvalidTypeException: If object is not a Dataset or DatasetDict
    """
    if isinstance(obj, (Dataset, DatasetDict)):
        return obj

    raise AISBenchInvalidTypeException(
        UTILS_CODES.INVALID_DATASET_TYPE,
        f"Expected a datasets.Dataset or a datasets.DatasetDict object, but got {type(obj).__name__}: {obj}"
    )


def check_list(obj) -> list:
    """Check if object is a list.

    Args:
        obj: Object to check

    Returns:
        The list object if valid

    Raises:
        AISBenchInvalidTypeException: If object is not a list
    """
    if isinstance(obj, list):
        return obj
    raise AISBenchInvalidTypeException(
        UTILS_CODES.INVALID_LIST_TYPE,
        f"Expected a List object, but got {type(obj).__name__}: {obj}"
    )


def check_str(obj) -> str:
    """Check if object is a string.

    Args:
        obj: Object to check

    Returns:
        The string object if valid

    Raises:
        AISBenchInvalidTypeException: If object is not a string
    """
    if isinstance(obj, str):
        return obj

    raise AISBenchInvalidTypeException(
        UTILS_CODES.INVALID_STRING_TYPE,
        f"Expected a str object, but got {type(obj).__name__}: {obj}"
    )


def check_dict(obj) -> Dict:
    """Check if object is a dictionary.

    Args:
        obj: Object to check

    Returns:
        The dict object if valid

    Raises:
        AISBenchInvalidTypeException: If object is not a dictionary
    """
    if isinstance(obj, Dict):
        return obj
    raise AISBenchInvalidTypeException(
        UTILS_CODES.INVALID_DICT_TYPE,
        f"Expected a Dict object, but got {type(obj).__name__}: {obj}"
    )


_T = TypeVar("_T")


def check_type(obj: Any, expected_type: Type[_T]) -> _T:
    """Generic type checker with detailed error messages.

    Args:
        obj: Object to check
        expected_type: Expected type class

    Returns:
        The object cast to expected type if valid

    Raises:
        AISBenchInvalidTypeException: If object is not of expected type
    """
    if not isinstance(expected_type, type):
        raise AISBenchInvalidTypeException(
            UTILS_CODES.INVALID_TYPE_SPECIFIER,
            f"Invalid type specifier: {repr(expected_type)}"
        )

    if obj is None:
        if expected_type is type(None):
            return obj
        raise AISBenchInvalidTypeException(
            UTILS_CODES.INVALID_TYPE,
            f"Expected {expected_type.__name__}, got None"
        )

    if isinstance(obj, expected_type):
        return obj

    actual_type = type(obj).__name__
    obj_repr = repr(obj)[:100] + ('...' if len(repr(obj)) > 100 else '')
    raise AISBenchInvalidTypeException(
        UTILS_CODES.TYPE_MISMATCH,
        f"Expected {expected_type.__name__}, got {actual_type}: {obj_repr}"
    )


def _check_positive_int_value(obj) -> bool:
    """Check if object can be converted to a positive integer.

    Args:
        obj: Object to check

    Returns:
        True if object is a positive integer, False otherwise
    """
    try:
        return int(obj) > 0
    except (ValueError, TypeError):
        logger.warning(f"Value {obj} cannot be converted to positive int.")
        return False


def _check_percentage_float(obj) -> bool:
    """Check if object is a valid percentage (0 < value <= 1).

    Args:
        obj: Object to check

    Returns:
        True if object is valid percentage, False otherwise
    """
    try:
        return 0 < float(obj) <= 1
    except (ValueError, TypeError):
        logger.warning(f"Value {obj} is not a valid percentage float.")
        return False



def check_meta_json_dict(obj) -> Dict:
    """Validate meta JSON dictionary structure and types.

    Args:
        obj: Dictionary to validate

    Returns:
        The validated dictionary

    Raises:
        AISBenchInvalidTypeException: If validation fails
    """
    logger.debug("Validating meta JSON dict structure")

    VALID_KEY_VALUE_TYPES = {
        "output_config": {
            "method": str,
            "params": {
                "min_value": Union[int, str],
                "max_value": Union[int, str],
                "percentage_distribute": list,
            },
        },
        "request_count": Union[int, str],
        "sampling_mode": str,
    }

    def validate_recursive(data, valid_key_value_types):
        data = check_dict(data)
        extra_keys = set(data.keys()) - set(valid_key_value_types.keys())
        if extra_keys:
            raise AISBenchConfigError(
                UTILS_CODES.ILLEGAL_KEYS_IN_CONFIG,
                f"There are illegal keys: {', '.join(extra_keys)}"
            )

        for key, value in data.items():
            expected_type = valid_key_value_types[key]

            if isinstance(expected_type, dict):
                # deal with nested condition
                validate_recursive(value, expected_type)
            elif (
                hasattr(expected_type, "__origin__")
                and expected_type.__origin__ is Union
            ):
                if not isinstance(value, expected_type.__args__):
                    raise AISBenchInvalidTypeException(
                        UTILS_CODES.TYPE_MISMATCH,
                        f"Expected type: {expected_type}, but got {type(value)}"
                    )
            elif not isinstance(value, expected_type):
                raise AISBenchInvalidTypeException(
                    UTILS_CODES.TYPE_MISMATCH,
                    f"Expected type: {expected_type}, but got {type(value)}"
                )
            else:
                continue

    validate_recursive(obj, VALID_KEY_VALUE_TYPES)

    if "request_count" in obj:
        if not _check_positive_int_value(obj["request_count"]):
            raise AISBenchConfigError(
                UTILS_CODES.INVALID_REQUEST_COUNT,
                "Please make sure that the value of parameter 'request_count' can be converted to int(greater than 0)."
            )

    logger.debug("Meta JSON dict validation passed")
    return obj


def _check_percentage_distribute(obj) -> bool:
    """Check if percentage distribution list is valid.

    Args:
        obj: List to check

    Returns:
        True if valid distribution, False otherwise
    """
    if not isinstance(obj, list):
        return False

    percentage_sum = Decimal("0.0")

    for i in obj:
        if not isinstance(i, list) or len(i) != 2:
            return False
        if not (_check_positive_int_value(i[0]) and _check_percentage_float(i[1])):
            return False
        percentage_sum += Decimal(str(i[1]))
    return percentage_sum == Decimal("1.0")

def check_output_config_from_meta_json(obj) -> bool:
    """Validate output_config section from meta JSON.

    Args:
        obj: Dictionary containing output_config

    Returns:
        True if validation passes

    Raises:
        ConfigError: If validation fails
    """
    if obj == {} or "output_config" not in obj:
        logger.debug("No output_config found in meta JSON")
        return False

    logger.debug("Validating output_config from meta JSON")
    output_config = obj["output_config"]
    method = output_config.get("method", None)
    param = output_config.get("params", None)

    if not param:
        raise AISBenchConfigError(
            UTILS_CODES.MISSING_PARAMS,
            "Make sure to set the 'params' parameter in the 'output_config'."
        )

    if method == "uniform":
        if "min_value" in param and "max_value" in param:
            if (not _check_positive_int_value(param["min_value"])) or (
                not _check_positive_int_value(param["max_value"])
            ):
                raise AISBenchConfigError(
                    UTILS_CODES.INVALID_MIN_MAX_VALUE,
                    "Please make sure that the value of parameter 'min_value' and 'max_value' can be converted to int(greater than 0)."
                )
            if int(param["min_value"]) > int(param["max_value"]):
                raise AISBenchConfigError(
                    UTILS_CODES.MIN_GREATER_THAN_MAX,
                    "When the uniform distribution is set, parameter 'min_value' must be less than or equal to parameter 'max_value'."
                )
            logger.debug(f"Uniform distribution validated: min={param['min_value']}, max={param['max_value']}")
            return True
        raise AISBenchConfigError(
            UTILS_CODES.MISSING_PARAMS,
            "When the uniform distribution is set, parameter 'min_value' and 'max_value' must be provided."
        )
    if method == "percentage":
        if "percentage_distribute" not in param:
            raise AISBenchConfigError(
                UTILS_CODES.MISSING_PARAMS,
                "When the percentage distribution is set, parameter 'percentage_distribute' must be provided."
            )
        if not _check_percentage_distribute(param["percentage_distribute"]):
            logger.warning(f"Invalid percentage_distribute format: {param['percentage_distribute']}")
            err_msg = """
            Ensure the configuration data follows the format [max_tokens, percentage], where:
            - 'max_tokens' must be a positive number (greater than 0).
            - 'percentage' must be a float between 0 and 1 (greater than 0 and inclusive 1).
            - The sum of all 'percentage' values must equal exactly 1.
            Example valid format: [[1000, 0.5],[500,0.5]] or [[2000, 1.0]]
            Example invalid formats: [[0, 0.5]] (max_tokens <= 0), [[1000, 1.5]] (percentage > 1), [[1000, 0.3], [500,0.2]] (sum not 1)
            """
            raise AISBenchConfigError(UTILS_CODES.INVALID_PERCENTAGE_DISTRIBUTE, err_msg)
        logger.debug(f"Percentage distribution validated: {param['percentage_distribute']}")
        return True
    raise AISBenchConfigError(
        UTILS_CODES.UNSUPPORTED_DISTRIBUTION_METHOD,
        f"Type of data distribution(method): {method} is not supported, legal methods chosen from ['uniform', 'percentage']."
    )
