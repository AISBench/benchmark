from typing import Any, Dict, List, Union, TypeVar, Type
from decimal import Decimal
from datasets import Dataset, DatasetDict

from ais_bench.benchmark.utils.logging.logging import get_logger

logger = get_logger()


def check_type_list(obj, typelist: List):
    for _type in typelist:
        if _type is None:
            if obj is None:
                return obj
        elif isinstance(obj, _type):
            return obj
    raise TypeError(
        f"Expected an object in {[_.__name__ if _ is not None else None for _ in typelist]} type, but got {obj}"  # noqa
    )


def check_dataset(obj) -> Union[Dataset, DatasetDict]:
    if isinstance(obj, Dataset) or isinstance(obj, DatasetDict):
        return obj
    else:
        raise TypeError(
            f"Expected a datasets.Dataset or a datasets.DatasetDict object, but got {obj}"  # noqa
        )


def check_list(obj) -> List:
    if isinstance(obj, List):
        return obj
    else:
        raise TypeError(f"Expected a List object, but got {obj}")


def check_str(obj) -> str:
    if isinstance(obj, str):
        return obj
    else:
        raise TypeError(f"Expected a str object, but got {obj}")


def check_dict(obj) -> Dict:
    if isinstance(obj, Dict):
        return obj
    else:
        raise TypeError(f"Expected a Dict object, but got {obj}")


_T = TypeVar("_T")


def check_type(obj: Any, expected_type: Type[_T]) -> _T:
    if not isinstance(expected_type, type):
        raise ValueError(f"Invalid type specifier: {repr(expected_type)}")

    if obj is None:
        if expected_type is type(None):
            return obj
        raise TypeError(f"Expected {expected_type.__name__}, got None")

    if isinstance(obj, expected_type):
        return obj

    actual_type = type(obj).__name__
    raise TypeError(
        f"Expected {expected_type.__name__}, got {actual_type}: {repr(obj)[:100]}{'...' if len(repr(obj)) > 100 else ''}"
    )


def _check_positive_int_value(obj) -> bool:
    if isinstance(obj, str) and not obj.isdigit():
        return False
    if int(obj) > 0:
        return True
    else:
        return False


def _check_percentage_float(obj) -> bool:
    if isinstance(obj, str):
        try:
            float(obj)
        except ValueError:
            return False
    elif float(obj) > 0 and float(obj) <= 1:
        return True
    else:
        return False


def check_meta_json_dict(obj) -> Dict:
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
            raise ValueError(f"There are illegal keys: {', '.join(extra_keys)}")
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
                    raise TypeError(
                        f"Expected type: {expected_type}, but got {type(value)}"
                    )
            elif not isinstance(value, expected_type):
                raise TypeError(
                    f"Expected type: {expected_type}, but got {type(value)}"
                )
            else:
                continue

    validate_recursive(obj, VALID_KEY_VALUE_TYPES)
    if "request_count" in obj:
        if not _check_positive_int_value(obj["request_count"]):
            raise ValueError(
                "Please make sure that the value of parameter 'request_count' can be converted to int(greater than 0)."
            )
    return obj


def _check_percentage_distribute(obj) -> bool:
    if not isinstance(obj, list):
        return False
    is_shape_valid = True
    percentage_sum = Decimal("0.0")
    is_value_valid = True
    for i in obj:
        if not isinstance(i, list) or len(i) != 2:
            is_shape_valid = False
            break
        if (not _check_positive_int_value(i[0])) or (not _check_percentage_float(i[1])):
            is_value_valid = False
        percentage_sum += Decimal(str(i[1]))
    if is_shape_valid and is_value_valid and percentage_sum == 1:
        return True
    return False


def check_output_config_from_meta_json(obj) -> bool:
    if obj == {} or "output_config" not in obj:
        return False
    output_config = obj["output_config"]
    method = output_config.get("method", None)
    param = output_config.get("params", None)
    if not param:
        raise ValueError(
            "Make sure to set the 'params' parameter in the 'output_config'."
        )
    if method == "uniform":
        if "min_value" in param and "max_value" in param:
            if (not _check_positive_int_value(param["min_value"])) or (
                not _check_positive_int_value(param["max_value"])
            ):
                raise ValueError(
                    "Please make sure that the value of parameter 'min_value' and 'max_value' can be converted to int(greater than 0)."
                )
            if int(param["min_value"]) > int(param["max_value"]):
                raise ValueError(
                    "When the uniform distribution is set, parameter 'min_value' must be less than or equal to parameter 'max_value'."
                )
            return True
        else:
            raise ValueError(
                "When the uniform distribution is set, parameter 'min_value' and 'max_value' must be provided."
            )
    elif method == "percentage":
        if "percentage_distribute" not in param:
            raise ValueError(
                "When the percentage distribution is set, parameter 'percentage_distribute' must be provided."
            )
        if not _check_percentage_distribute(param["percentage_distribute"]):
            err_msg = """
            Ensure the configuration data follows the format [max_tokens, percentage], where:
            - 'max_tokens' must be a positive number (greater than 0).
            - 'percentage' must be a float between 0 and 1 (greater than 0 and inclusive 1).
            - The sum of all 'percentage' values must equal exactly 1.
            Example valid format: [[1000, 0.5],[500,0.5]] or [[2000, 1.0]]
            Example invalid formats: [[0, 0.5]] (max_tokens <= 0), [[1000, 1.5]] (percentage > 1), [[1000, 0.3], [500,0.2]] (sum not 1)
            """
            raise ValueError(err_msg)
        return True
    else:
        raise ValueError(f"Type of data distribution: {method} Not supported.")
