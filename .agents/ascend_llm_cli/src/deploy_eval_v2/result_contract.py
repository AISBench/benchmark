"""Evaluation result schema and coordinator-bound identity validation."""
from functools import lru_cache
from pathlib import Path
from jsonschema import Draft202012Validator,FormatChecker
from .json_io import load_json
from .protocol import ProtocolError,now


@lru_cache(maxsize=1)
def validator():
    return Draft202012Validator(load_json(Path(__file__).parent/'schemas/result.schema.json'),format_checker=FormatChecker())


def evaluation_result(result,*,task_id,operation_id,service_id,case_id):
    if not isinstance(result,dict):
        raise ProtocolError('INVALID_INPUT: evaluation result object')
    value=dict(result)
    identity=dict(schema_version='1.0',task_id=task_id,operation_id=operation_id,service_id=service_id,case_id=case_id)
    for key,expected in identity.items():
        if key in value and value[key]!=expected:
            raise ProtocolError('RECOVERY_REQUIRED: evaluation result identity mismatch')
        value[key]=expected
    value.setdefault('time',now())
    if next(validator().iter_errors(value),None) is not None:
        raise ProtocolError('INVALID_INPUT: evaluation result schema')
    if value['performance_validity']=='VERIFIED' and value['evidence_status']!='COMPLETE':
        raise ProtocolError('INVALID_INPUT: verified performance requires complete evidence')
    return value
