"""Owned service handoff validation; external plans retain their own contract."""
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit
from jsonschema import Draft202012Validator,FormatChecker
from .json_io import load_json
from .protocol import ProtocolError


@lru_cache(maxsize=1)
def schema_validator():
    return Draft202012Validator(load_json(Path(__file__).parent/'schemas/service.schema.json'),format_checker=FormatChecker())


def validate_service(service):
    if next(schema_validator().iter_errors(service),None) is not None:
        raise ProtocolError('RECOVERY_REQUIRED: owned service contract')
    try:
        endpoint=urlsplit(service['endpoint'])
        if (endpoint.scheme not in ('http','https') or not endpoint.hostname or endpoint.username is not None or
                endpoint.password is not None or endpoint.query or endpoint.fragment or endpoint.port==0 or
                any(ord(c)<=32 for c in service['endpoint'])):
            raise ValueError('endpoint')
        reference=service['health_evidence_ref']
        if (any(part in ('','.','..') for part in reference.split('/')) or '\\' in reference or
                ':' in reference or any(ord(c)<32 for c in reference)):
            raise ValueError('evidence reference')
        containers=service['containers']
        if len(containers)!=service['deployment_summary']['container_groups']:
            raise ValueError('group count')
        if len({item['container_id'] for item in containers.values()})!=len(containers):
            raise ValueError('duplicate container identity')
    except ValueError as error:
        raise ProtocolError('RECOVERY_REQUIRED: inconsistent service evidence') from error
