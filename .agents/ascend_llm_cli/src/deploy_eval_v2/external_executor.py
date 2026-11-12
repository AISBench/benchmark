"""Frozen external-service executor; no deployment/release authority."""
import hashlib
import os
from pathlib import Path
import traceback
from .compiler import verify_frozen, canonical
from .evaluate import execute_external_case
from .process_guard import Cancelled
from .protocol import ProtocolError
from .result_contract import evaluation_result


def persist(path, value):
    raw = canonical(value)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return hashlib.sha256(raw).hexdigest()


class ExternalExecutor:
    def __init__(self, root, frozen, bundle_digest, runtime_digest, verified_host,
                 *, case_runner=execute_external_case):
        self.root = Path(root)
        self.frozen = Path(frozen)
        self.bundle_digest = bundle_digest
        self.runtime_digest = runtime_digest
        self.verified_host = verified_host
        self.case_runner = case_runner
        values = self.inputs()
        self.task_id = values['plan.json']['task_id']
        self.service_id = values['plan.json']['service']['service_id']
        self.initial_identity = dict(task_id=self.task_id, service_id=self.service_id,
                                     ownership='external', bundle_sha256=bundle_digest)

    def inputs(self):
        values = verify_frozen(self.frozen, self.bundle_digest, self.runtime_digest)
        plan, bindings = values['plan.json'], values['bindings.json']
        if plan.get('intent') != 'external_eval':
            raise ProtocolError('INVALID_INPUT: external evaluator requires external plan')
        if bindings['testers'][plan['test_binding']]['host'] != self.verified_host:
            raise ProtocolError('INVALID_INPUT: tester identity mismatch')
        return values

    def validate(self, command):
        values = self.inputs()
        if command['task_id'] != self.task_id:
            raise ProtocolError('INVALID_INPUT: task identity')
        if command['kind'] == 'cancel':
            return
        if command['kind'] != 'eval' or command['service_id'] != self.service_id:
            raise ProtocolError('INVALID_INPUT: external service is evaluation only')
        case = next((c for c in values['plan.json']['cases'] if c['case_id'] == command['case_id']), None)
        if case is None:
            raise ProtocolError('INVALID_INPUT: unsupported or unknown case')

    def __call__(self, command, operation, cancelled):
        self.validate(command)
        values = self.inputs()
        operation_root = self.root/'results'/operation
        operation_root.mkdir(parents=True, exist_ok=False)
        evidence = operation_root/'stage-result.json'
        data = dict(service_id=self.service_id, case_id=command['case_id'],
                    result_ref=evidence.relative_to(self.root).as_posix())
        try:
            result = self.case_runner(values['plan.json'], values['bindings.json'], command['case_id'],
                                      operation_root/'benchmark', cancelled,
                                      verified_local_host=self.verified_host)
            result=evaluation_result(result,task_id=self.task_id,operation_id=operation,
                service_id=self.service_id,case_id=command['case_id'])
            data['result_sha256'] = persist(evidence, result)
            data.update(execution_status=result['execution_status'],
                        evidence_status=result['evidence_status'],
                        performance_validity=result['performance_validity'])
            return 'evaluation.completed', 'EVALUATED', data
        except Exception as error:
            with (operation_root/'error.log').open('w', encoding='utf-8') as stream:
                traceback.print_exc(file=stream)
            stopped = isinstance(error, Cancelled)
            result = dict(task_id=self.task_id, operation_id=operation,
                          execution_status='CANCELLED' if stopped else 'FAILED',
                          evidence_status='PARTIAL', performance_validity='UNVERIFIED',
                          external_service_action='NONE', cleanup_status='UNVERIFIED',
                          error_ref=(operation_root/'error.log').relative_to(self.root).as_posix())
            result=evaluation_result(result,task_id=self.task_id,operation_id=operation,
                service_id=self.service_id,case_id=command['case_id'])
            data['result_sha256'] = persist(evidence, result)
            data.update(evidence_ref=data['result_ref'], cleanup_status='UNVERIFIED',
                        error_code='CANCELLED' if stopped else 'EVALUATION_FAILED')
            return ('operation.cancelled' if stopped else 'operation.failed',
                    'CANCELLED' if stopped else 'FAILED', data)
