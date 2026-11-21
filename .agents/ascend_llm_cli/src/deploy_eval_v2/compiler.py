"""Content-addressed review binding; no SSH or execution side effects.

Reviewed deployment inputs also freeze deterministic bound topology and scripts.
Production executors must still attest the remote environment before execution.
"""
import json
import os
from pathlib import Path
import tempfile
from jsonschema import Draft202012Validator
from .json_io import load_json
from .protocol import ProtocolError
from .hashes import canonical,digest
from .durable import rename_new_directory
from .review_summary import summarize


def review_inputs(plan, bindings, runtime_digest):
    if not isinstance(plan, dict) or not isinstance(bindings, dict):
        raise ProtocolError('INVALID_INPUT')
    if not isinstance(runtime_digest, str) or len(runtime_digest) != 64 or any(c not in '0123456789abcdef' for c in runtime_digest):
        raise ProtocolError('INVALID_INPUT: runtime digest')
    return {'plan_sha256': digest(plan), 'bindings_sha256': digest(bindings),
            'runtime_sha256': runtime_digest,'summary_sha256':digest(summarize(plan,bindings))}


def verify_review(plan, bindings, review, runtime_digest):
    if isinstance(plan, dict) and plan.get('intent') == 'deploy':
        from .deployment_model import validate_deploy_plan
        validate_deploy_plan(plan)
        if plan['unresolved']:
            raise ProtocolError('INVALID_INPUT: resolve deployment before review')
        from .deployment_compiler import compile_bound_deployment
        topology = compile_bound_deployment(plan, bindings)
        if 'evaluation' in plan:
            from .owned_eval import owned_eval_inputs
            owned_eval_inputs(plan,bindings,dict(service_id='reviewed-service',endpoint=topology['endpoint'],
                                                model=plan['deployment']['model']['served_name'],
                                                metrics=topology['metric_instances']))
    if isinstance(plan, dict) and plan.get('intent') == 'external_eval':
        from .plan_model import normalize_external_plan, validate_eval_bindings
        normalized, derived = normalize_external_plan(plan)
        if derived or normalized['unresolved']:
            raise ProtocolError('INVALID_INPUT: normalize and resolve plan before review')
        validate_eval_bindings(plan, bindings)
    schema = load_json(Path(__file__).parent/'schemas/review.schema.json')
    if next(Draft202012Validator(schema).iter_errors(review), None) is not None:
        raise ProtocolError('INVALID_INPUT: review schema')
    expected = review_inputs(plan, bindings, runtime_digest)
    if not isinstance(review, dict) or set(review) != {'schema_version', 'status', 'inputs', 'unresolved', 'evidence_refs'}:
        raise ProtocolError('INVALID_INPUT: review structure')
    if review['schema_version'] != '1.0' or review['status'] != 'APPROVED' or review['unresolved'] != []:
        raise ProtocolError('INVALID_INPUT: review not approved')
    if review['inputs'] != expected:
        raise ProtocolError('INVALID_INPUT: stale review')
    refs = review['evidence_refs']
    if not isinstance(refs, list) or not refs or any(not isinstance(r, str) or not r or len(r) > 512 for r in refs):
        raise ProtocolError('INVALID_INPUT: review evidence required')
    return expected


def freeze_inputs(destination, plan, bindings, review, runtime_digest):
    verify_review(plan, bindings, review, runtime_digest)
    destination = Path(destination)
    if destination.exists():
        raise ProtocolError('REQUEST_CONFLICT: input directory exists')
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = {'plan.json': plan, 'bindings.json': bindings, 'review.json': review,
              'review-summary.json':summarize(plan,bindings)}
    if plan.get('intent') == 'deploy':
        values['compiled.json'] = compiled_deployment(plan, bindings)
    if any(len(canonical(value)) > 4*1024*1024 for value in values.values()):
        raise ProtocolError('INVALID_INPUT: input too large')
    manifest = {'schema_version': '1.0', 'runtime_sha256': runtime_digest,
                'files': {name: digest(value) for name, value in values.items()}}
    # An exclusive directory claim avoids overwriting a racing freeze operation.
    claim = destination.parent/('.'+destination.name+'.freeze.lock')
    try:
        descriptor = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ProtocolError('RESOURCE_BUSY') from exc
    try:
        os.close(descriptor)
        with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.freeze-') as temporary:
            staging = Path(temporary)/'inputs'
            staging.mkdir()
            for name, value in {**values, 'manifest.json': manifest}.items():
                with (staging/name).open('xb') as stream:
                    stream.write(canonical(value))
                    stream.flush()
                    os.fsync(stream.fileno())
            if destination.exists():
                raise ProtocolError('REQUEST_CONFLICT')
            rename_new_directory(staging,destination)
    finally:
        claim.unlink()
    return {'path': str(destination), 'sha256': digest(manifest)}


def verify_frozen(directory, expected_digest, runtime_digest):
    root = Path(directory)
    def read(name):
        path = root/name
        if path.is_symlink():
            raise ProtocolError('INVALID_INPUT: symlink input')
        return load_json(path)
    manifest = read('manifest.json')
    if not isinstance(manifest, dict) or digest(manifest) != expected_digest or manifest.get('schema_version') != '1.0' or manifest.get('runtime_sha256') != runtime_digest:
        raise ProtocolError('INVALID_INPUT: frozen identity mismatch')
    names = set(manifest.get('files', {}))
    if names not in ({'plan.json', 'bindings.json', 'review.json','review-summary.json'},
                     {'plan.json', 'bindings.json', 'review.json','review-summary.json', 'compiled.json'}):
        raise ProtocolError('INVALID_INPUT: manifest files')
    values = {name: read(name) for name in manifest['files']}
    if any(digest(value) != manifest['files'][name] for name, value in values.items()):
        raise ProtocolError('INVALID_INPUT: frozen input changed')
    verify_review(values['plan.json'], values['bindings.json'], values['review.json'], runtime_digest)
    if values['review-summary.json']!=summarize(values['plan.json'],values['bindings.json']):
        raise ProtocolError('INVALID_INPUT: frozen review summary differs from inputs')
    if values['plan.json'].get('intent') == 'deploy':
        if values.get('compiled.json') != compiled_deployment(values['plan.json'], values['bindings.json']):
            raise ProtocolError('INVALID_INPUT: frozen compilation mismatch')
    elif 'compiled.json' in values:
        raise ProtocolError('INVALID_INPUT: unexpected deployment compilation')
    return values


def compiled_deployment(plan, bindings):
    from .deployment_compiler import compile_bound_deployment
    from .deploy_scripts import compile_service_scripts
    # JSON object order is not semantic. Freeze writes canonical key ordering;
    # compile from that same representation so verification produces identical argv.
    topology = compile_bound_deployment(json.loads(canonical(plan)), json.loads(canonical(bindings)))
    return dict(schema_version='1.0', model=plan['deployment']['model']['served_name'],
                topology=topology, scripts=compile_service_scripts(topology))


def local_command(kind, request):
    """Local request files carry all locations and expected identity digests."""
    expected = ({'plan','bindings','destination','runtime_sha256'} if kind=='draft-review' else
                {'plan', 'bindings', 'review', 'destination', 'runtime_sha256'}
                if kind == 'freeze-inputs' else {'directory', 'sha256', 'runtime_sha256'})
    if not isinstance(request, dict) or set(request) != expected or any(not isinstance(v, str) or not 1 <= len(v) <= 512 for v in request.values()):
        raise ProtocolError('INVALID_INPUT: compiler request')
    if kind=='draft-review':
        plan,bindings=load_json(request['plan']),load_json(request['bindings'])
        inputs=review_inputs(plan,bindings,request['runtime_sha256'])
        destination=Path(request['destination'])
        destination.mkdir(parents=True,exist_ok=False)
        from .durable import atomic_json
        atomic_json(destination/'review-summary.json',summarize(plan,bindings))
        atomic_json(destination/'review.json',dict(schema_version='1.0',status='DRAFT',inputs=inputs,
                    unresolved=plan.get('unresolved',[]),evidence_refs=[]))
        return dict(protocol_version='1.0',task_id=plan.get('task_id'),status='EVENTS',events=[],next_cursor=0,
                    has_more=False,review_directory=str(destination),summary_sha256=inputs['summary_sha256'],approved=False)
    elif kind == 'freeze-inputs':
        reference = freeze_inputs(request['destination'], load_json(request['plan']),
                                  load_json(request['bindings']), load_json(request['review']),
                                  request['runtime_sha256'])
    else:
        verify_frozen(request['directory'], request['sha256'], request['runtime_sha256'])
        reference = {'path': request['directory'], 'sha256': request['sha256']}
    return {'protocol_version': '1.0', 'status': 'EVENTS', 'task_id': None,
            'events': [], 'next_cursor': 0, 'has_more': False,
            'input_bundle': reference, 'validation_scope': 'input_integrity_and_review_binding',
            'production_ready': False}
