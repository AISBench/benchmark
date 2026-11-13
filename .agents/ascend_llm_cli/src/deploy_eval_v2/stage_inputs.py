"""Independent business inputs; preparation never approves or contacts a host."""
import copy
from pathlib import Path
from .durable import atomic_json
from .protocol import ProtocolError, envelope
from .compiler import digest
from .plan_model import normalize_external_plan, validate_eval_bindings


def prepare(request):
    if not isinstance(request,dict) or request.get('stage') not in ('deploy','eval'):
        raise ProtocolError('INVALID_INPUT: stage must be deploy or eval')
    if not isinstance(request.get('destination'),str) or not request['destination'].strip():
        raise ProtocolError('INVALID_INPUT: destination must be a nonempty path')
    destination=Path(request['destination'])
    if destination.exists():
        raise ProtocolError('REQUEST_CONFLICT: preparation destination exists')
    if request['stage']=='deploy':
        allowed={'stage','destination','deployment','bindings'}
        if set(request)!=allowed:
            raise ProtocolError('INVALID_INPUT: deployment input fields')
        plan=copy.deepcopy(request['deployment']);bindings=copy.deepcopy(request['bindings'])
        if not isinstance(plan,dict) or not isinstance(bindings,dict):
            raise ProtocolError('INVALID_INPUT: deployment and bindings must be objects')
        if 'evaluation' in plan or 'testers' in bindings:
            raise ProtocolError('INVALID_INPUT: deployment-only input cannot embed evaluation')
        if not isinstance(plan.get('policy'),dict) or plan['policy'].get('on_idle')!='keep_service':
            raise ProtocolError('INVALID_INPUT: deployment-only service must be retained')
        from .deployment_model import validate_deploy_plan
        from .deployment_compiler import compile_bound_deployment
        validate_deploy_plan(plan)
        compile_bound_deployment(plan,bindings)
        derived=[]
    else:
        allowed={'stage','destination','task_id','service','endpoint','model','cases','tester','policy','health_path'}
        if set(request)-allowed or ('service' in request)==('endpoint' in request):
            raise ProtocolError('INVALID_INPUT: provide service or endpoint, exclusively')
        if 'service' in request and 'model' in request:
            raise ProtocolError('INVALID_INPUT: model must come from supplied service')
        if 'service' in request:
            supplied=copy.deepcopy(request['service'])
            if not isinstance(supplied,dict):
                raise ProtocolError('INVALID_INPUT: service must be an object')
            if supplied.get('ownership')=='owned':
                from .service_contract import validate_service
                validate_service(supplied)
            endpoint,model=supplied['endpoint'],supplied['model']
            service_id=supplied.get('service_id','svc-'+digest([endpoint,model])[:32])
        else:
            endpoint,model=request['endpoint'],request['model']
            service_id='svc-'+digest([endpoint,model])[:32]
        # A new evaluation task never inherits the producer's release authority.
        service=dict(service_id=service_id,endpoint=endpoint,model=model,ownership='external',
                     deployment_summary=None,permissions=dict(reset_cache=False,restart=False,release=False))
        service['health_path'] = supplied.get('health_path', '/health') if 'service' in request else request.get('health_path', '/health')
        plan=dict(schema_version='1.0',task_id=request['task_id'],intent='external_eval',service=service,
                  cases=copy.deepcopy(request['cases']),test_binding='tester',unresolved=[],
                  policy=copy.deepcopy(request.get('policy',dict(idle_timeout_seconds=900,
                      on_idle='keep_external_service',on_failure='stop_owned_clients'))))
        plan,derived=normalize_external_plan(plan)
        bindings=dict(schema_version='1.0',task_id=request['task_id'],testers={'tester':copy.deepcopy(request['tester'])})
        validate_eval_bindings(plan,bindings)
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    from .json_io import load_json
    schemas={p.name:load_json(p) for p in (Path(__file__).parent/'schemas').glob('*.json')}
    registry=Registry().with_resources((name,Resource.from_contents(value)) for name,value in schemas.items())
    validator=Draft202012Validator(schemas['stage-input.schema.json'],registry=registry)
    if next(validator.iter_errors(request),None) is not None:
        raise ProtocolError('INVALID_INPUT: stage input schema')
    destination.mkdir(parents=True,exist_ok=False)
    atomic_json(destination/'plan.json',plan)
    atomic_json(destination/'bindings.json',bindings)
    response=envelope(plan['task_id'],0)
    response.update(plan=str(destination/'plan.json'),bindings=str(destination/'bindings.json'),
                    derived=derived,review_required=True,remote_checks='NOT_RUN')
    return response


def export_contracts(request):
    import shutil
    if not isinstance(request,dict) or set(request)!={'destination'}:
        raise ProtocolError('INVALID_INPUT: contract export request')
    if not isinstance(request.get('destination'),str) or not request['destination'].strip():
        raise ProtocolError('INVALID_INPUT: destination must be a nonempty path')
    destination=Path(request['destination'])
    if destination.exists():
        raise ProtocolError('REQUEST_CONFLICT: contract destination exists')
    shutil.copytree(Path(__file__).parent/'schemas',destination)
    response=envelope(None,0)
    response.update(directory=str(destination),schemas=sorted(p.name for p in destination.glob('*.json')))
    return response
