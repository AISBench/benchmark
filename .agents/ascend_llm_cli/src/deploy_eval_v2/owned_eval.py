"""Adapt frozen owned-service cases to the shared evaluation contract.

The evaluation adapter gets client-only authority. Service release remains the
responsibility of the owning executor and its separately frozen failure policy.
"""
from .plan_model import normalize_external_plan,validate_eval_bindings
from .protocol import ProtocolError


def owned_eval_inputs(plan,bindings,service):
    evaluation=plan.get('evaluation')
    if not evaluation or not bindings.get('testers'):
        raise ProtocolError('INVALID_INPUT: owned evaluation plan/bindings missing')
    adapted=dict(schema_version='1.0',task_id=plan['task_id'],intent='service_eval',
        service=dict(service_id=service['service_id'],endpoint=service['endpoint'],model=service['model'],
                     ownership='owned',deployment_summary=service.get('deployment_summary'),
                     permissions=dict(reset_cache=False,restart=False,release=False)),
        cases=evaluation['cases'],test_binding=evaluation['test_binding'],unresolved=plan['unresolved'],
        policy=dict(idle_timeout_seconds=plan['policy']['idle_timeout_seconds'],on_idle='keep_service',
                    on_failure='stop_owned_clients'))
    test_bindings=dict(schema_version='1.0',task_id=plan['task_id'],testers=bindings['testers'])
    if 'metrics' in service:
        adapted['service']['metrics']=service['metrics']
    if any(case['reset_cache'] for case in evaluation['cases']):
        adapted['service']['permissions']['reset_cache']=True
        adapted['service']['reset_targets']=[dict(id=item['id'],metrics_url=item['url'],
            url=item['url'].removesuffix('/metrics')+'/reset_prefix_cache') for item in service.get('metrics',[])]
    normalized,derived=normalize_external_plan(adapted)
    if derived or normalized['unresolved']:
        raise ProtocolError('INVALID_INPUT: normalize owned evaluation before review')
    validate_eval_bindings(adapted,test_bindings)
    return adapted,test_bindings
