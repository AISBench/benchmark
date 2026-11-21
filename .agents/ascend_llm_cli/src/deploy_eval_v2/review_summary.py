"""Deterministic review material, bound to full inputs without executing them."""
import copy
from .hashes import digest
from .container_baseline import DEFAULT_CONTAINER,DEVICES


def summarize(plan,bindings):
    result=dict(schema_version='1.0',task_id=plan.get('task_id'),intent=plan.get('intent'),
                plan_sha256=digest(plan),bindings_sha256=digest(bindings),
                policy=copy.deepcopy(plan.get('policy')),execution=copy.deepcopy(plan.get('execution')),
                unresolved=copy.deepcopy(plan.get('unresolved',[])),site_bindings=copy.deepcopy(bindings))
    if plan.get('intent')=='deploy':
        deployment=plan.get('deployment',{})
        result['deployment']=copy.deepcopy(deployment)
        runtime=deployment.get('runtime',{})
        container=runtime.get('container',{})
        if isinstance(container,dict):
            result['effective_container']={**DEFAULT_CONTAINER,'devices':list(DEVICES),**copy.deepcopy(container)}
            result['container_defaults']=[key for key in (*DEFAULT_CONTAINER,'devices') if key not in container]
        result['evaluation']=copy.deepcopy(plan.get('evaluation'))
        result['authority']=dict(deployment='owned',failure_cleanup=plan.get('policy',{}).get('on_failure'))
    elif plan.get('intent')=='external_eval':
        result.update(service=copy.deepcopy(plan.get('service')),cases=copy.deepcopy(plan.get('cases')),
                      test_binding=plan.get('test_binding'),authority='external service permissions only')
    else:
        result['scope']='non-production input integrity only'
    return result
