"""Pure confirmed task model; no lifecycle or remote execution imports."""
import copy,json,re,shlex
from pathlib import Path
PACKAGE=Path(__file__).resolve().parents[1]
TOKEN=re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_.]*)\}\}")

def load_json(path):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result: raise ValueError('Duplicate JSON key: '+key)
            result[key]=value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=pairs)

def normalize(spec):
    s=copy.deepcopy(spec)
    e=s.setdefault('execution',{})
    for k,v in {'smoke':'default','max_retries':3,'startup_timeout_seconds':1800,'dependency_timeout_seconds':120,'smoke_timeout_seconds':240,
                'proxy_start_after_models_healthy':True,'enable_cache_reset_api':True}.items():
        e.setdefault(k,v)
    if s.get('schema_version')=='3.0':
        from container_baseline import DEFAULT_CONTAINER, DEVICES
        for dep in s.get('deployments', []):
            if e['enable_cache_reset_api']:
                for instance in dep.get('instances', []):
                    if instance['role'] != 'proxy':
                        env=instance['launch'].setdefault('env', {})
                        if env.get('VLLM_SERVER_DEV_MODE','1') != '1':
                            raise ValueError('enable_cache_reset_api conflicts with confirmed VLLM_SERVER_DEV_MODE; resolve before execution')
                        env.setdefault('VLLM_SERVER_DEV_MODE','1')
            container = dep['runtime'].setdefault('container', {})
            for key, value in {**DEFAULT_CONTAINER, 'devices': DEVICES}.items():
                container.setdefault(key, copy.deepcopy(value))
        cleanup=s.setdefault('cleanup',{})
        cleanup.setdefault('after_case','keep_service')
        cleanup.setdefault('task_end','stop_container')
        for case in s.get('cases',[]):
            case.setdefault('mode', 'perf')
            if case['mode'] != 'perf':
                continue
            for k,v in {'formal_requests':case['concurrency']*4,'dataset_id':'default','num_warmups':1,
                        'steady_state':False,'request_rate':0,'warmup':True,'reset_cache':False,'timeout_seconds':5400}.items():
                case.setdefault(k,v)
    for subtask in s.get('sub_tasks', []):
        subtask.setdefault('after_case', s['cleanup']['after_case'])
        subtask.setdefault('phase_id', 'pp1')
        subtask.setdefault('on_failure', 'record_only')
    return s

def logging_assignments(bootstrap, keys):
    """Inspect assignment commands, not mentions in comments/echo/grep arguments."""
    for number, line in enumerate(bootstrap.splitlines(), 1):
        try:
            tokens = shlex.split(line, comments=True)
        except ValueError:
            continue  # Full shell syntax belongs to generated script validation.
        if tokens and tokens[0] == 'export':
            tokens = tokens[1:]
        for token in tokens:
            if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*=', token):
                break
            key = token.split('=', 1)[0]
            if key in keys:
                yield key, number


def validate_task(spec):
    from topology import validate_deployment, context_limit
    errors = []
    deployments = {d['id']:d for d in spec['deployments']}
    cases = {c['id']:c for c in spec['cases']}
    if len(deployments) != len(spec['deployments']) or len(cases) != len(spec['cases']):
        errors.append('Duplicate deployment/case ID')
    for dep in deployments.values():
        for instance in dep['instances']:
            launch=instance.get('launch',{})
            if launch.get('preserve_reference'):
                continue
            keys = ('ASCEND_SLOG_PRINT_TO_STDOUT','ASCEND_GLOBAL_LOG_LEVEL')
            for key in keys:
                if key in launch.get('env',{}):
                    errors.append('Remove operator logging assignment: '+key+' in '+instance['id']+'.launch.env')
            for key, line in logging_assignments(launch.get('bootstrap',''), keys):
                errors.append('Remove operator logging assignment: '+key+' in '+instance['id']+'.bootstrap line '+str(line))
        errors.extend(validate_deployment(dep))
    if errors:
        return errors
    seen = set()
    for task in spec['sub_tasks']:
        if task['id'] in seen:
            errors.append('Duplicate sub_task ID')
        seen.add(task['id'])
        dep, case = deployments.get(task['deployment_id']), cases.get(task['case_id'])
        if dep is None or case is None:
            errors.append('Unknown deployment/case reference')
            continue
        limits = [context_limit(i['launch'].get('args',{})) for i in dep['instances'] if i['role'] != 'proxy']
        if case.get('mode','perf')=='perf' and limits and case.get('input_tokens',0)+case['output_tokens'] > min(limits):
            errors.append('Context limit exceeded: '+task['id'])
        if case.get('reset_cache') and any(i['launch'].get('env',{}).get('VLLM_SERVER_DEV_MODE')!='1' for i in dep['instances'] if i['role']!='proxy'):
            errors.append('reset_cache requires confirmed VLLM_SERVER_DEV_MODE=1 on every model instance: '+task['id'])
    for case in cases.values():
        if case.get('mode','perf') != 'perf':
            continue
        if case.get('formal_requests',case['concurrency']*4) < case['concurrency']:
            errors.append('Requests cannot reach concurrency: '+case['id'])
    if spec['state']=='ready' and spec['unresolved']:
        errors.append('Ready task has unresolved items')
    errors.extend(validate_stages(spec))
    return errors

def lookup(context, name):
    value = context
    for key in name.split('.'):
        if not isinstance(value, dict) or key not in value:
            raise ValueError('Unresolved binding: '+name)
        value = value[key]
    return value

def expand(value, context, shell=False):
    if isinstance(value, dict):
        return {k: expand(v, context, shell) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v, context, shell) for v in value]
    if not isinstance(value, str):
        return value
    match = TOKEN.fullmatch(value)
    if match and not shell:
        return lookup(context, match[1])
    def replace(match):
        result = lookup(context, match[1])
        if isinstance(result, (dict, list)):
            raise ValueError('Structured binding must occupy entire value: '+match[1])
        return shlex.quote(str(result)) if shell else str(result)
    # Expand our tokens once. Other brace syntax (e.g. Docker Go templates)
    # and braces in resolved binding values are literal text, not new tokens.
    # Missing recognized bindings still fail in lookup above.
    return TOKEN.sub(replace, value)

def higher_concurrency_blocker(spec, subtask, results):
    cases={c['id']:c for c in spec['cases']}
    current=cases[subtask['case_id']]
    for previous in spec['sub_tasks']:
        if previous['id']==subtask['id']: break
        if previous.get('on_failure')!='skip_higher_concurrency': continue
        if previous['deployment_id']!=subtask['deployment_id']: continue
        if results.get(previous['id'],{}).get('status')!='FAILED': continue
        earlier=cases[previous['case_id']]
        if earlier.get('input_tokens')==current.get('input_tokens') and earlier['concurrency']<current['concurrency']:
            return previous['id']
    return None

def validate(spec):
    from jsonschema import Draft202012Validator
    errors=[f"{e.json_path}: {e.message}" for e in Draft202012Validator(load_json(PACKAGE/"schemas/job_spec.schema.json")).iter_errors(spec)]
    return errors or validate_task(spec)


def validate_bindings(bindings):
    from jsonschema import validate as schema_validate, FormatChecker
    from pathlib import PurePosixPath
    from shared_storage import declared_groups
    schema_validate(bindings,load_json(PACKAGE/'schemas/bindings.schema.json'),format_checker=FormatChecker())
    declared_groups(bindings)
    for node in bindings['nodes'].values():
        paths=[node['work_dir'],*node['model_paths'].values(),*node.get('paths',{}).values()]
        paths.extend(value for mount in node.get('mounts',[]) for value in (mount['source'],mount['target']))
        for value in paths:
            if not PurePosixPath(value).is_absolute() or '..' in PurePosixPath(value).parts or any(c in value for c in ('\n','\r',':')):
                raise ValueError('Invalid absolute Linux binding path: '+value)


def validate_stages(spec):
    errors=[]
    tasks={s['id']:s for s in spec['sub_tasks']}
    deps={d['id']:d for d in spec['deployments']}
    used=[]; stages=set()
    for stage in spec.get('stages',[]):
        if stage['id'] in stages: errors.append('Duplicate stage ID')
        stages.add(stage['id']); reserved=set(); branches=set()
        for branch in stage['branches']:
            if branch['id'] in branches: errors.append('Duplicate branch ID')
            branches.add(branch['id']); nodes=set()
            for sid in branch['sub_tasks']:
                used.append(sid)
                if sid not in tasks:
                    errors.append('Unknown stage sub_task: '+sid); continue
                dep=deps.get(tasks[sid]['deployment_id'])
                if dep: nodes.update(i['placement']['node'] for i in dep['instances'])
            if nodes & reserved: errors.append('Parallel branches share physical nodes')
            reserved.update(nodes)
    if 'stages' in spec and (len(used)!=len(set(used)) or set(used)!=set(tasks)):
        errors.append('Stages must reference every sub_task exactly once')
    return errors
