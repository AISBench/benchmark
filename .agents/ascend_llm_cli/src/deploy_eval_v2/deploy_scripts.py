"""Deterministic service scripts; no process creation during compilation.

Bootstrap is reviewed user shell text. Confirmed launch environment is applied
after bootstrap, matching the baseline's device-selection guarantee. Dependency
checks deliberately do not execute bootstrap or start a model.
"""
import hashlib
import re
import shlex
from .protocol import ProtocolError


def render_service_script(command, log_path):
    if (not isinstance(log_path, str) or not log_path.startswith('/') or
            any(ord(c) < 32 for c in log_path)):
        raise ProtocolError('INVALID_INPUT: service log path')
    argv = command['argv']
    if not argv or any(not isinstance(a, str) or '\x00' in a for a in argv):
        raise ProtocolError('INVALID_INPUT: service argv')
    env = command['env']
    if any(not re.fullmatch('[A-Za-z_][A-Za-z0-9_]*', key) for key in env):
        raise ProtocolError('INVALID_INPUT: environment name')
    # Redirection happens before any bootstrap output, including sourced scripts.
    lines = ['#!/bin/bash', 'exec >>'+shlex.quote(log_path)+' 2>&1 || exit $?',
             command['bootstrap']]
    lines.extend('export '+key+'='+shlex.quote(str(value)) for key, value in sorted(env.items()))
    lines.append('exec '+shlex.join(argv))
    return '\n'.join(lines)+'\n'


def render_dependency_script(command, log_path):
    env = command['env']
    if command['role'] == 'proxy':
        argv = command['argv']
        if len(argv) < 2 or argv[0] != 'python3':
            raise ProtocolError('INVALID_INPUT: proxy dependency argv')
        check = 'from pathlib import Path; p=Path('+repr(argv[1])+'); assert p.is_file(); p.open("rb").read(1)'
    else:
        check = 'import torch_npu; import vllm; print(torch_npu.__file__); print(vllm.__file__)'
    return render_service_script(dict(argv=['python3', '-c', check], env=env,
                                      bootstrap='', role=command['role']), log_path)


def compile_service_scripts(compiled):
    """Return per-host scripts with hashes and explicit host/path artifact locations."""
    result = {}
    for group_id, group in compiled['groups'].items():
        scripts = []
        for command in group['commands']:
            name = command['name']
            if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', name):
                raise ProtocolError('INVALID_INPUT: instance script name')
            for purpose, renderer in [('service', render_service_script), ('dependency', render_dependency_script)]:
                path = group['root']+'/scripts/'+name+'.'+purpose+'.sh'
                log = group['root']+'/logs/'+name+'.'+purpose+'.log'
                content = renderer(command, log)
                scripts.append(dict(instance=name, purpose=purpose, host=group['host'], path=path,
                                    log=dict(host=group['host'], path=log), content=content,
                                    sha256=hashlib.sha256(content.encode('utf-8')).hexdigest()))
        result[group_id] = scripts
    return result
