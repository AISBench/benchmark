"""Control-side CLI. Remote lifecycle is never executed by this process."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import sys
from .contracts import validate_command, validate_query, exit_code
from .protocol import ProtocolError, encode, envelope
from .ssh_transport import exchange, ssh_argv
from .json_io import load_json


def load(path, limit=65536):
    return load_json(path, limit)


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ProtocolError('INVALID_INPUT: command arguments')


def main(argv=None):
    task = None
    request = {}
    try:
        parser = Parser()
        parser.add_argument('--version', action='version', version='deploy-eval-v2 0.1.0a7; protocol 1.0; production_executor=guarded; acceptance=local-only')
        parser.add_argument('--config', default=os.environ.get('DEPLOY_EVAL_V2_CONFIG'))
        commands = parser.add_subparsers(dest='kind', required=True)
        for name in ('deploy', 'eval', 'release', 'cancel', 'freeze-inputs', 'verify-inputs','draft-review','package-runtime','publish-task','start-task','check-ready','install-skills','socket-start','stop-runtime','prepare-inputs','export-contracts'):
            commands.add_parser(name).add_argument('--request', required=True)
        for name in ('get-status', 'wait-event', 'read-result'):
            child = commands.add_parser(name)
            child.add_argument('--task-ref', required=True)
            if name == 'wait-event':
                child.add_argument('--after-seq', type=int, required=True)
                child.add_argument('--wait-ms', type=int, default=30000)
                child.add_argument('--limit', type=int, default=10)
            if name == 'read-result':
                child.add_argument('--artifact-ref', required=True)
                child.add_argument('--offset', type=int, default=0)
                child.add_argument('--max-bytes', type=int, default=8192)
                child.add_argument('--expected-sha256')
                locators=child.add_mutually_exclusive_group()
                locators.add_argument('--tail-bytes',type=int)
                locators.add_argument('--line-start',type=int)
                locators.add_argument('--keyword')
                child.add_argument('--line-end',type=int)
                child.add_argument('--scan-offset',type=int,default=0)
                child.add_argument('--scan-line',type=int,default=1)
        args = parser.parse_args(argv)
        if args.kind in ('freeze-inputs', 'verify-inputs','draft-review','package-runtime','publish-task','start-task','check-ready','install-skills','socket-start','stop-runtime','prepare-inputs','export-contracts'):
            request=load(args.request)
            if args.kind=='prepare-inputs':
                from .stage_inputs import prepare
                response=prepare(request)
            elif args.kind=='export-contracts':
                from .stage_inputs import export_contracts
                response=export_contracts(request)
            elif args.kind in ('socket-start','stop-runtime'):
                from .socket_validation import start_socket,stop_runtime
                if not args.config:
                    raise ProtocolError('INVALID_INPUT: config required')
                response=(start_socket if args.kind=='socket-start' else stop_runtime)(request,load(args.config))
            elif args.kind=='install-skills':
                from .skill_bundle import install
                response=install(request)
            elif args.kind=='package-runtime':
                from .publication_cli import package_runtime
                response=package_runtime(request)
            elif args.kind in ('publish-task','start-task','check-ready'):
                from .publication_cli import publish,start_task
                from .readiness import check_ready
                if not args.config:
                    raise ProtocolError('INVALID_INPUT: config required')
                task=request.get('task_ref') if isinstance(request,dict) else None
                response={'publish-task':publish,'start-task':start_task,'check-ready':check_ready}[args.kind](request,load(args.config))
            else:
                from .compiler import local_command
                response = local_command(args.kind, request)
            if len(encode(response)) > 8192:
                raise ProtocolError('INVALID_INPUT: output reference too large')
            sys.stdout.buffer.write(encode(response)+b'\n')
            return exit_code(response)
        if not args.config:
            raise ProtocolError('INVALID_INPUT: config required')
        config = load(args.config)
        if hasattr(args, 'request'):
            request = load(args.request)
            if not isinstance(request, dict) or 'task_id' in request or 'kind' in request:
                raise ProtocolError('INVALID_INPUT: request contains internal routing fields')
            task = request.pop('task_ref')
            request.update(kind=args.kind, task_id=task)
            request.setdefault('protocol_version', '1.0')
            validate_command(request)
        else:
            task = args.task_ref
            request = {k: v for k, v in vars(args).items() if k not in ('config', 'task_ref') and v is not None}
            request.update(task_id=task, protocol_version='1.0')
            validate_query(request)
        from .publication_cli import task_binding
        binding = task_binding(config,task)
        root = PurePosixPath(binding['root'])
        bootstrap='import sys; sys.path.insert(0, '+repr(str(root)+'/runtime')+'); from deploy_eval_v2.remote_client import main; raise SystemExit(main())'
        remote = [binding.get('python', 'python3'), '-B', '-c', bootstrap, '--root', str(root)]
        response = exchange(ssh_argv(binding['host'], remote), request,
                            timeout=request.get('wait_ms', 30000)/1000+15,
                            response_limit=request.get('max_bytes', 8192))
        code = exit_code(response)
    except (ProtocolError, ValueError, KeyError, OSError, TypeError) as exc:
        error = str(exc).split(':', 1)[0] if isinstance(exc, ProtocolError) else 'INVALID_INPUT'
        # Preserve the last known observation position, never claim a new cursor.
        response = envelope(task if isinstance(task, str) and len(task) <= 64 else None,
                            request.get('after_seq', 0) if isinstance(request, dict) and type(request.get('after_seq', 0)) is int else 0, 'ERROR')
        if isinstance(request, dict):
            for field in ('request_id', 'operation_id'):
                value = request.get(field)
                if isinstance(value, str) and len(value) <= 64:
                    response[field] = value
        response['error'] = dict(code=error, message=error, retryable=False)
        code = exit_code(response)
    sys.stdout.buffer.write(encode(response)+b'\n')
    return code
