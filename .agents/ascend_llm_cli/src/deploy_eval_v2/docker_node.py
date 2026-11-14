"""Node-local Docker effects with durable identities; called by the task runtime.

Every method receives a shared stage deadline. No pull, fuzzy ownership lookup,
or automatic retry of a command with unknown outcome is permitted.
"""
import json
from pathlib import Path
import re
import socket
from .compiler import digest
from .container_baseline import docker_run_args
from .process_guard import remaining, Cancelled, run_logged
from .protocol import ProtocolError
from .remote_guard import require_owned_container
from .process_io import bounded_command


class DockerNode:
    def __init__(self, guard, runner=bounded_command):
        self.guard = guard
        self.runner = runner

    def call(self, argv, deadline, cancelled):
        if cancelled.is_set():
            raise Cancelled('Operation cancelled')
        code, output, _ = self.runner(argv, timeout=min(30, remaining(deadline)), limit=8192)
        if code:
            raise ProtocolError('RECOVERY_REQUIRED: Docker command unsuccessful')
        return output

    def inspect_owned(self, cid, deadline, cancelled):
        if not isinstance(cid, str) or not re.fullmatch('[a-f0-9]{64}', cid):
            raise ProtocolError('INVALID_INPUT: immutable container ID required')
        template = '{"Id":{{json .Id}},"Config":{"Labels":{{json .Config.Labels}}}}'
        raw = self.call(['docker', 'inspect', '--format', template, cid], deadline, cancelled)
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise ProtocolError('RECOVERY_REQUIRED: invalid Docker ownership response') from exc
        require_owned_container(data, self.guard.owner['task_id'], self.guard.owner['token'], cid)
        return data

    def create(self, command_id, group, deadline, cancelled):
        # The coordinator claims all groups before entering the create phase.
        resources = ['npu:'+str(n) for n in group['npu_ids']]+['tcp:'+str(p) for p in group['ports']]
        def action():
            self.guard.claim(resources)
            for device in group['npu_ids']:
                code, _, _ = self.runner(['fuser', '/dev/davinci'+str(device)],
                                          timeout=min(30, remaining(deadline)), limit=8192)
                if code != 1:
                    raise ProtocolError('RESOURCE_BUSY: NPU occupancy not verified clear')
            # Keep all probe sockets alive together to catch overlap before create.
            probes = []
            try:
                for port in group['ports']:
                    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    probes.append(probe)
                    probe.bind(('0.0.0.0', port))
            except OSError as exc:
                raise ProtocolError('RESOURCE_BUSY: deployment port occupied') from exc
            finally:
                for probe in probes:
                    probe.close()
            image = self.call(['docker', 'image', 'inspect', '--format', '{{.Id}}', group['image']], deadline, cancelled).decode('ascii').strip()
            if not re.fullmatch('sha256:[a-f0-9]{64}', image):
                raise ProtocolError('RECOVERY_REQUIRED: invalid image identity')
            name = 'dev2-'+digest(dict(owner=self.guard.owner, command_id=command_id))[:32]
            argv = ['docker', 'create', '--name', name, '--label', 'bench.task='+self.guard.owner['task_id'],
                    '--label', 'bench.execution='+self.guard.owner['token'], '--log-driver', 'none']
            argv += docker_run_args(group['container'])
            for device in group['container']['devices']:
                if not Path(device).exists():
                    raise ProtocolError('INVALID_INPUT: required host device unavailable')
                argv += ['--device', device]
            for mount in group['mounts']:
                if not Path(mount['source']).exists():
                    raise ProtocolError('INVALID_INPUT: required host mount unavailable')
                # Docker's volume grammar has separators independent of shell quoting.
                if any(':' in mount[key] for key in ('source', 'target')):
                    raise ProtocolError('INVALID_INPUT: unsupported mount separator')
                argv += ['-v', mount['source']+':'+mount['target']+(':ro' if mount.get('read_only') else '')]
            argv += [image, 'sleep', 'infinity']
            raw = self.call(argv, deadline, cancelled).decode('ascii').strip()
            self.inspect_owned(raw, deadline, cancelled)
            return dict(container_id=raw, image_id=image, name=name, resources=resources)
        return self.guard.command(command_id, dict(kind='create', group=group), action)

    def start_container(self, command_id, cid, deadline, cancelled):
        def action():
            self.inspect_owned(cid, deadline, cancelled)
            self.call(['docker', 'start', cid], deadline, cancelled)
            return dict(container_id=cid, started=True)
        return self.guard.command(command_id, dict(kind='start', container_id=cid), action)

    def start_service(self, command_id, cid, script, deadline, cancelled):
        def action():
            self.inspect_owned(cid, deadline, cancelled)
            import hashlib
            path = Path(script['path'])
            if path.is_symlink() or path.stat().st_size > 4*1024*1024:
                raise ProtocolError('INVALID_INPUT: service script path')
            if hashlib.sha256(path.read_bytes()).hexdigest() != script['sha256']:
                raise ProtocolError('INVALID_INPUT: service script changed')
            # Detached exec belongs to an owned container, not the SSH session.
            # Acceptance is not readiness; coordinator must run health and smoke.
            self.call(['docker', 'exec', '-d', cid, 'bash', str(path)], deadline, cancelled)
            return dict(container_id=cid, instance=script['instance'], status='START_ACCEPTED', log=script['log'])
        return self.guard.command(command_id, dict(kind='service', container_id=cid, script=script), action)

    def install_scripts(self, group, scripts):
        import hashlib
        root = Path(group['root']).resolve()
        for directory in ('scripts', 'logs'):
            (root/directory).mkdir(parents=True, exist_ok=True)
        for script in scripts:
            path = Path(script['path'])
            if path.is_symlink() or path.resolve().parent != root/'scripts':
                raise ProtocolError('INVALID_INPUT: script outside task directory')
            content = script['content'].encode('utf-8')
            if hashlib.sha256(content).hexdigest() != script['sha256']:
                raise ProtocolError('INVALID_INPUT: script content digest')
            try:
                with path.open('xb') as stream:
                    stream.write(content)
                    stream.flush()
                    import os
                    os.fsync(stream.fileno())
            except FileExistsError:
                if path.stat().st_size != len(content) or path.read_bytes() != content:
                    raise ProtocolError('REQUEST_CONFLICT: existing script differs')

    def check_dependency(self, command_id, cid, script, deadline, cancelled):
        def action():
            self.inspect_owned(cid, deadline, cancelled)
            import hashlib
            path = Path(script['path'])
            if path.is_symlink() or path.stat().st_size > 4*1024*1024 or hashlib.sha256(path.read_bytes()).hexdigest() != script['sha256']:
                raise ProtocolError('INVALID_INPUT: dependency script changed')
            code = run_logged(['docker', 'exec', cid, 'bash', str(path)],
                              script['log']['path']+'.transport', deadline, cancelled)
            if code:
                raise ProtocolError('INVALID_INPUT: container dependency check failed')
            return dict(container_id=cid, instance=script['instance'], dependency_status='VERIFIED', log=script['log'])
        # If exec is cancelled, the coordinator must remove the owned container;
        # killing the docker CLI alone is not proof the in-container process stopped.
        return self.guard.command(command_id, dict(kind='dependency', container_id=cid, script=script), action)
