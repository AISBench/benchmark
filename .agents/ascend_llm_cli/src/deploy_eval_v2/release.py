"""Release by immutable container identity plus task/execution labels."""
import json
import re
import socket
import time
import errno
from pathlib import Path
from .durable import atomic_json
from .compiler import digest
from .remote_guard import require_owned_container
from .process_io import bounded_command
from .protocol import ProtocolError


def verify_resources_absent(resources, runner, deadline):
    """Observe device users and port availability without terminating other owners."""
    results, probes = [], []
    try:
        for resource in resources:
            status = 'UNVERIFIED'
            try:
                kind, value = resource.split(':', 1)
                if not value.isdecimal() or time.monotonic() >= deadline:
                    raise ValueError('invalid resource or expired deadline')
                if kind == 'npu':
                    code, output, error = runner(['fuser', '/dev/davinci'+value],
                        timeout=min(30, deadline-time.monotonic()), limit=8192)
                    if code == 1 and not output.strip() and not error.strip():
                        status = 'ABSENT_VERIFIED'
                elif kind == 'tcp' and 0 < int(value) < 65536:
                    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    probes.append(probe)
                    if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                        probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                    # Closed server connections can retain TIME_WAIT briefly.
                    # Keep exclusive bind semantics; wait within the release budget.
                    while True:
                        try:
                            probe.bind(('0.0.0.0', int(value)))
                            break
                        except OSError as error:
                            if error.errno != errno.EADDRINUSE or time.monotonic()+1 >= deadline:
                                raise
                            time.sleep(min(.5, deadline-time.monotonic()))
                    status = 'ABSENT_VERIFIED'
            except (OSError, ValueError, ProtocolError, TimeoutError):
                pass
            results.append(dict(resource=resource, status=status))
    finally:
        for probe in probes:
            probe.close()
    return results


def release_container(container_id, task_id, token, runner=bounded_command, *, deadline=None):
    if not re.fullmatch('[a-f0-9]{64}', container_id):
        raise ProtocolError('INVALID_INPUT: full container identity required')
    def call(argv):
        remaining = 30 if deadline is None else deadline-time.monotonic()
        if remaining <= 0:
            raise ProtocolError('RECOVERY_REQUIRED: release deadline exceeded')
        return runner(argv, timeout=min(30, remaining), limit=8192)
    def present():
        code, output, _ = call(['docker', 'ps', '-a', '--no-trunc', '--filter',
                                  'id='+container_id, '--format', '{{.ID}}'])
        if code:
            raise ProtocolError('RECOVERY_REQUIRED: Docker inventory unavailable')
        try:
            ids = output.decode('ascii').splitlines()
        except UnicodeError as exc:
            raise ProtocolError('RECOVERY_REQUIRED: invalid Docker inventory') from exc
        if any(value != container_id for value in ids):
            raise ProtocolError('RECOVERY_REQUIRED: ambiguous container inventory')
        return bool(ids)
    if not present():
        return {'container_id': container_id, 'status': 'ABSENT_VERIFIED'}
    template = '{"Id":{{json .Id}},"Config":{"Labels":{{json .Config.Labels}}}}'
    code, output, _ = call(['docker', 'inspect', '--format', template, container_id])
    if code:
        raise ProtocolError('RECOVERY_REQUIRED: container inspection unavailable')
    try:
        inspection = json.loads(output)
    except ValueError as exc:
        raise ProtocolError('RECOVERY_REQUIRED: invalid ownership evidence') from exc
    require_owned_container(inspection, task_id, token, container_id)
    code, _, _ = call(['docker', 'rm', '-f', container_id])
    if code or present():
        raise ProtocolError('RECOVERY_REQUIRED: container release unverified')
    return {'container_id': container_id, 'status': 'RELEASED_VERIFIED'}


def release_host_resources(guard, command_id, containers, resources, evidence_path,
                           *, timeout=120, runner=bounded_command):
    """Release a frozen host inventory; retain all claims unless all absence is proven.

    Caller supplies the persisted deployment inventory, never names discovered by
    fuzzy matching. Unknown command outcomes require recovery rather than replay.
    This is a host primitive, not cross-host aggregation or an authority grant.
    """
    if (not containers or len(set(containers)) != len(containers) or
            any(not isinstance(cid, str) or not re.fullmatch('[a-f0-9]{64}', cid) for cid in containers) or
            not resources or any(not isinstance(r, str) or not r or len(r)>256 for r in resources) or
            isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 3600):
        raise ProtocolError('INVALID_INPUT: release inventory')
    payload = dict(containers=sorted(containers), resources=sorted(set(resources)), timeout=timeout,
                   evidence_path=str(Path(evidence_path).absolute()))
    def action():
        deadline = time.monotonic()+timeout
        results = []
        for cid in payload['containers']:
            try:
                result = release_container(cid, guard.owner['task_id'], guard.owner['token'], runner,
                                           deadline=deadline)
            except (ProtocolError, OSError) as error:
                # Bound the summary, do not expose daemon diagnostics in events.
                result = dict(container_id=cid, status='UNVERIFIED',
                              error_code=str(error).split(':', 1)[0] if isinstance(error, ProtocolError) else 'IO_ERROR')
            results.append(result)
        verified = all(r['status'] in ('ABSENT_VERIFIED', 'RELEASED_VERIFIED') for r in results)
        resource_results = verify_resources_absent(payload['resources'], runner, deadline) if verified else []
        verified = verified and all(r['status'] == 'ABSENT_VERIFIED' for r in resource_results)
        report = dict(schema_version='1.0', owner=guard.owner, resources=payload['resources'],
                      container_results=results, resource_results=resource_results,
                      cleanup_status='VERIFIED' if verified else 'UNVERIFIED',
                      claims_released=False)
        # Persist resource evidence before unlocking. A crash leaves conservative claims.
        atomic_json(Path(evidence_path), report)
        if verified:
            guard.release_claims(payload['resources'], resources_verified_absent=True)
            report['claims_released'] = True
            atomic_json(Path(evidence_path), report)
        return dict(cleanup_status=report['cleanup_status'], claims_released=report['claims_released'],
                    evidence_sha256=digest(report))
    return guard.command(command_id, payload, action)
