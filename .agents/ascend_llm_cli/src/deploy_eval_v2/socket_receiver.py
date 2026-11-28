"""Authenticate uploaded acceptance runtime before zipimport."""
import hashlib
import json
from pathlib import Path
import re
import sys

task = None
try:
    request = json.loads(sys.stdin.buffer.readline(8193))
    if isinstance(request.get('task_id'), str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,63}', request['task_id']):
        task = request['task_id']
    root = Path(sys.argv[1])
    if not root.is_absolute() or root == Path('/') or '..' in root.parts or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('root')
    checksum = request['archive_sha256']
    if not isinstance(checksum, str) or not re.fullmatch('[a-f0-9]{64}', checksum):
        raise ValueError('digest')
    archive = root/'.incoming'/(checksum+'.zip')
    if archive.is_symlink() or archive.parent.is_symlink() or archive.stat().st_size > 64*1024*1024:
        raise ValueError('archive')
    with archive.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != checksum:
        raise ValueError('checksum')
    sys.path.insert(0, str(archive))
    from deploy_eval_v2.publication import install_archive
    install_archive(archive, root/'runtime', request['runtime_sha256'])
    for name in list(sys.modules):
        if name == 'deploy_eval_v2' or name.startswith('deploy_eval_v2.'):
            del sys.modules[name]
    sys.path[0] = str(root/'runtime')
    from deploy_eval_v2.socket_validation import start_installed
    result = start_installed(root, request)
except Exception as error:
    result = dict(protocol_version='1.0', task_id=task, status='ERROR',
                  error=dict(code='RECOVERY_REQUIRED', message=str(error)[:512], retryable=False))
print(json.dumps(result, separators=(',', ':')))
