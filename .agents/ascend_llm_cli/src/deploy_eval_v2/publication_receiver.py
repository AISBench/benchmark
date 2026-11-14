"""Standalone install bootstrap: verify uploaded code before importing it."""
import hashlib
import json
from pathlib import Path
import re
import sys


def main():
    task=None
    try:
        root=Path(sys.argv[1])
        raw=sys.stdin.buffer.readline(8193)
        if len(raw)>8192 or not raw.endswith(b'\n'):
            raise ValueError('install request budget')
        request=json.loads(raw)
        fields={'protocol_version','task_id','runtime_archive_sha256','inputs_archive_sha256','runtime_sha256','bundle_sha256'}
        if set(request)!=fields or request['protocol_version']!='1.0':
            raise ValueError('install request fields')
        if not isinstance(request['task_id'],str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,63}',request['task_id']):
            raise ValueError('install task identity')
        task=request['task_id']
        if any(not isinstance(request[k],str) or not re.fullmatch('[a-f0-9]{64}',request[k]) for k in fields-{'protocol_version','task_id'}):
            raise ValueError('install digest')
        if not root.is_absolute() or root==Path(root.anchor) or '..' in root.parts or any(p.is_symlink() for p in (root,*root.parents)):
            raise ValueError('install root')
        archives=[]
        for field in ('runtime_archive_sha256','inputs_archive_sha256'):
            archive=root/'.incoming'/(request[field]+'.zip')
            if archive.is_symlink() or archive.parent.is_symlink() or archive.stat().st_size>64*1024*1024:
                raise ValueError('install archive path')
            digest=hashlib.sha256()
            with archive.open('rb') as stream:
                while block:=stream.read(65536):
                    digest.update(block)
            if digest.hexdigest()!=request[field]:
                raise ValueError('install archive changed')
            archives.append(archive)
        # zipimport is allowed only after authenticating the full runtime archive.
        sys.path.insert(0,str(archives[0]))
        from deploy_eval_v2.publication import install_archive
        runtime=install_archive(archives[0],root/'runtime',request['runtime_sha256'])
        # Schema data uses filesystem Paths, so input validation imports installed code.
        for name in list(sys.modules):
            if name=='deploy_eval_v2' or name.startswith('deploy_eval_v2.'):
                del sys.modules[name]
        sys.path[0]=str(root/'runtime')
        from deploy_eval_v2.input_publication import install_input_archive
        from deploy_eval_v2.compiler import verify_frozen
        inputs=install_input_archive(archives[1],root/'inputs',request['bundle_sha256'],request['runtime_sha256'],expected_task_id=task)
        values=verify_frozen(root/'inputs',request['bundle_sha256'],request['runtime_sha256'])
        if values['plan.json'].get('task_id')!=task:
            raise ValueError('installed task identity mismatch')
        result=dict(protocol_version='1.0',task_id=task,status='EVENTS',runtime=runtime,inputs=inputs)
        code=0
    except Exception:
        result=dict(protocol_version='1.0',task_id=task,status='ERROR',
                    error=dict(code='RECOVERY_REQUIRED',message='Publication not verified; inspect inputs and Python dependencies',retryable=False))
        code=2
    sys.stdout.write(json.dumps(result,separators=(',',':'))+'\n')
    return code


if __name__=='__main__':
    raise SystemExit(main())
