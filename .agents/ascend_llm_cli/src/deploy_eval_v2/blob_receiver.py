"""Standalone stdlib bootstrap for content-addressed SSH uploads.

This file's exact source is passed to the remote Python interpreter before the
runtime package exists. Never import deploy_eval_v2 or execute uploaded content.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile


def pairs(items):
    result={}
    for key,value in items:
        if key in result:
            raise ValueError('duplicate key')
        result[key]=value
    return result


def receive(root,source):
    raw=source.readline(8193)
    if len(raw)>8192 or not raw.endswith(b'\n'):
        raise ValueError('upload header')
    request=json.loads(raw,object_pairs_hook=pairs)
    if (not isinstance(request,dict) or set(request)!={'protocol_version','task_id','sha256','size'} or
            request['protocol_version']!='1.0' or not isinstance(request['task_id'],str) or
            not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,63}',request['task_id']) or
            not isinstance(request['sha256'],str) or not re.fullmatch('[a-f0-9]{64}',request['sha256']) or
            type(request['size']) is not int or not 0<request['size']<=64*1024*1024):
        raise ValueError('upload identity or size')
    root=Path(root)
    if not root.is_absolute() or root==Path(root.anchor) or '..' in root.parts:
        raise ValueError('upload root')
    if any(path.is_symlink() for path in (root,*root.parents)):
        raise ValueError('upload root symlink')
    directory=root/'.incoming'
    if directory.is_symlink():
        raise ValueError('upload directory symlink')
    directory.mkdir(parents=True,exist_ok=True)
    target=directory/(request['sha256']+'.zip')
    claim=directory/(request['sha256']+'.lock')
    fd=os.open(claim,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.close(fd)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=directory,prefix='.upload-',delete=False) as output:
            temporary=Path(output.name)
            remaining=request['size']
            digest=hashlib.sha256()
            while remaining:
                block=source.read(min(65536,remaining))
                if not block:
                    raise ValueError('incomplete upload')
                output.write(block);digest.update(block);remaining-=len(block)
            if source.read(1) or digest.hexdigest()!=request['sha256']:
                raise ValueError('upload digest or size mismatch')
            output.flush();os.fsync(output.fileno())
        reused=target.exists()
        if target.is_symlink():
            raise ValueError('upload target symlink')
        if reused:
            existing=hashlib.sha256()
            if target.stat().st_size!=request['size']:
                raise ValueError('existing upload differs')
            with target.open('rb') as stream:
                while block:=stream.read(65536):
                    existing.update(block)
            if existing.hexdigest()!=request['sha256']:
                raise ValueError('existing upload differs')
        else:
            temporary.rename(target)
        return dict(protocol_version='1.0',task_id=request['task_id'],status='EVENTS',
                    uploaded_sha256=request['sha256'],size=request['size'],reused=reused)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
        claim.unlink()


def main():
    try:
        result=receive(sys.argv[1],sys.stdin.buffer)
        code=0
    except Exception:
        # Unknown identity cannot be guessed after malformed/incomplete framing.
        result=dict(protocol_version='1.0',task_id=None,status='ERROR',
                    error=dict(code='UNKNOWN_OUTCOME',message='Upload not verified',retryable=False))
        code=3
    sys.stdout.write(json.dumps(result,separators=(',',':'))+'\n')
    return code


if __name__=='__main__':
    raise SystemExit(main())
