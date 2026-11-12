"""Install the five flat skill sources without overwriting existing skills."""
from pathlib import Path
import hashlib
import os
import shutil
import tempfile
from .protocol import ProtocolError, envelope

NAMES = ('ascend-llm-router','ascend-llm-plan','ascend-llm-deployment','ascend-llm-eval','ascend-llm-release')


def inventory(root):
    root = Path(root)
    files, size = {}, 0
    for name in NAMES:
        directory = root/name
        if directory.is_symlink() or not (directory/'SKILL.md').is_file():
            raise ProtocolError('INVALID_INPUT: incomplete five-skill bundle')
        for path in sorted(directory.rglob('*')):
            if path.is_symlink():
                raise ProtocolError('INVALID_INPUT: skill symlink')
            if not path.is_file():
                continue
            size += path.stat().st_size
            if size > 8*1024*1024 or len(files) >= 128:
                raise ProtocolError('INVALID_INPUT: skill bundle too large')
            files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        text = (directory/'SKILL.md').read_text(encoding='utf-8')
        if (not text.startswith('---\n') or '\nname: '+name+'\n' not in text or
                '协议 1.0' not in text or 'CLI' not in text or '--version' not in text):
            raise ProtocolError('INVALID_INPUT: skill identity or dependency declaration')
    routing = (root/NAMES[0]/'SKILL.md').read_text(encoding='utf-8')
    if any(name not in routing for name in NAMES):
        raise ProtocolError('INVALID_INPUT: routing dependencies incomplete')
    return files


def install(request):
    if not isinstance(request,dict) or set(request) != {'source','destination'}:
        raise ProtocolError('INVALID_INPUT: skill installation request')
    source, destination = (Path(request[key]).resolve() for key in ('source','destination'))
    expected = inventory(source)
    destination.mkdir(parents=True,exist_ok=True)
    # Validate every existing destination before adding any skill. Never replace
    # an installed draft, stable skill or a user's edited source implicitly.
    existing=[]
    for name in NAMES:
        target=destination/name
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_dir():
                raise ProtocolError('REQUEST_CONFLICT: installed skill differs')
            actual={}
            for path in target.rglob('*'):
                if path.is_symlink():
                    raise ProtocolError('REQUEST_CONFLICT: installed skill symlink')
                if path.is_file():
                    if path.stat().st_size>8*1024*1024:
                        raise ProtocolError('REQUEST_CONFLICT: installed skill differs')
                    actual[path.relative_to(destination).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != {key:value for key,value in expected.items() if key.startswith(name+'/')}:
                raise ProtocolError('REQUEST_CONFLICT: installed skill differs')
            existing.append(name)
    with tempfile.TemporaryDirectory(prefix='.deploy-eval-v2-',dir=destination) as temporary:
        stage=Path(temporary)
        for name in NAMES:
            shutil.copytree(source/name,stage/name)
        if inventory(stage)!=expected:
            raise ProtocolError('REQUEST_CONFLICT: skill source changed while copying')
        for name in NAMES:
            if name not in existing:
                # rename refuses an existing nonempty destination; no destructive merge.
                os.rename(stage/name,destination/name)
    inventory(destination)
    response=envelope(None,0)
    response.update(skills=list(NAMES),cli_version='0.1.0a6',protocol_version='1.0',installed=True)
    return response
