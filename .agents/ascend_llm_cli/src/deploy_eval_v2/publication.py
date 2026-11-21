"""Immutable runtime payloads. Receiver never extracts untrusted archive paths."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile
from .hashes import canonical, digest
from .json_io import load_json, unique_object, reject_constant
from .protocol import ProtocolError
from .durable import rename_new_directory

MAX_FILE = 8*1024*1024
MAX_TOTAL = 64*1024*1024
MAX_FILES = 512


def hash_file(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):
            result.update(block)
    return result.hexdigest()


def valid_name(name):
    path = PurePosixPath(name)
    return (not path.is_absolute() and len(path.parts) >= 2 and
            path.parts[0] == 'deploy_eval_v2' and '..' not in path.parts and
            str(path) == name and '\\' not in name and '\x00' not in name and
            path.suffix in ('.py', '.json') and '__pycache__' not in path.parts)


def build_archive(package, destination):
    package, destination = Path(package), Path(destination)
    if package.is_symlink():
        raise ProtocolError('INVALID_INPUT: package symlink')
    files = {}
    total = 0
    for path in sorted(package.rglob('*')):
        if path.is_symlink():
            raise ProtocolError('INVALID_INPUT: package contains symlink')
        if not path.is_file() or '__pycache__' in path.parts or path.suffix not in ('.py', '.json'):
            continue
        name = 'deploy_eval_v2/'+path.relative_to(package).as_posix()
        size = path.stat().st_size
        total += size
        if not valid_name(name) or size > MAX_FILE or total > MAX_TOTAL or len(files) >= MAX_FILES:
            raise ProtocolError('INVALID_INPUT: runtime payload limit')
        files[name] = dict(sha256=hash_file(path), size=size)
    if 'deploy_eval_v2/server.py' not in files or 'deploy_eval_v2/__init__.py' not in files:
        raise ProtocolError('INVALID_INPUT: incomplete runtime package')
    manifest = dict(schema_version='1.0', files=files)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('manifest.json', canonical(manifest))
        for name in files:
            path = package/str(PurePosixPath(name).relative_to('deploy_eval_v2'))
            archive.write(path, name)
    return dict(runtime_sha256=digest(manifest), archive_sha256=hash_file(destination),
                archive=str(destination), file_count=len(files))


def verify_runtime(root, expected_digest):
    root = Path(root)
    manifest = load_json(root/'manifest.json')
    if not isinstance(manifest, dict) or manifest.get('schema_version') != '1.0' or digest(manifest) != expected_digest:
        raise ProtocolError('INVALID_INPUT: runtime manifest identity')
    files = manifest.get('files')
    if not isinstance(files, dict) or not 1 <= len(files) <= MAX_FILES:
        raise ProtocolError('INVALID_INPUT: runtime manifest files')
    for name, evidence in files.items():
        if not valid_name(name) or not isinstance(evidence, dict):
            raise ProtocolError('INVALID_INPUT: runtime file entry')
        path = root.joinpath(*PurePosixPath(name).parts)
        if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
            raise ProtocolError('INVALID_INPUT: runtime symlink')
        if not path.is_file() or path.stat().st_size != evidence.get('size') or hash_file(path) != evidence.get('sha256'):
            raise ProtocolError('INVALID_INPUT: runtime content changed')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != set(files) | {'manifest.json'}:
        raise ProtocolError('INVALID_INPUT: unmanifested runtime content')
    return manifest


def install_archive(archive_path, destination, expected_digest):
    """Local receiver primitive; only creates a new immutable publication."""
    destination = Path(destination)
    if destination.exists():
        verify_runtime(destination, expected_digest)
        return {'reused': True, 'runtime_sha256': expected_digest}
    destination.parent.mkdir(parents=True, exist_ok=True)
    claim = destination.parent/('.'+destination.name+'.publish.lock')
    try:
        descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ProtocolError('RESOURCE_BUSY: runtime publication') from exc
    os.close(descriptor)
    try:
        with zipfile.ZipFile(archive_path) as archive, tempfile.TemporaryDirectory(dir=destination.parent, prefix='.publish-') as temp:
            items = archive.infolist()
            names = [item.filename for item in items]
            if len(items) > MAX_FILES+1 or len(names) != len(set(names)) or 'manifest.json' not in names:
                raise ProtocolError('INVALID_INPUT: archive entries')
            if sum(item.file_size for item in items) > MAX_TOTAL or any(item.file_size > MAX_FILE for item in items):
                raise ProtocolError('INVALID_INPUT: archive size')
            if any(name != 'manifest.json' and not valid_name(name) for name in names):
                raise ProtocolError('INVALID_INPUT: archive path')
            staging = Path(temp)/'runtime'
            staging.mkdir()
            for item in items:
                path = staging.joinpath(*PurePosixPath(item.filename).parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, path.open('xb') as target:
                    count = 0
                    for block in iter(lambda: source.read(65536), b''):
                        count += len(block)
                        if count > item.file_size:
                            raise ProtocolError('INVALID_INPUT: archive size mismatch')
                        target.write(block)
                    target.flush()
                    os.fsync(target.fileno())
            verify_runtime(staging, expected_digest)
            if destination.exists():
                raise ProtocolError('REQUEST_CONFLICT: publication raced')
            rename_new_directory(staging,destination)
        return {'reused': False, 'runtime_sha256': expected_digest}
    finally:
        claim.unlink()
