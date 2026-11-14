"""Bounded, immutable transport archives for already reviewed frozen inputs."""
import os
from pathlib import Path
import tempfile
import zipfile
from .compiler import verify_frozen
from .publication import hash_file
from .protocol import ProtocolError
from .durable import rename_new_directory

NAMES={'plan.json','bindings.json','review.json','review-summary.json','compiled.json','manifest.json'}
FILE_LIMIT=4*1024*1024
TOTAL_LIMIT=6*FILE_LIMIT


def build_input_archive(directory,destination,bundle_digest,runtime_digest):
    root=Path(directory)
    values=verify_frozen(root,bundle_digest,runtime_digest)
    names=set(values)|{'manifest.json'}
    if {p.name for p in root.iterdir()}!=names:
        raise ProtocolError('INVALID_INPUT: unexpected frozen input files')
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):
            path=root/name
            if path.is_symlink() or path.stat().st_size>FILE_LIMIT:
                raise ProtocolError('INVALID_INPUT: input archive file')
            archive.write(path,name)
    return dict(archive=str(destination),archive_sha256=hash_file(Path(destination)),bundle_sha256=bundle_digest)


def install_input_archive(archive_path,destination,bundle_digest,runtime_digest,*,expected_task_id=None):
    destination=Path(destination)
    if destination.is_symlink():
        raise ProtocolError('INVALID_INPUT: input destination symlink')
    if destination.exists():
        values=verify_frozen(destination,bundle_digest,runtime_digest)
        if expected_task_id is not None and values['plan.json'].get('task_id')!=expected_task_id:
            raise ProtocolError('INVALID_INPUT: input task identity')
        if {p.name for p in destination.iterdir()}!=set(values)|{'manifest.json'}:
            raise ProtocolError('INVALID_INPUT: unexpected installed inputs')
        return dict(reused=True,bundle_sha256=bundle_digest)
    destination.parent.mkdir(parents=True,exist_ok=True)
    claim=destination.parent/('.'+destination.name+'.install.lock')
    try:
        fd=os.open(claim,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError as exc:
        raise ProtocolError('RESOURCE_BUSY: input publication') from exc
    os.close(fd)
    try:
        with zipfile.ZipFile(archive_path) as archive,tempfile.TemporaryDirectory(dir=destination.parent,prefix='.inputs-') as temp:
            items=archive.infolist()
            names=[i.filename for i in items]
            if (len(items)>6 or len(set(names))!=len(names) or not set(names)<=NAMES or
                    'manifest.json' not in names or sum(i.file_size for i in items)>TOTAL_LIMIT or
                    any(i.file_size>FILE_LIMIT or i.is_dir() or (i.external_attr>>16)&0o170000==0o120000 for i in items)):
                raise ProtocolError('INVALID_INPUT: input archive inventory')
            staging=Path(temp)/'inputs'
            staging.mkdir()
            for item in items:
                with archive.open(item) as source,(staging/item.filename).open('xb') as output:
                    count=0
                    while block:=source.read(65536):
                        count+=len(block)
                        if count>item.file_size:
                            raise ProtocolError('INVALID_INPUT: input archive size')
                        output.write(block)
                    output.flush();os.fsync(output.fileno())
            values=verify_frozen(staging,bundle_digest,runtime_digest)
            if expected_task_id is not None and values['plan.json'].get('task_id')!=expected_task_id:
                raise ProtocolError('INVALID_INPUT: input task identity')
            if set(names)!=set(values)|{'manifest.json'}:
                raise ProtocolError('INVALID_INPUT: input archive manifest files')
            if destination.exists():
                raise ProtocolError('REQUEST_CONFLICT: input publication raced')
            rename_new_directory(staging,destination)
        return dict(reused=False,bundle_sha256=bundle_digest)
    finally:
        claim.unlink()
