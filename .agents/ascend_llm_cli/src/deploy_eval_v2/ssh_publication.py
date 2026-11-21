"""Control-side streaming upload; no unlimited capture or archive base64 copy."""
from pathlib import Path,PurePosixPath
from .publication import hash_file
from .protocol import ProtocolError
from .ssh_transport import ssh_argv,exchange


def upload_archive(host,python,root,archive,task_id,*,timeout=120,transport=exchange):
    if (not isinstance(root,str) or not root.startswith('/') or root=='/' or
            '..' in PurePosixPath(root).parts or any(c in root for c in '\x00\r\n\\') or
            not isinstance(python,str) or not python.startswith('/')):
        raise ProtocolError('INVALID_INPUT: upload target')
    archive=Path(archive)
    if archive.is_symlink() or not 0<archive.stat().st_size<=64*1024*1024:
        raise ProtocolError('INVALID_INPUT: upload archive')
    checksum=hash_file(archive)
    source=Path(__file__).with_name('blob_receiver.py').read_text(encoding='utf-8')
    request=dict(protocol_version='1.0',task_id=task_id,sha256=checksum,size=archive.stat().st_size)
    response=transport(ssh_argv(host,[python,'-B','-c',source,root]),request,
                       input_file=archive,timeout=timeout)
    if (response.get('uploaded_sha256')!=checksum or response.get('size')!=request['size'] or
            response.get('status')!='EVENTS'):
        raise ProtocolError('UNKNOWN_OUTCOME: upload identity not verified')
    return dict(host=host,path=root+'/.incoming/'+checksum+'.zip',sha256=checksum,size=request['size'])


def publish_task(host,python,root,runtime_archive,input_archive,task_id,runtime_digest,bundle_digest,*,transport=exchange):
    runtime=upload_archive(host,python,root,runtime_archive,task_id,transport=transport)
    inputs=upload_archive(host,python,root,input_archive,task_id,transport=transport)
    source=Path(__file__).with_name('publication_receiver.py').read_text(encoding='utf-8')
    request=dict(protocol_version='1.0',task_id=task_id,runtime_archive_sha256=runtime['sha256'],
                 inputs_archive_sha256=inputs['sha256'],runtime_sha256=runtime_digest,bundle_sha256=bundle_digest)
    result=transport(ssh_argv(host,[python,'-B','-c',source,root]),request,timeout=120)
    if result.get('status')!='EVENTS' or result.get('runtime',{}).get('runtime_sha256')!=runtime_digest or result.get('inputs',{}).get('bundle_sha256')!=bundle_digest:
        raise ProtocolError('RECOVERY_REQUIRED: remote publication not verified')
    return result
