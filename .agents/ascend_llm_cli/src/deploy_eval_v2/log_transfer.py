"""Bounded diagnostic spool collection over already authenticated worker channels.

Acknowledgement never stops resources or reruns operations. Keep the remote spool
on any ambiguous transfer; an acknowledged spool is truncated under its lock.
"""
import base64
import hashlib
import json
from pathlib import Path
from .protocol import ProtocolError, now
from .task_log import locked_stream


def serve(root, action, *, offset=0, sha256=None):
    path=Path(root)/'task.jsonl'
    if type(offset) is not int or offset<0:
        raise ProtocolError('INVALID_INPUT: log offset')
    with locked_stream(path) as stream:
        stream.seek(0)
        hasher=hashlib.sha256();size=0
        for data in iter(lambda:stream.read(65536),b''):
            hasher.update(data);size+=len(data)
        signature=hasher.hexdigest()
        if action=='log-ack':
            if signature!=sha256:
                raise ProtocolError('REQUEST_CONFLICT: log spool changed before acknowledgement')
            stream.truncate(0)
            stream.flush()
            __import__('os').fsync(stream.fileno())
            return dict(acknowledged=True,sha256=signature)
        if action!='log-page' or offset>size:
            raise ProtocolError('INVALID_INPUT: log page')
        stream.seek(offset);raw=stream.read(1024)
        return dict(sha256=signature,size=size,offset=offset,
                    data=base64.b64encode(raw).decode('ascii'))


def collect(root, source, exchange):
    """Persist each byte range once before acknowledging the complete source file."""
    path=Path(root)/'task.jsonl'
    offset=0;signature=None;size=None
    # This is evidence collection only. No retry may submit a lifecycle command.
    while True:
        page=exchange('log-page',log_offset=offset)
        if (not isinstance(page,dict) or type(page.get('size')) is not int or page['size']<0 or
                page.get('offset')!=offset or not isinstance(page.get('sha256'),str) or
                len(page['sha256'])!=64):
            raise ProtocolError('TRANSPORT_PROTOCOL_ERROR: log page identity')
        try:
            raw=base64.b64decode(page['data'],validate=True)
        except (ValueError,KeyError,TypeError) as error:
            raise ProtocolError('TRANSPORT_PROTOCOL_ERROR: log page bytes') from error
        if signature is None: signature,size=page['sha256'],page['size']
        if signature!=page['sha256'] or size!=page['size'] or len(raw)>1024 or offset+len(raw)>size:
            raise ProtocolError('REQUEST_CONFLICT: log spool changed during collection')
        if not raw:
            if offset!=size:
                raise ProtocolError('TRANSPORT_PROTOCOL_ERROR: incomplete log page')
            break
        key=hashlib.sha256((source+signature+str(offset)).encode()).hexdigest()
        with locked_stream(path) as stream:
            stream.seek(0)
            exists=any(json.loads(line).get('transfer_id')==key for line in stream)
            if not exists:
                record=dict(record_type='diagnostic',time=now(),source=source,
                            message=raw.decode('utf-8',errors='replace'),event='remote.log',
                            transfer_id=key,remote_sha256=signature,remote_offset=offset,
                            remote_data=page['data'])
                stream.seek(0,2)
                stream.write(json.dumps(record,ensure_ascii=False,separators=(',',':')).encode()+b'\n')
                stream.flush();__import__('os').fsync(stream.fileno())
        offset+=len(raw)
    # Validate the accumulated transfer before letting the source discard bytes.
    hasher=hashlib.sha256();received=0
    with locked_stream(path) as stream:
        stream.seek(0)
        records={}
        for line in stream:
            record=json.loads(line)
            if record.get('event')=='remote.log' and record.get('source')==source and record.get('remote_sha256')==signature:
                records[record['remote_offset']]=record
        for start,record in sorted(records.items()):
            if start!=received: raise ProtocolError('RECOVERY_REQUIRED: collected log gap')
            data=base64.b64decode(record['remote_data'],validate=True)
            hasher.update(data);received+=len(data)
    if received!=size or hasher.hexdigest()!=signature:
        raise ProtocolError('RECOVERY_REQUIRED: collected log digest')
    ack=exchange('log-ack',log_sha256=signature)
    if ack!={'acknowledged':True,'sha256':signature}:
        raise ProtocolError('UNKNOWN_OUTCOME: log acknowledgement')
    return dict(source=source,bytes=size,sha256=signature,collected=True)
