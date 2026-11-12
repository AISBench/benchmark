"""Serialized runtime start receipts survive SSH loss and prevent blind restart."""
import argparse
import json
from pathlib import Path
import sys
from .durable import atomic_json
from .hashes import digest
from .json_io import load_json,unique_object,reject_constant
from .launcher import validate_identity,ensure_started
from .protocol import ProtocolError,envelope,encode
from .remote_guard import exclusive


def start(root,request,*,starter=ensure_started):
    fields={'protocol_version','task_id','runtime_sha256','bundle_sha256','nonce','python'}
    if not isinstance(request,dict) or set(request)!=fields:
        raise ProtocolError('INVALID_INPUT: startup request')
    identity={key:value for key,value in request.items() if key!='python'}
    identity['task_root']=str(root)
    validate_identity(identity)
    root=Path(root)
    control=root/'.runtime'
    with exclusive(control/'launch.lock'):
        record_path=control/'launch-receipt.json'
        identity_path=control/'launch-identity.json'
        if identity_path.exists() and load_json(identity_path)!=identity:
            raise ProtocolError('REQUEST_CONFLICT: runtime launch identity differs')
        previous=load_json(record_path) if record_path.exists() else None
        if previous is not None and previous.get('identity_sha256')!=digest(identity):
            raise ProtocolError('RECOVERY_REQUIRED: launch receipt identity mismatch')
        if not identity_path.exists():
            atomic_json(identity_path,identity)
        (root/'logs').mkdir(parents=True,exist_ok=True)
        receipt=dict(identity_sha256=digest(identity),status='ACCEPTED')
        if previous is None:
            atomic_json(record_path,receipt)
        try:
            result=starter(identity,request['python'],str(identity_path),allow_start=previous is None)
        except Exception as error:
            atomic_json(record_path,dict(receipt,status='UNVERIFIED',error=str(error)[:1024]))
            raise
        atomic_json(record_path,dict(receipt,status='VERIFIED',evidence=result))
    response=envelope(identity['task_id'],0)
    response.update(runtime=result,started=True,stage_submitted=False)
    return response


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    args=parser.parse_args()
    task=None
    try:
        raw=sys.stdin.buffer.readline(8193)
        if len(raw)>8192 or not raw.endswith(b'\n'):
            raise ProtocolError('INVALID_INPUT: startup request budget')
        request=json.loads(raw,object_pairs_hook=unique_object,parse_constant=reject_constant)
        if isinstance(request,dict) and isinstance(request.get('task_id'),str) and len(request['task_id'])<=64:
            task=request['task_id']
        response=start(args.root,request)
        code=0
    except Exception as error:
        reason=str(error).split(':',1)[0] if isinstance(error,ProtocolError) else 'RECOVERY_REQUIRED'
        response=envelope(task,0,'ERROR')
        response['error']=dict(code=reason,message=reason,retryable=False)
        code=3 if reason=='UNKNOWN_OUTCOME' else 2
    sys.stdout.buffer.write(encode(response)+b'\n')
    return code


if __name__=='__main__':
    raise SystemExit(main())
