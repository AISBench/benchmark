"""Explicit shared-storage declarations, verified before any bundle upload."""
import json
from pathlib import Path, PurePosixPath
import uuid

def declared_groups(bindings):
    from jsonschema import validate
    schema = json.loads((Path(__file__).parent.parent/'schemas/shared_storage.schema.json').read_text(encoding='utf-8'))
    groups = bindings.get('shared_storage_groups', [])
    validate(groups, schema)
    seen=set(); result=[]
    for group in groups:
        members=[]
        for node in group['nodes']:
            if node not in bindings['nodes'] or node in seen:
                raise ValueError('Unknown or repeated shared-storage node: '+node)
            seen.add(node)
            binding=bindings['nodes'][node]
            members.append((binding['host'],binding['work_dir']))
        if len(set(members)) != len(members): raise ValueError('Duplicate shared-storage destination')
        for _,path in members:
            p=PurePosixPath(path)
            if not p.is_absolute() or '..' in p.parts: raise ValueError('Invalid shared root')
        result.append((group['id'],members))
    return result

def verify_groups(bindings, ssh, report_path):
    groups=declared_groups(bindings)
    report={'status':'RUNNING','groups':[], 'submission_decision':'pending_attended_review'}
    verified=[]
    try:
        for name,members in groups:
            item={'id':name,'members':members,'checks':[]}; report['groups'].append(item)
            # Each member writes a unique file, all members must read identical bytes.
            for writer,root in members:
                filename=str(PurePosixPath(root)/('.skill-storage-probe-'+uuid.uuid4().hex))
                token=uuid.uuid4().hex
                try:
                    ssh(writer,['python3','-c',"import pathlib,sys; p=pathlib.Path(sys.argv[1]); f=p.open('x'); f.write(sys.argv[2]); f.close()",filename,token])
                    for reader,reader_root in members:
                        actual=ssh(reader,['python3','-c',"import pathlib,sys; print(pathlib.Path(sys.argv[1]).read_text())",str(PurePosixPath(reader_root)/PurePosixPath(filename).name)]).strip()
                        if actual!=token: raise ValueError('Shared-storage contents differ: '+name)
                        item['checks'].append({'writer':writer,'reader':reader,'matched':True})
                finally:
                    # Unique owned path only; missing files are harmless after failed creation.
                    ssh(writer,['python3','-c',"import pathlib,sys; pathlib.Path(sys.argv[1]).unlink(missing_ok=True)",filename])
            verified.append(members)
        report['status']='PASSED'
        return verified
    except Exception as exc:
        report.update(status='FAILED',error=str(exc)); raise
    finally:
        report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
