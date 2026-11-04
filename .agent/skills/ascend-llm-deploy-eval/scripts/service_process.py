"""Container-side owned process groups; PID plus start-time prevents PID reuse."""
import json
import hashlib
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def ticks(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except FileNotFoundError:
        return None


def members(pgid):
    found=[]
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields=path.read_text().split(') ',1)[1].split()
            if fields[0]!='Z' and int(fields[2])==pgid: found.append(int(path.parent.name))
        except (FileNotFoundError,ProcessLookupError): pass
    return found


def main(action,root):
    path=root/'service-processes.json'
    data=json.loads(path.read_text()) if path.exists() else []
    if action in ['start','start_models','start_proxy']:
        if data and action!='start_proxy': raise ValueError('Existing service process registry; stop before start')
        service=json.loads((root/'scripts/service.json').read_text())
        spec=json.loads((root/'resolved.json').read_text())
        job=Path(spec.get('storage', {}).get('root', root))
        signature=hashlib.sha256(json.dumps(service,sort_keys=True).encode()).hexdigest()
        if action!='start_proxy':
            parent=job/'deployments'; parent.mkdir(parents=True,exist_ok=True)
            number=1
            while True:
                run=parent/(signature[:16]+f'-{number:02}')
                try: run.mkdir(); break
                except FileExistsError: number+=1
            metadata={'deployment_run_id':run.name,'sha256':signature,'service':service,
                      'started_at':datetime.now(timezone.utc).isoformat(),'runtime_dir':str(root)}
            (run/'deployment.json').write_text(json.dumps(metadata,indent=2))
            (root/'deployment-artifacts.json').write_text(json.dumps({'path':str(run),'id':run.name}))
        else:
            run=Path(json.loads((root/'deployment-artifacts.json').read_text())['path'])
        for item in service['commands']:
            if action=='start_models' and item.get('role')=='proxy': continue
            if action=='start_proxy' and item.get('role')!='proxy': continue
            if any(old['name']==item['name'] for old in data):
                raise ValueError('Process already registered: '+item['name'])
            with (run/(item['name']+'.log')).open('a') as log:
                proc=subprocess.Popen(item['argv'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            data.append({'name':item['name'],'pid':proc.pid,'start_ticks':ticks(proc.pid)})
            path.write_text(json.dumps(data,indent=2))
    elif action=='probe':
        observed=[ticks(x['pid']) for x in data]
        if any(now is not None and now!=old['start_ticks'] for old,now in zip(data,observed)):
            raise ValueError('Process identity changed')
        print(json.dumps({'alive':bool(data) and all(x is not None for x in observed)}))
        return
    elif action=='identity':
        if not data or any(ticks(x['pid'])!=x['start_ticks'] or x['start_ticks'] is None for x in data):
            raise ValueError('Service process missing or identity changed')
    elif action=='stop':
        for item in reversed(data):
            pid=item['pid']; current=ticks(pid)
            if current is not None and current!=item['start_ticks']:
                raise ValueError('PID ownership changed; refuse kill')
            if members(pid):
                os.killpg(pid,signal.SIGTERM)
                deadline=time.monotonic()+20
                while members(pid) and time.monotonic()<deadline: time.sleep(.2)
                if members(pid): os.killpg(pid,signal.SIGKILL)
                deadline=time.monotonic()+10
                while members(pid) and time.monotonic()<deadline: time.sleep(.2)
                if members(pid): raise RuntimeError('Owned process group remains')
        marker=root/'deployment-artifacts.json'
        if marker.exists():
            meta_path=Path(json.loads(marker.read_text())['path'])/'deployment.json'
            meta=json.loads(meta_path.read_text()); meta['stopped_at']=datetime.now(timezone.utc).isoformat()
            meta_path.write_text(json.dumps(meta,indent=2))
        path.write_text('[]')
        data={'released':True}
    else: raise ValueError('Unknown process action')
    print(json.dumps(data))

if __name__=='__main__': main(sys.argv[1],Path(sys.argv[2]))
