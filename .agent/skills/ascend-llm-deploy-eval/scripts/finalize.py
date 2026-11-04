"""Explicit, idempotent policy release; keep shared images/weights and evidence."""
import json
import re
import time
import os
import signal
from service_process import ticks
from pathlib import Path
from stage_runtime import command,entry,inspect_owned,read_plan,save,state,transition,update,pool_state,save_pool


def release(root,policy,scope):
    s=read_plan(root)
    write_errors=[]
    def record(fn,*args,**kwargs):
        try: return fn(*args,**kwargs)
        except OSError as exc: write_errors.append(repr(exc))

    shared=pool_state(s) if s.get('container_registry') else {}
    if scope=='job' and shared.get('active_run_dir') and shared['active_run_dir']!=str(root):
        return release(Path(shared['active_run_dir']),policy,scope)
    if s.get('container_registry'):
        if shared.get('removed') or not shared.get('container_id'):
            return {'scope':scope,'policy':policy,'status':'released','verification':'Task container absent or already removed'}
        if scope!='job' and shared.get('active_run_dir') not in [None,'',str(root)]:
            raise ValueError('Shared container belongs to another active deployment')
        # Ownership is read from the verified registry; writing state is not a release prerequisite.
    data=state(root); evidence={'scope':scope,'policy':policy,'status':'kept' if policy=='keep_service' else 'released'}
    if shared.get('container_id'):
        data.update(container_id=shared['container_id'],image_id=shared['image_id'])
    if policy=='keep_service':
        pass
    elif not data.get('container_id'):
        evidence['verification']='No container created by this job'
    else:
        c=inspect_owned(s,data['container_id'])
        if c['Image']!=data['image_id']: raise ValueError('Registered image changed')
        if policy=='stop_service' and c['State']['Running']:
            evidence['processes']=json.loads(command(['docker','exec',c['Id'],'python3',str(Path(__file__).with_name('service_process.py')),'stop',str(root)],timeout=180))
        elif policy=='stop_container':
            snapshot_text=''
            identities={}
            if c['State']['Running']:
                snapshot_text=command(['docker','top',c['Id'],'-eo','pid,lstart,args'])
                identities={int(pid):ticks(int(pid)) for pid in re.findall(r'^\s*(\d+)\s',snapshot_text,re.M)}
                record((root/'logs/processes-before-finalize.txt').write_text,snapshot_text)
                command(['docker','stop','-t','30',c['Id']],timeout=180)
            current=inspect_owned(s,c['Id'])['State']
            evidence['container_state']=current
            if current['Running'] or current['Pid']!=0: raise RuntimeError('Container stop verification failed')
            snapshot=root/'logs/processes-before-finalize.txt'
            known=set(re.findall(r'^\s*(\d+)\s',snapshot_text or (snapshot.read_text() if snapshot.exists() else ''),re.M))
            for attempt in range(4):
                npu=command(['npu-smi','info'])
                active=set(re.findall(r'^\|\s*\d+\s*\|\s*(\d+)\s*\|',npu,re.M))
                if not known.intersection(active): break
                if attempt==3: raise RuntimeError('Owned NPU process remains after container stop')
                for pid in known.intersection(active):
                    pid=int(pid)
                    if not identities.get(pid) or ticks(pid)!=identities[pid]:
                        raise RuntimeError('Residual process identity not verified')
                    try: os.kill(pid,signal.SIGTERM if attempt==0 else signal.SIGKILL)
                    except ProcessLookupError: pass
                time.sleep(3)
            record((root/'logs/npu-after-finalize.txt').write_text,npu)
            sockets=command(['ss','-lntp'])
            record((root/'logs/ports-after-finalize.txt').write_text,sockets)
            remaining = known.intersection(re.findall(r'pid=(\d+)', sockets))
            for pid_text in remaining:
                pid = int(pid_text)
                if not identities.get(pid) or ticks(pid) != identities[pid]:
                    raise RuntimeError('Residual listener identity not verified')
                try: os.kill(pid, signal.SIGTERM)
                except ProcessLookupError: pass
            if remaining:
                time.sleep(1)
                for pid_text in remaining:
                    pid = int(pid_text)
                    if ticks(pid) == identities[pid] and identities[pid]:
                        try: os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError: pass
                time.sleep(1)
                sockets = command(['ss', '-lntp'])
                record((root/'logs/ports-after-finalize.txt').write_text, sockets)
            if known.intersection(re.findall(r'pid=(\d+)', sockets)):
                raise RuntimeError('Owned process still has listening socket')
            evidence['port_states'] = command(['ss', '-antp'])

            evidence['owned_npu_processes_remaining']=[]
            evidence['owned_listeners_remaining']=[]
            processes=root/'service-processes.json'
            if processes.exists():
                evidence['stopped_processes']=json.loads(processes.read_text())
                record(save,processes,[])
        if policy=='stop_service' and not c['State']['Running']:
            record(save,root/'service-processes.json',[])
        if s.get('container_registry'):
            record(save_pool,s,active_run_dir='')
            if scope=='job' and policy=='stop_container':
                command(['docker','rm',c['Id']])
                if command(['docker','ps','-aq','--filter','id='+c['Id']]).strip():
                    raise RuntimeError('Task container removal not verified')
                record(save_pool,s,removed=True)
                evidence['container_removed']=True
        record(transition,root,'RELEASED',smoke_identity=None)
    data=state(root); data['releases'].append(evidence); record(save,root/'run_state.json',data)
    if write_errors: evidence['recording_errors']=write_errors
    return evidence


if __name__=='__main__': raise SystemExit(entry('finalize'))
