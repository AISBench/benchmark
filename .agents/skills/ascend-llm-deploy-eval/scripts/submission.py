"""Linux detached launcher with durable identity and exit evidence."""

if __name__ == '__main__':
    raise SystemExit('Retired lifecycle entry. Use scripts/run.py --mode FULL|scripts; historical frozen jobs retain their own code.')
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from stage_runtime import save

def now(): return datetime.now(timezone.utc).isoformat()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['submit','worker','status']);ap.add_argument('--run-dir',type=Path,required=True);a=ap.parse_args()
    root=a.run_dir.resolve();record=root/'submission.json'
    if a.action=='status':
        for name in ['summary.json','task-state.json','storage-state.json','submission.json']:
            path=root/name
            if path.exists(): print(json.dumps({'source':str(path),'state':json.loads(path.read_text())}))
        return
    if a.action=='submit':
        if not (root/'resolved-cluster.json').exists(): raise ValueError('Preflight must pass before submission')
        # Reservation prevents replay even if the process exits before status is written.
        with (root/'submission-reserved').open('x') as f: f.write(now())
        (root/'.internal/logs').mkdir(parents=True, exist_ok=True)
        with (root/'.internal/logs/submission.log').open('ab') as log:
            process=subprocess.Popen([sys.executable,__file__,'worker','--run-dir',str(root)],
                stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,cwd=root)
        print(json.dumps({'status':'SUBMITTED','wrapper_pid':process.pid,'run_dir':str(root)}));return
    data={'wrapper_pid':os.getpid(),'started_at':now(),'status':'RUNNING'}
    save(record,data)
    try:
        process=subprocess.Popen(['bash',str(root/'run.sh')],stdin=subprocess.DEVNULL,cwd=root)
        data.update(pid=process.pid,start_ticks=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19])
        save(record,data);code=process.wait()
        data.update(status='EXITED',exit_code=code)
    except Exception as exc:
        data.update(status='FAILED',error=repr(exc));raise
    finally:
        data['finished_at']=now();save(record,data)

if __name__=='__main__': main()
