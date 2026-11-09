"""Execute generated Bash with fake Docker/AISBench, actual result validator.

No SSH, containers, devices or model processes are used. Linux also runs this test;
Git Bash covers shell glue on Windows, with a path adapter for container Python.
"""
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from test_compact_entry import inputs
import run as entry


def bash_path(path):
    value=str(path).replace('\\','/')
    return '/'+value[0].lower()+value[2:] if len(value)>1 and value[1]==':' else value


DOCKER=r'''#!/usr/bin/env bash
set -euo pipefail
command=$1; shift
case "$command" in
 info) exit 0;;
 inspect)
  test -f "$FAKE_DIR/exists" || exit 1
  if [[ "${1:-}" == -f ]]; then
   case "$2" in
    *bench.execution*) cat "$FAKE_DIR/token";;
    *bench.task*) printf '%s' "$FAKE_JOB";;
    *) printf 'fake-container-id';;
   esac
  fi;;
 run)
  while (($#)); do
   if [[ "$1" == --label && "$2" == bench.execution=* ]]; then printf '%s' "${2#bench.execution=}" > "$FAKE_DIR/token"; fi
   shift
  done
  touch "$FAKE_DIR/exists"; printf 'fake-container-id';;
 rm) rm "$FAKE_DIR/exists";;
 exec)
  if [[ "$1" == -d ]]; then
   printf 'fake model started\n' > "${@: -1}"; exit 0
  fi
  shift # container name
  if [[ "$1" == bash && "$2" == -c && "$3" == *kill\ -0* ]]; then exit 0; fi
  "$@";;
 *) echo "Unexpected fake Docker operation: $command" >&2; exit 2;;
esac
'''

AISBENCH=r'''#!/usr/bin/env bash
set -euo pipefail
while (($#)); do
 if [[ "$1" == --work-dir ]]; then OUT=$2; shift; fi
 shift
done
mkdir -p "$OUT/time/performances/service" "$OUT/time/logs"
printf '{"id":0,"success":true,"input_tokens":1,"output_tokens":1}\n' > "$OUT/time/performances/service/dataset_details.jsonl"
printf 'fake worker\n' > "$OUT/time/logs/worker.log"
printf 'fake AISBench complete\n'
exit "${FAKE_AISBENCH_EXIT:-0}"
'''


class GeneratedLifecycle(unittest.TestCase):
    def test_shell_lifecycle_success_failed_test_and_foreign_container(self):
        bash=Path('C:/Program Files/Git/bin/bash.exe') if os.name=='nt' else shutil.which('bash')
        if not bash or not Path(bash).exists():self.skipTest('Bash unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);bin_dir=base/'bin';bin_dir.mkdir();state=base/'fake';state.mkdir()
            weights=base/'weights';weights.mkdir();dataset=base/'data.jsonl'
            dataset.write_text('{"question":"hello","answer":"yes"}\n')
            spec,bindings=inputs();case=spec['cases'][0]
            case.update(concurrency=1,formal_requests=1,input_tokens=1,output_tokens=8,reset_cache=False)
            dep=spec['deployments'][0];dep['runtime']['container']['devices']=['/dev/null']
            binding=bindings['nodes']['node_a'];binding['work_dir']=bash_path(base)
            binding['model_paths']['dsv4']=bash_path(weights)
            binding['benchmark']['datasets'][case['dataset_id']]['path']=bash_path(dataset)
            # Reserve an ephemeral contiguous range for each declared port block.
            # Fixed example ports can belong to real workloads on a Linux test host.
            reserved=[]
            try:
                for instance in dep['instances']:
                    for name,width in instance['ports'].items():
                        for attempt in range(100):
                            block=[]
                            try:
                                first=socket.socket();block.append(first);first.bind(('0.0.0.0',0))
                                port=first.getsockname()[1]
                                if port+width>65536:raise OSError('range overflow')
                                for offset in range(1,width):
                                    item=socket.socket();block.append(item);item.bind(('0.0.0.0',port+offset))
                                reserved.extend(block)
                                binding['deployments'][dep['id']]['ports'][instance['id']][name]=port
                                break
                            except OSError:
                                for item in block:item.close()
                        else:raise RuntimeError('No isolated test port range available')
                root=base/spec['job_id'];entry.generate(spec,bindings,root)
            finally:
                for item in reserved:item.close()
            bridge=base/'evidence_bridge.py'
            bridge.write_text('''import os,sys
from pathlib import Path
root=Path(os.environ['FAKE_ROOT_WIN']);sys.path.insert(0,str(root/'.runtime'))
import test_evidence as evidence
# Test transport adapters: container POSIX paths and Prometheus fixture.
evidence.dataset_for=lambda *args: {'path':os.environ['FAKE_DATA_WIN'],'input_column':'question','output_column':'answer'}
real_capture=evidence.capture
evidence.capture=lambda instances,folder,phase: real_capture(instances,folder,phase,fetch=lambda url:'vllm:prefix_cache_queries_total 1\\nvllm:prefix_cache_hits_total 0\\n')
args=sys.argv[1:];output=root/'results'/args[3].rsplit('/',1)[-1]
evidence.check(str(root),args[1],args[2],str(output),*args[4:])
''')
            helpers={'docker':DOCKER,'ais_bench':AISBENCH,'fuser':'#!/usr/bin/env bash\nexit 1\n',
                     'npu-smi':'#!/usr/bin/env bash\necho fake-npu\n',
                     'python3':'#!/usr/bin/env bash\nif [[ "$1" == *check_test.py ]]; then shift; exec '+shlex.quote(str(Path(sys.executable)).replace('\\','/'))+' '+shlex.quote(str(bridge).replace('\\','/'))+' "$@"; fi\nexec '+shlex.quote(str(Path(sys.executable)).replace('\\','/'))+' "$@"\n'}
            for name,text in helpers.items():
                file=bin_dir/name;file.write_text(text,newline='\n');file.chmod(0o755)
            env={**os.environ,'PATH':str(bin_dir)+os.pathsep+os.environ['PATH'],'FAKE_DIR':bash_path(state),
                 'FAKE_JOB':spec['job_id'],'FAKE_ROOT_WIN':str(root),'FAKE_DATA_WIN':str(dataset)}
            def call(relative,*args,expected=0):
                result=subprocess.run([str(bash),bash_path(root/relative),*args],env=env,capture_output=True,text=True,timeout=30)
                self.assertEqual(result.returncode,expected,result.stdout+result.stderr)
                return result
            stem='scripts/'+dep['id']+'/node_a'
            token='a'*32
            call(stem+'-container.sh',token)
            call(stem+'-start.sh','run01');call(stem+'-probe.sh')
            test='scripts/tests/'+spec['sub_tasks'][0]['id']+'.sh'
            call(test,'try01')
            result=root/'results'/(spec['sub_tasks'][0]['id']+'-try01')
            self.assertEqual(json.loads((result/'validated.json').read_text())['status'],'PASSED')
            self.assertTrue((result/'prefix-evidence.json').exists())
            env['FAKE_AISBENCH_EXIT']='9';call(test,'try02',expected=1)
            failed=root/'results'/(spec['sub_tasks'][0]['id']+'-try02')
            self.assertEqual(json.loads((failed/'execution.json').read_text())['returncode'],9)
            self.assertFalse((failed/'validated.json').exists())
            # A second failed create must never authorize removing the first service.
            call(stem+'-container.sh','b'*32,expected=1)
            call(stem+'-stop.sh','b'*32);self.assertTrue((state/'exists').exists())
            call(stem+'-stop.sh',token);self.assertFalse((state/'exists').exists())


if __name__=='__main__':unittest.main()
