import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import ast
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import submit_job
from node_identity import node_root as physical_node_root

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"

def execute_statements(file, predicate, scope):
    tree = ast.parse(file.read_text(encoding='utf-8'))
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.stmt) and predicate(n)]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(file), 'exec'), scope)

class SharedStorageRegression(unittest.TestCase):
    def simulate(self, shared, stale=False):
        with tempfile.TemporaryDirectory() as tmp:
            roots = {'a':Path(tmp)/'a', 'b':Path(tmp)/('a' if shared else 'b')}
            uploads=[]
            def local(host, path): return roots[host]/path.lstrip('/')
            def upload(host, destination, archive):
                root=local(host,destination); root.mkdir(parents=True,exist_ok=False)
                (root/'payload').write_text('one package'); uploads.append(host)
            def ssh(host, argv):
                if argv[0]=='touch': local(host,argv[1]).touch()
                elif argv[0]=='rm': local(host,argv[-1]).unlink(missing_ok=True)
                elif argv[0]=='python3': return str(local(host,argv[-1]).is_file())
                return ''
            if stale:
                root=local('a','/job'); root.mkdir(parents=True)
                (root/'.upload-receipt-old-run').touch()
            with patch.object(submit_job,'ssh',ssh),patch.object(submit_job,'upload',upload):
                if stale:
                    with self.assertRaises(FileExistsError): submit_job.upload_destinations([('a','/job'),('b','/job')],Path('bundle'),[[('a','/job'),('b','/job')]] if shared else [])
                else:
                    submit_job.upload_destinations([('a','/job'),('b','/job')],Path('bundle'),[[('a','/job'),('b','/job')]] if shared else [])
                    self.assertEqual(len(uploads),1 if shared else 2)
                    self.assertEqual(list(local('a','/job').glob('.upload-receipt-*')),[])
    def test_shared_physical_root(self): self.simulate(True)
    def test_independent_physical_roots(self): self.simulate(False)
    def test_stale_receipt_rejected(self): self.simulate(True,True)
    def test_node_and_container_paths_do_not_overlap(self):
        values=[]
        for slot,host in [('p','a'),('d','b')]:
            scope=dict(physical_node_root=physical_node_root,Path=Path,hashlib=hashlib,binding={'work_dir':'/shared'},spec={'job_id':'20261104-001','target':{'host':host}},dep={'id':'pd'},slot=slot)
            execute_statements(SCRIPTS/'cluster_runtime.py',lambda n:isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='node_root' for t in n.targets) and 'binding' in ast.unparse(n),scope)
            scope.update(pool={'task_root':'/shared/job','name':'same-container'},execution={})
            execute_statements(SCRIPTS/'task_preflight.py',lambda n:isinstance(n,ast.Assign) and ('host_key =' in ast.unparse(n) or "execution['container_registry'] =" in ast.unparse(n)),scope)
            scope.update(task_root=Path('/shared/job'),s={'target':{'host':host,'container':'same-container'}})
            execute_statements(SCRIPTS/'deploy.py',lambda n:isinstance(n,ast.Assign) and ast.unparse(n).startswith('task_root =') and 'hashlib' in ast.unparse(n),scope)
            values.append((scope['node_root'],scope['execution']['container_registry'],str(scope['task_root'])))
        for index in range(3): self.assertNotEqual(values[0][index],values[1][index])
    def test_warmup_and_formal_command_selection(self):
        source=SCRIPTS.parent/'assets/aisbench-prefix/aisbench_test.py'
        tree=ast.parse(source.read_text(encoding='utf-8')); calls=[]
        scope=dict(test_accuracy=False,OUTPUT_DIR='./outputs',DEFAULT_PERFORMANCE_TEST='stable_stage')
        funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='generate_aisbench_command']
        exec(compile(ast.Module(body=funcs,type_ignores=[]),str(source),'exec'),scope)
        self.assertIn('--summarizer default_perf',scope['generate_aisbench_command']('default_perf'))
        self.assertIn('--summarizer stable_stage',scope['generate_aisbench_command']('stable_stage'))
        warmups=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_benchmark_run' and 'default_perf' in ast.unparse(n)]
        self.assertEqual(len(warmups),1)


if __name__=='__main__': unittest.main()
