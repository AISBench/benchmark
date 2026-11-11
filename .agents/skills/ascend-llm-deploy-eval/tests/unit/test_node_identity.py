import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import unittest
from node_identity import node_job_id,node_root
from cluster_runtime import container_pools
class NodeIdentityTests(unittest.TestCase):
    def test_parent_archive_and_stable_identity(self):
        parent='20261104-001'
        self.assertEqual(node_job_id(parent,'node_p'),'20261104-001-node-node_p')
        a=node_root({'work_dir':'/shared'},parent,'p')
        b=node_root({'work_dir':'/alias'},parent,'d')
        self.assertEqual(a.relative_to('/shared').parts[0],parent)
        self.assertNotEqual(a.name,b.name)
        self.assertEqual(a.name,'p')
    def test_invalid_node_id(self):
        for name in ('../p','x'*33,''):
            with self.assertRaises(ValueError): node_job_id('20261104-001',name)
    def test_container_pools_keep_parent_and_isolate_nodes(self):
        dep={'id':'pd','runtime':{},'model':{'id':'m'},'instances':[{'placement':{'node':n,'devices':[0]}} for n in ['p','d']]}
        bindings={'nodes':{n:{'host':n,'work_dir':'/shared','model_paths':{'m':'/weights'}} for n in ['p','d']}}
        pools=container_pools({'job_id':'20261104-001','deployments':[dep]},bindings)
        p,d=pools[('pd','p')],pools[('pd','d')]
        self.assertEqual(p['parent_job_id'],d['parent_job_id'])
        for key in ['name','job_id','node_root']: self.assertNotEqual(p[key],d[key])
        self.assertEqual(p['task_root'],d['task_root'])
    def test_ipc_path_budget(self):
        from preflight import check_ipc_path
        self.assertLessEqual(check_ipc_path(),107)
        with self.assertRaises(ValueError): check_ipc_path('/'+'x'*100)
        with self.assertRaises(ValueError): check_ipc_path('/'+'界'*30)
if __name__=='__main__': unittest.main()
