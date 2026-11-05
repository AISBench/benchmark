import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from cluster_runtime import higher_concurrency_blocker


class BranchTests(unittest.TestCase):
    def test_failure_blocks_only_higher_concurrency_same_length_and_deployment(self):
        cases=[{'id':'a','input_tokens':16,'concurrency':4},
               {'id':'b','input_tokens':16,'concurrency':16},
               {'id':'c','input_tokens':64,'concurrency':16}]
        tasks=[{'id':c['id'],'case_id':c['id'],'deployment_id':'shape-a',
                'on_failure':'skip_higher_concurrency'} for c in cases]
        spec={'cases':cases,'sub_tasks':tasks};results={'a':{'status':'FAILED'}}
        self.assertEqual(higher_concurrency_blocker(spec,tasks[1],results),'a')
        self.assertIsNone(higher_concurrency_blocker(spec,tasks[2],results))
        self.assertIsNone(higher_concurrency_blocker(spec,dict(tasks[1],deployment_id='shape-b'),results))
        self.assertIsNone(higher_concurrency_blocker(spec,tasks[1],{'a':{'status':'PASSED'}}))
