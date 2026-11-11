import unittest
from test_compact_entry import inputs
from task_model import normalize

class CleanupDefaultsTests(unittest.TestCase):
    def test_partial_cleanup_preserves_explicit_subtask(self):
        spec,_=inputs()
        spec['cleanup']={'task_end':'stop_container'}
        spec['sub_tasks'][0]['after_case']='stop_container'
        result=normalize(spec)
        self.assertEqual(result['cleanup']['after_case'],'keep_service')
        self.assertEqual(result['sub_tasks'][0]['after_case'],'stop_container')
        self.assertNotIn('after_case',spec['cleanup'])

    def test_partial_cleanup_supplies_inherited_default(self):
        spec,_=inputs()
        spec['cleanup']={}
        spec['sub_tasks'][0].pop('after_case',None)
        result=normalize(spec)
        self.assertEqual(result['sub_tasks'][0]['after_case'],'keep_service')
        self.assertEqual(result['cleanup']['task_end'],'stop_container')
