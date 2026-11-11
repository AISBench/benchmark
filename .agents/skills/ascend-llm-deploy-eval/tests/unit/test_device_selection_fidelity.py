import unittest
from test_topology import deployment, bindings
from topology import compile_deployment

class DeviceSelectionTests(unittest.TestCase):
    def dep(self):
        dep=deployment()
        for item in dep['instances']:
            if item['role']!='proxy':item['launch'].update(preserve_reference=True,engine_id=item['id'])
        return dep
    def command(self,dep):
        return compile_deployment(dep,bindings())['groups']['a/model']['commands'][0]
    def test_explicit_cli_retained_without_injected_env(self):
        dep=self.dep();dep['instances'][0]['launch']['args']['device-ids']='0,1,2,3'
        cmd=self.command(dep)
        self.assertEqual(cmd['argv'][cmd['argv'].index('--device-ids')+1],'0,1,2,3')
        self.assertNotIn('ASCEND_RT_VISIBLE_DEVICES',cmd['env'])
    def test_explicit_env_retained_without_cli(self):
        for key in ('ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES'):
            dep=self.dep();dep['instances'][0]['launch']['env']={key:'0,1,2,3'}
            cmd=self.command(dep);self.assertEqual(cmd['env'],{key:'0,1,2,3'})
            self.assertNotIn('--device-ids',cmd['argv'])
    def test_conflicting_selection_rejected(self):
        for kind in ('cli','ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES'):
            dep=self.dep();launch=dep['instances'][0]['launch']
            if kind=='cli':launch['args']['device-ids']='4,5,6,7'
            else:launch['env']={kind:'4,5,6,7'}
            with self.assertRaisesRegex(ValueError,'device selection conflicts'):
                self.command(dep)
    def test_matching_both_are_retained(self):
        dep=self.dep();launch=dep['instances'][0]['launch']
        launch['args']['device-ids']='0,1,2,3';launch['env']={'ASCEND_RT_VISIBLE_DEVICES':'0,1,2,3'}
        cmd=self.command(dep);self.assertIn('--device-ids',cmd['argv'])
        self.assertEqual(cmd['env']['ASCEND_RT_VISIBLE_DEVICES'],'0,1,2,3')
if __name__=='__main__':unittest.main()
