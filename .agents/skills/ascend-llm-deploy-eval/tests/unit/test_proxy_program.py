import copy,tempfile,unittest,shlex
from pathlib import Path
from test_compact_entry import inputs
import run as entry

class ProxyProgramTests(unittest.TestCase):
    def test_two_deployments_same_node_use_distinct_programs(self):
        spec,bindings=inputs();first=spec['deployments'][0];second=copy.deepcopy(first);second['id']='other'
        spec['deployments'].append(second)
        sub=copy.deepcopy(spec['sub_tasks'][0]);sub.update(id='second',deployment_id='other');spec['sub_tasks'].append(sub)
        node=bindings['nodes']['node_a'];node['deployments']['other']=copy.deepcopy(node['deployments'][first['id']])
        node['paths'].update(reference_proxy='/reference/proxy.py',image_proxy='/image/proxy.py')
        for dep,key in ((first,'reference_proxy'),(second,'image_proxy')):
            proxy=next(i for i in dep['instances'] if i['role']=='proxy');proxy['launch'].update(proxy_program_key=key,args={'workers':2,'max-retries':4})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'job';entry.generate(spec,bindings,root)
            commands=[]
            for dep in (first,second):
                proxy=next(i for i in dep['instances'] if i['role']=='proxy')
                text=(root/'scripts'/dep['id']/('node_a-'+proxy['id']+'.sh')).read_text()
                commands.append(shlex.split(next(line for line in text.splitlines() if line.startswith('exec python3 '))))
            self.assertEqual(commands[0][2],'/reference/proxy.py');self.assertEqual(commands[1][2],'/image/proxy.py')
            self.assertEqual(commands[0][3:],commands[1][3:])
            self.assertIn('--prefiller-hosts',commands[0]);self.assertIn('--decoder-ports',commands[0]);self.assertIn('--workers',commands[0]);self.assertIn('--max-retries',commands[0])
    def test_default_and_invalid_bindings(self):
        spec,bindings=inputs();proxy=next(i for i in spec['deployments'][0]['instances'] if i['role']=='proxy')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'default';entry.generate(spec,bindings,root)
            self.assertTrue(any('exec python3 /proxy.py ' in f.read_text() for f in (root/'scripts').rglob('*.sh')))
            for key in ('missing','bad.key','bad-key','', '$(touch pwned)'):
                with self.subTest(key=key):
                    proxy['launch']['proxy_program_key']=key
                    with self.assertRaises(ValueError):entry.generate(spec,bindings,Path(tmp)/'invalid')
                    self.assertFalse((Path(tmp)/'invalid').exists())
    def test_model_role_rejected(self):
        spec,bindings=inputs();model=next(i for i in spec['deployments'][0]['instances'] if i['role']!='proxy');model['launch']['proxy_program_key']='proxy_program'
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'only valid for proxy'):entry.generate(spec,bindings,Path(tmp)/'job')
