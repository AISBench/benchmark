import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from test_compact_entry import inputs
import run as entry
from mount_profiles import resolve_mounts

class MountProfileTests(unittest.TestCase):
    def test_override_and_duplicates(self):
        binding={'mount_profiles':['ascend-runtime'], 'mounts':[{'source':'/custom/driver','target':'/usr/local/Ascend/driver','read_only':False}]}
        mounts=resolve_mounts(binding)
        self.assertEqual([m for m in mounts if m['target']=='/usr/local/Ascend/driver'],binding['mounts'])
        self.assertTrue(next(m for m in mounts if m['target']=='/usr/local/dcmi')['read_only'])
        binding['mounts']*=2
        with self.assertRaises(ValueError):resolve_mounts(binding)
        with self.assertRaises(ValueError):resolve_mounts({'mount_profiles':['unknown']})
        self.assertEqual(resolve_mounts({}),[])

    def test_generation_checks_sources_before_create(self):
        spec,bindings=inputs()
        bindings['nodes']['node_a']['mount_profiles']=['ascend-runtime','ascend-hixl']
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'job'
            entry.generate(spec,bindings,root)
            script=next((root/'scripts').rglob('*-container.sh')).read_text()
            for path in ('/usr/local/Ascend/driver','/usr/local/dcmi','/etc/hixlep'):
                self.assertLess(script.index('test -e '+path),script.index('docker run '))
                self.assertIn(path+':'+path+':ro',script)
            self.assertTrue(script.startswith('#!/usr/bin/env bash\nset -e'))

    @unittest.skipUnless(os.name == 'posix', 'Requires Linux bash')
    def test_missing_profile_source_never_creates_container(self):
        spec,bindings=inputs()
        node=bindings['nodes']['node_a']
        node['mount_profiles']=['ascend-runtime']
        with tempfile.TemporaryDirectory() as tmp:
            temp=Path(tmp)
            missing=str(temp/'absent-driver')
            node['mounts']=[{'source':missing,'target':'/usr/local/Ascend/driver'}, {'source':tmp,'target':'/usr/local/dcmi'}]
            # Use a portable existing device; no real Docker or NPU operation occurs.
            spec['deployments'][0]['runtime']['container']['devices']=['/dev/null']
            root=temp/'job'
            entry.generate(spec,bindings,root)
            marker=temp/'created'
            docker=temp/'docker'
            docker.write_text('#!/bin/sh\ncase "$1" in\n info) exit 0;;\n inspect) exit 1;;\n run) touch '+str(marker)+';;\nesac\n')
            docker.chmod(0o755)
            script=next((root/'scripts').rglob('*-container.sh'))
            result=subprocess.run(['bash','-x',str(script)],env={**os.environ,'PATH':str(temp)+':'+os.environ['PATH']},capture_output=True)
            self.assertEqual(result.returncode,1, result.stderr.decode())
            self.assertEqual(result.stderr.decode().splitlines()[-1], '+ test -e '+missing)
            self.assertFalse(marker.exists())
