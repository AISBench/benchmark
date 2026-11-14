import unittest
from unittest.mock import patch
from deploy_eval_v2.launcher import launch_argv,ensure_started
from deploy_eval_v2.protocol import ProtocolError

def identity():
    return dict(protocol_version='1.0',task_id='t',task_root='/work/t',runtime_sha256='a'*64,bundle_sha256='b'*64,nonce='c'*64)

class LauncherTests(unittest.TestCase):
    def test_direct_python_argv(self):
        argv=launch_argv(identity(),'/python','/work/t/identity.json',fake=True)
        self.assertEqual(argv[0],'/python')
        self.assertIn('--fake',argv)
        self.assertNotIn('systemd-run',argv)

    def test_uncertain_start_never_restarts(self):
        with patch('pathlib.Path.exists',return_value=False),patch('deploy_eval_v2.launcher.spawn_owned') as spawn:
            with self.assertRaisesRegex(ProtocolError,'prior launch'):
                ensure_started(identity(),'/python','/work/t/identity.json',fake=True,allow_start=False)
            spawn.assert_not_called()
