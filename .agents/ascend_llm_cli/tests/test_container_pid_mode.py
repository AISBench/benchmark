import unittest

from deploy_eval_v2.container_baseline import docker_run_args


class ContainerPidModeTests(unittest.TestCase):
    def test_private_uses_docker_default_namespace(self):
        args = docker_run_args({'privileged': False, 'pid_mode': 'private'})
        self.assertFalse(any(arg.startswith('--pid') for arg in args))
        self.assertIn('--privileged=false', args)

    def test_host_and_omitted_keep_host_namespace(self):
        for container in ({'privileged': True}, {'privileged': True, 'pid_mode': 'host'}):
            with self.subTest(container=container):
                self.assertEqual(docker_run_args(container).count('--pid=host'), 1)
