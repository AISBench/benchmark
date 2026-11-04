"""Run with python3 -m unittest discover -s scripts -p test_container_baseline.py."""
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from container_baseline import baseline_mounts, docker_run_args, DEVICES, MOUNT_PATHS


class ContainerBaselineTests(unittest.TestCase):
    def test_required_mounts_and_idempotency(self):
        mounts = baseline_mounts([], exists=lambda p: True)
        self.assertEqual(mounts, baseline_mounts(mounts, exists=lambda p: True))
        self.assertEqual(len(mounts), len({m['target'] for m in mounts}))
        self.assertIn('/etc/hccl_rootinfo.json', [m['target'] for m in mounts])

    def test_missing_rootinfo_blocks(self):
        with self.assertRaisesRegex(ValueError, '/etc/hccl_rootinfo.json'):
            baseline_mounts([], exists=lambda p: p != '/etc/hccl_rootinfo.json')

    def test_explicit_source_override(self):
        result = baseline_mounts([dict(source='/custom/rootinfo', target='/etc/hccl_rootinfo.json', read_only=True)], exists=lambda p: p != '/etc/hccl_rootinfo.json')
        self.assertEqual(result[0]['source'], '/custom/rootinfo')
        self.assertTrue(result[0]['read_only'])

    def test_container_overrides(self):
        args = docker_run_args(dict(shm_size='64g', privileged=False, user='1000', pid_mode='private'))
        self.assertIn('--privileged=false', args)
        self.assertIn('--shm-size=64g', args)
        self.assertIn('--pid=private', args)
        self.assertIn('1000', args)

    def test_defaults_frozen_without_overwriting(self):
        from plan_run import normalize
        result = normalize(dict(schema_version='3.0', deployments=[dict(runtime=dict(image='test', container=dict(shm_size='64g', privileged=False)))], cases=[], sub_tasks=[]))
        container = result['deployments'][0]['runtime']['container']
        self.assertEqual(container['shm_size'], '64g')
        self.assertFalse(container['privileged'])
        self.assertEqual(container['devices'], DEVICES)
        self.assertEqual(container['pid_mode'], 'host')

    def test_read_only_override_and_workspace_preserved(self):
        original = [dict(source='/etc/hccl_rootinfo.json', target='/etc/hccl_rootinfo.json', read_only=True),
                    dict(source='/work/job', target='/work/job', read_only=False)]
        result = baseline_mounts(original, exists=lambda p: True)
        self.assertTrue(result[0]['read_only'])
        self.assertIn(original[1], result)

    def test_host_execution_flags(self):
        args = docker_run_args(dict(shm_size='500g'))
        for flag in ['--pid=host', '--net=host', '--privileged=true', '--shm-size=500g', '-it']:
            self.assertIn(flag, args)
        self.assertEqual(len(DEVICES), 12)


if __name__ == '__main__':
    unittest.main()
