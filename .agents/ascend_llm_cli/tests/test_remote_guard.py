from pathlib import Path
import tempfile
import unittest
from deploy_eval_v2.remote_guard import HostGuard, require_owned_container, exclusive
from deploy_eval_v2.protocol import ProtocolError


class RemoteGuardTests(unittest.TestCase):
    def test_precreate_cleanup_requires_absence_of_durable_effects(self):
        with tempfile.TemporaryDirectory() as folder:
            guard=HostGuard(folder,'t','a'*64)
            guard.claim(['npu:0'])
            self.assertTrue(guard.release_unstarted(['npu:0'])['claims_released'])
            guard.claim(['npu:0'])
            def lost():
                raise OSError('create outcome lost')
            with self.assertRaises(OSError):
                guard.command('create1',{'kind':'create'},lost)
            with self.assertRaisesRegex(ProtocolError,'prior resource effect'):
                guard.release_unstarted(['npu:0'])
            with self.assertRaisesRegex(ProtocolError,'RESOURCE_BUSY'):
                HostGuard(folder,'other','b'*64).claim(['npu:0'])
    def test_cross_task_claims_survive_reconstruction(self):
        with tempfile.TemporaryDirectory() as folder:
            first = HostGuard(folder, 'a', 'a'*64)
            first.claim(['npu:0', 'tcp:18000'])
            HostGuard(folder, 'a', 'a'*64).claim(['npu:0'])
            second = HostGuard(folder, 'b', 'b'*64)
            with self.assertRaisesRegex(ProtocolError, 'RESOURCE_BUSY'):
                second.claim(['npu:0'])
            with self.assertRaises(ProtocolError):
                first.release_claims(['npu:0'], resources_verified_absent=False)
            first.release_claims(['npu:0'], resources_verified_absent=True)
            second.claim(['npu:0'])

    def test_command_loss_never_reexecutes(self):
        with tempfile.TemporaryDirectory() as folder:
            guard = HostGuard(folder, 'a', 'a'*64)
            calls = []
            def action():
                calls.append(1)
                return {'container_id': 'abc'}
            self.assertEqual(guard.command('create', {'image': 'x'}, action), {'container_id': 'abc'})
            self.assertEqual(guard.command('create', {'image': 'x'}, action), {'container_id': 'abc'})
            self.assertEqual(len(calls), 1)
            with self.assertRaisesRegex(ProtocolError, 'REQUEST_CONFLICT'):
                guard.command('create', {'image': 'y'}, action)
            def crash():
                raise OSError('lost after side effect')
            with self.assertRaises(OSError):
                guard.command('unknown', {}, crash)
            with self.assertRaisesRegex(ProtocolError, 'RECOVERY_REQUIRED'):
                guard.command('unknown', {}, action)
            self.assertEqual(len(calls), 1)

    def test_exclusive_lock_and_ownership(self):
        with tempfile.TemporaryDirectory() as folder:
            with exclusive(Path(folder)/'lock'):
                with self.assertRaises(ProtocolError):
                    with exclusive(Path(folder)/'lock'):
                        self.fail('must not acquire')
        inspect = {'Id': 'id', 'Config': {'Labels': {'bench.task': 't', 'bench.execution': 'token'}}}
        require_owned_container(inspect, 't', 'token', 'id')
        for task, token, cid in [('other', 'token', 'id'), ('t', 'old', 'id'), ('t', 'token', 'reused-name')]:
            with self.assertRaises(ProtocolError):
                require_owned_container(inspect, task, token, cid)

    def test_foreign_claim_does_not_partially_unlock_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            first = HostGuard(folder, 'a', 'a'*64)
            second = HostGuard(folder, 'b', 'b'*64)
            first.claim(['npu:0'])
            second.claim(['npu:1'])
            with self.assertRaises(ProtocolError):
                first.release_claims(['npu:0', 'npu:1'], resources_verified_absent=True)
            with self.assertRaisesRegex(ProtocolError, 'RESOURCE_BUSY'):
                second.claim(['npu:0'])
