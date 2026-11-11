import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import ownership_evidence as evidence


class OwnershipEvidenceTests(unittest.TestCase):
    def test_maps_live_owners_and_retains_disappeared_process(self):
        cid = 'a'*64
        def command(argv):
            if argv[0] == 'npu-smi':
                return {'returncode': 0, 'stdout': 'device memory snapshot'}
            if argv[0] == 'fuser':
                self.assertEqual(argv[1:], ['/dev/davinci0', '/dev/davinci4'])
                return {'returncode': 0, 'stdout': '123 456 123'}
            return {'returncode': 0, 'stdout': cid+' owner\n'}
        with tempfile.TemporaryDirectory() as tmp, patch.object(evidence, 'command', side_effect=command):
            proc = Path(tmp)
            (proc/'123').mkdir()
            (proc/'123'/'cgroup').write_text('0::/docker/'+cid)
            result = evidence.snapshot([0, 4], proc)
        self.assertEqual([p['pid'] for p in result['processes']], [123, 456])
        self.assertEqual(result['processes'][0]['containers'], {cid: 'owner'})
        self.assertEqual(result['processes'][1]['error'], 'FileNotFoundError')

    def test_missing_tools_remain_explicit_evidence(self):
        with patch.object(evidence.subprocess, 'run', side_effect=FileNotFoundError):
            result = evidence.snapshot([0])
        self.assertEqual(result['npu_smi']['error'], 'FileNotFoundError')
        self.assertEqual(result['processes'], [])

    def test_commands_are_bounded(self):
        with patch.object(evidence.subprocess, 'run', side_effect=evidence.subprocess.TimeoutExpired('npu-smi', 3)) as run:
            result = evidence.command(['npu-smi', 'info'])
        self.assertEqual(result, {'error': 'TimeoutExpired'})
        self.assertEqual(run.call_args.kwargs['timeout'], 3)
