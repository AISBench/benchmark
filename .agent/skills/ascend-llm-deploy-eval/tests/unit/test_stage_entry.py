import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[2]/'scripts'


class StageEntryTests(unittest.TestCase):
    def invoke(self, root):
        # Exercise the actual preflight CLI, isolating remote dispatch only.
        code = (
            'import runpy,sys,types; '
            f'sys.path.insert(0,{str(SCRIPTS)!r}); '
            'm=types.ModuleType("task_runtime"); '
            'm.stage_entry=lambda stage,root,bindings: '
            '{"status":"PASSED","message":"dispatched:"+stage}; '
            'sys.modules["task_runtime"]=m; '
            f'sys.argv=["preflight.py","--run-dir",{str(root)!r}]; '
            f'runpy.run_path({str(SCRIPTS/"preflight.py")!r},run_name="__main__")'
        )
        return subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)

    def test_common_plan_reaches_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'common').mkdir()
            (root/'common/plan.json').write_text('{}')
            result = self.invoke(root)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertEqual(json.loads(result.stdout)['message'], 'dispatched:preflight')

    def test_legacy_root_plan_does_not_mask_missing_task_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'plan.json').write_text('{}')
            result = self.invoke(root)
            self.assertEqual(result.returncode, 2, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report['status'], 'BLOCKED')
            self.assertIn(str(root/'common/plan.json'), report['message'])

    def test_relative_run_directory_is_rejected(self):
        result = self.invoke(Path('relative-job'))
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('Absolute run-dir required', json.loads(result.stdout)['message'])
