from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[2]/'scripts'


class StageEntryTests(unittest.TestCase):
    def invoke(self, root):
        # Retired dispatch must reject even if a valid historical plan exists.
        code = (
            'import runpy,sys,types; '
            f'sys.path.insert(0,{str(SCRIPTS)!r}); '
            'm=types.ModuleType("task_runtime"); '
            'm.stage_entry=lambda stage,root,bindings: '
            '{"status":"PASSED","message":"dispatched:"+stage}; '
            'sys.modules["task_runtime"]=m; '
            f'sys.argv=["preflight.py","--run-dir",{str(root)!r}]; '
            'from stage_runtime import entry; sys.exit(entry("preflight"))'
        )
        return subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)

    def test_common_plan_cannot_reactivate_retired_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'common').mkdir()
            (root/'common/plan.json').write_text('{}')
            result = self.invoke(root)
            self.assert_retired(result)
            self.assertEqual(sorted(p.relative_to(root).as_posix() for p in root.rglob('*')),
                             ['common', 'common/plan.json'])

    def test_legacy_root_plan_does_not_mask_missing_task_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'plan.json').write_text('{}')
            result = self.invoke(root)
            self.assert_retired(result)
            self.assertEqual([p.name for p in root.iterdir()], ['plan.json'])

    def test_relative_run_directory_is_rejected(self):
        result = self.invoke(Path('relative-job'))
        self.assert_retired(result)

    def assert_retired(self, result):
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Retired lifecycle entry', result.stderr)
        self.assertIn('scripts/run.py', result.stderr)
        self.assertEqual(result.stdout, '')
