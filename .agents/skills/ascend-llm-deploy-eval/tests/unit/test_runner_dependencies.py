import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import shutil
from stage_runtime import PACKAGE

class RunnerDependencies(unittest.TestCase):
    def test_storage_check_in_isolated_runner(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'scripts').mkdir()
            shutil.copytree(PACKAGE/'scripts', root/'common/skill-snapshot/scripts')
            env = {**os.environ, 'PYTHONPATH': str(root/'common/skill-snapshot/scripts')}
            result = subprocess.run([sys.executable, '-c',
                'import case_runner; from disk_guard import check; check(".", 1)'],
                cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
