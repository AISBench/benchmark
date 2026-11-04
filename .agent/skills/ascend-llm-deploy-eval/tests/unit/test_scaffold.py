import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SKILL = Path(__file__).resolve().parents[2]

class ScaffoldTests(unittest.TestCase):
    def test_copied_skill_blocks_without_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            copied = root / "installed"
            shutil.copytree(SKILL, copied)
            for stage in ("plan_run", "preflight", "deploy", "evaluate", "finalize"):
                result = subprocess.run([sys.executable, str(copied / "scripts" / (stage + ".py")),
                                         "--run-dir", str(root / "run")], cwd=root,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(json.loads(result.stdout)["status"], "BLOCKED")
            self.assertFalse((root / "run").exists())

if __name__ == "__main__":
    unittest.main()
