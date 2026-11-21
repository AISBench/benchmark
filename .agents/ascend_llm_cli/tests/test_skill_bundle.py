import tempfile
import unittest
from pathlib import Path
from deploy_eval_v2.skill_bundle import NAMES, inventory, install
from deploy_eval_v2.protocol import ProtocolError


class SkillBundleTests(unittest.TestCase):
    source=Path(__file__).resolve().parents[2]/'skills'

    def test_all_five_install_and_replay_preserves_stable(self):
        with tempfile.TemporaryDirectory() as folder:
            stable=Path(folder)/'ascend-llm-deploy-eval'
            stable.mkdir()
            (stable/'SKILL.md').write_text('untouched')
            request=dict(source=str(self.source),destination=folder)
            self.assertTrue(install(request)['installed'])
            self.assertTrue(install(request)['installed'])
            self.assertEqual(inventory(folder),inventory(self.source))
            self.assertEqual((stable/'SKILL.md').read_text(),'untouched')

    def test_conflict_prevents_partial_install(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/NAMES[-1]
            target.mkdir()
            (target/'SKILL.md').write_text('user edit')
            with self.assertRaisesRegex(ProtocolError,'REQUEST_CONFLICT'):
                install(dict(source=str(self.source),destination=folder))
            self.assertFalse((Path(folder)/NAMES[0]).exists())
            self.assertEqual((target/'SKILL.md').read_text(),'user edit')

    def test_missing_dependency_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ProtocolError,'incomplete'):
                inventory(folder)
