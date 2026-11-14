from pathlib import Path
import tempfile
import unittest
import zipfile
from deploy_eval_v2.publication import build_archive, install_archive, verify_runtime
from deploy_eval_v2.protocol import ProtocolError


class PublicationTests(unittest.TestCase):
    def build(self, root):
        package = root/'source'
        package.mkdir()
        (package/'__init__.py').write_text('')
        (package/'server.py').write_text('print("fixture")')
        archive = root/'runtime.zip'
        return archive, build_archive(package, archive)

    def test_publish_reuse_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive, evidence = self.build(root)
            target = root/'published'
            self.assertFalse(install_archive(archive, target, evidence['runtime_sha256'])['reused'])
            self.assertTrue(install_archive(archive, target, evidence['runtime_sha256'])['reused'])
            (target/'deploy_eval_v2/server.py').write_text('changed')
            with self.assertRaises(ProtocolError):
                install_archive(archive, target, evidence['runtime_sha256'])
            self.assertEqual((target/'deploy_eval_v2/server.py').read_text(), 'changed')

    def test_traversal_and_unmanifested_files_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive, evidence = self.build(root)
            with zipfile.ZipFile(archive, 'a') as stream:
                stream.writestr('../escape.py', 'bad')
            with self.assertRaises(ProtocolError):
                install_archive(archive, root/'published', evidence['runtime_sha256'])
            self.assertFalse((root/'published').exists())
            self.assertFalse((root/'escape.py').exists())

    def test_changed_digest_rejects_before_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive, _ = self.build(root)
            with self.assertRaises(ProtocolError):
                install_archive(archive, root/'published', '0'*64)
            self.assertFalse((root/'published').exists())
