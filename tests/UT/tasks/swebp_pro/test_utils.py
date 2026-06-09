"""Unit tests for ais_bench.benchmark.tasks.swebench_pro.utils"""
import unittest
import tempfile
import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ais_bench.benchmark.tasks.swebench_pro import utils as utils_module


class TestStripBinaryHunks(unittest.TestCase):
    """Test strip_binary_hunks function."""

    @classmethod
    def setUpClass(cls):
        cls.utils = utils_module

    def test_removes_binary_hunks(self):
        patch_with_binary = """diff --git a/file.py b/file.py
index abc123..def456 100644
--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
-old
+new
diff --git a/image.png b/image.png
index 123abc..456def 100644
Binary files a/image.png and b/image.png differ
"""
        result = self.utils.strip_binary_hunks(patch_with_binary)
        self.assertIn("diff --git a/file.py", result)
        self.assertNotIn("Binary files", result)
        self.assertNotIn("image.png", result)

    def test_keeps_non_binary_patches(self):
        patch_text = """diff --git a/file.py b/file.py
index abc123..def456 100644
--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
-old
+new
"""
        result = self.utils.strip_binary_hunks(patch_text)
        self.assertEqual(result, patch_text)

    def test_handles_empty_input(self):
        result = self.utils.strip_binary_hunks("")
        self.assertEqual(result, "")

    def test_handles_patch_without_binary(self):
        patch_text = "simple patch content"
        result = self.utils.strip_binary_hunks(patch_text)
        self.assertEqual(result, patch_text)

    def test_handles_multiple_binary_hunks(self):
        patch_with_binaries = """diff --git a/file1.py b/file1.py
index abc123..def456 100644
--- a/file1.py
+++ b/file1.py
@@ -1 +1 @@
-old1
+new1
diff --git a/bin1.bin b/bin1.bin
Binary files a/bin1.bin and b/bin1.bin differ
diff --git a/file2.py b/file2.py
index 123abc..456def 100644
--- a/file2.py
+++ b/file2.py
@@ -1 +1 @@
-old2
+new2
diff --git a/bin2.bin b/bin2.bin
Binary files a/bin2.bin and b/bin2.bin differ
"""
        result = self.utils.strip_binary_hunks(patch_with_binaries)
        self.assertIn("file1.py", result)
        self.assertIn("file2.py", result)
        self.assertNotIn("bin1.bin", result)
        self.assertNotIn("bin2.bin", result)

    def test_removes_git_binary_patch(self):
        patch_with_git_binary = """diff --git a/file.py b/file.py
index abc123..def456 100644
--- a/file.py
+++ b/file.py
@@ -1 +1 @@
-old
+new
diff --git a/bin.bin b/bin.bin
GIT binary patch
literal 123
abcdef1234567890
"""
        result = self.utils.strip_binary_hunks(patch_with_git_binary)
        self.assertIn("diff --git a/file.py", result)
        self.assertNotIn("GIT binary patch", result)

    def test_skips_empty_sections(self):
        patch_with_empty = """
diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1 +1 @@
-old
+new

"""
        result = self.utils.strip_binary_hunks(patch_with_empty)
        self.assertIn("diff --git a/file.py", result)


class TestGetDockerhubImageUri(unittest.TestCase):
    """Test get_dockerhub_image_uri function."""

    @classmethod
    def setUpClass(cls):
        cls.utils = utils_module

    def test_generates_correct_uri(self):
        instance = {
            "instance_id": "instance_test_123",
            "repo": "test/repo",
        }
        result = self.utils.get_dockerhub_image_uri(instance)
        self.assertIn("test.repo", result)
        self.assertTrue(result.startswith("jefzda/sweap-images:"))

    def test_handles_element_hq_special_case(self):
        instance = {
            "instance_id": "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan",
            "repo": "element-hq/element-web",
        }
        result = self.utils.get_dockerhub_image_uri(instance)
        self.assertIn("element-hq.element-web", result)
        self.assertTrue(result.startswith("jefzda/sweap-images:"))

    def test_handles_element_hq_other_case(self):
        instance = {
            "instance_id": "instance_element-hq__element-web-abc123-vnan",
            "repo": "element-hq/element-web",
        }
        result = self.utils.get_dockerhub_image_uri(instance)
        self.assertIn("element-hq.element", result)
        self.assertNotIn("-vnan", result)
        self.assertTrue(result.startswith("jefzda/sweap-images:"))

    def test_strips_vnan_suffix_for_other_repos(self):
        instance = {
            "instance_id": "instance_other_repo-abc123-vnan",
            "repo": "owner/repo",
        }
        result = self.utils.get_dockerhub_image_uri(instance)
        self.assertNotIn("-vnan", result)
        self.assertTrue(result.startswith("jefzda/sweap-images:"))

    def test_keeps_non_vnan_hash(self):
        instance = {
            "instance_id": "instance_owner_repo-abc123",
            "repo": "owner/repo",
        }
        result = self.utils.get_dockerhub_image_uri(instance)
        self.assertIn("owner.repo", result)
        self.assertTrue(result.startswith("jefzda/sweap-images:"))

    def test_truncates_long_tags(self):
        long_hash = "a" * 200
        instance = {
            "instance_id": f"instance_test_{long_hash}",
            "repo": "test/repo",
        }
        result = self.utils.get_dockerhub_image_uri(instance)
        tag = result.split(":")[-1]
        self.assertLessEqual(len(tag), 128)


class TestMergeNestedDicts(unittest.TestCase):
    """Test merge_nested_dicts function."""

    @classmethod
    def setUpClass(cls):
        cls.utils = utils_module

    def test_merges_simple_dicts(self):
        dict1 = {"a": 1, "b": 2}
        dict2 = {"c": 3, "d": 4}
        result = self.utils.merge_nested_dicts(dict1, dict2)
        self.assertEqual(result, {"a": 1, "b": 2, "c": 3, "d": 4})

    def test_merges_nested_dicts(self):
        dict1 = {"a": {"x": 1}, "b": 2}
        dict2 = {"a": {"y": 2}, "c": 3}
        result = self.utils.merge_nested_dicts(dict1, dict2)
        self.assertEqual(result["a"], {"x": 1, "y": 2})
        self.assertEqual(result["b"], 2)
        self.assertEqual(result["c"], 3)

    def test_second_dict_overrides_first(self):
        dict1 = {"a": 1, "b": {"x": 1}}
        dict2 = {"a": 2, "b": {"x": 2}}
        result = self.utils.merge_nested_dicts(dict1, dict2)
        self.assertEqual(result["a"], 2)
        self.assertEqual(result["b"]["x"], 2)

    def test_handles_empty_dicts(self):
        result = self.utils.merge_nested_dicts({}, {})
        self.assertEqual(result, {})


class TestEvalWithDocker(unittest.TestCase):
    """Test eval_with_docker function."""

    @classmethod
    def setUpClass(cls):
        cls.utils = utils_module

    def test_returns_existing_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            uid_dir = os.path.join(tmpdir, "test-123")
            os.makedirs(uid_dir, exist_ok=True)
            output_path = os.path.join(uid_dir, "prefix_output.json")
            expected_output = {"result": "existing"}
            with open(output_path, "w") as f:
                json.dump(expected_output, f)

            instance = {"instance_id": "test-123"}
            result = self.utils.eval_with_docker(
                "test patch",
                instance,
                tmpdir,
                Path(tmpdir),
                Path(tmpdir),
                MagicMock(),
                "prefix",
                MagicMock(),
                3600
            )

            self.assertEqual(result, expected_output)

    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.assemble_workspace_files')
    def test_handles_missing_scripts(self, mock_assemble):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_assemble.side_effect = FileNotFoundError("Script not found")
            mock_logger = MagicMock()

            instance = {"instance_id": "test-123"}
            result = self.utils.eval_with_docker(
                "test patch",
                instance,
                tmpdir,
                Path(tmpdir),
                Path(tmpdir),
                mock_logger,
                "prefix",
                MagicMock(),
                3600
            )

            self.assertIsNone(result)
            mock_logger.error.assert_called_once()


class TestBuildProblemStatement(unittest.TestCase):
    """Test build_problem_statement function."""

    @classmethod
    def setUpClass(cls):
        cls.utils = utils_module

    def test_builds_basic_statement(self):
        row = {"problem_statement": "Fix the bug"}
        result = self.utils.build_problem_statement(row)
        self.assertEqual(result, "Fix the bug")

    def test_includes_requirements(self):
        row = {
            "problem_statement": "Fix the bug",
            "requirements": "pip install pytest"
        }
        result = self.utils.build_problem_statement(row)
        self.assertIn("Fix the bug", result)
        self.assertIn("Requirements:", result)
        self.assertIn("pip install pytest", result)

    def test_includes_interface(self):
        row = {
            "problem_statement": "Fix the bug",
            "interface": "def new_func():"
        }
        result = self.utils.build_problem_statement(row)
        self.assertIn("Fix the bug", result)
        self.assertIn("New interfaces introduced:", result)
        self.assertIn("def new_func():", result)

    def test_includes_both_requirements_and_interface(self):
        row = {
            "problem_statement": "Fix the bug",
            "requirements": "pip install pytest",
            "interface": "def new_func():"
        }
        result = self.utils.build_problem_statement(row)
        self.assertIn("Fix the bug", result)
        self.assertIn("Requirements:", result)
        self.assertIn("New interfaces introduced:", result)


class TestPrepareRun(unittest.TestCase):
    """Test prepare_run function."""

    @classmethod
    def setUpClass(cls):
        cls.utils = utils_module

    def test_creates_new_output_when_not_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            existing, output_path, workspace = self.utils.prepare_run(
                "test-123", tmpdir, "prefix", redo=False
            )

            self.assertIsNone(existing)
            self.assertTrue(os.path.exists(workspace))
            self.assertTrue(os.path.isdir(workspace))

    def test_returns_existing_output_when_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            uid_dir = os.path.join(tmpdir, "test-123")
            os.makedirs(uid_dir, exist_ok=True)
            output_path = os.path.join(uid_dir, "prefix_output.json")
            expected = {"result": "existing"}
            with open(output_path, "w") as f:
                json.dump(expected, f)

            existing, output_path, workspace = self.utils.prepare_run(
                "test-123", tmpdir, "prefix", redo=False
            )

            self.assertEqual(existing, expected)

    def test_redo_ignores_existing_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            uid_dir = os.path.join(tmpdir, "test-123")
            os.makedirs(uid_dir, exist_ok=True)
            output_path = os.path.join(uid_dir, "prefix_output.json")
            with open(output_path, "w") as f:
                json.dump({"result": "existing"}, f)

            existing, output_path, workspace = self.utils.prepare_run(
                "test-123", tmpdir, "prefix", redo=True
            )

            self.assertIsNone(existing)


class TestWriteFilesLocal(unittest.TestCase):
    """Test write_files_local function."""

    @classmethod
    def setUpClass(cls):
        cls.utils = utils_module

    def test_writes_files_to_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = {
                "file1.txt": "content1",
                "file2.py": "print('hello')",
            }
            self.utils.write_files_local(tmpdir, files)

            self.assertTrue(os.path.exists(os.path.join(tmpdir, "file1.txt")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "file2.py")))

            with open(os.path.join(tmpdir, "file1.txt")) as f:
                self.assertEqual(f.read(), "content1")


class TestWritePatchSnapshot(unittest.TestCase):
    """Test write_patch_snapshot function."""

    @classmethod
    def setUpClass(cls):
        cls.utils = utils_module

    def test_writes_patch_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            uid_dir = os.path.join(tmpdir, "test-123")
            os.makedirs(uid_dir, exist_ok=True)

            self.utils.write_patch_snapshot(tmpdir, "test-123", "prefix", "test patch content")

            patch_path = os.path.join(uid_dir, "prefix_patch.diff")
            self.assertTrue(os.path.exists(patch_path))
            with open(patch_path) as f:
                self.assertEqual(f.read(), "test patch content")


class TestSaveEntryscriptCopy(unittest.TestCase):
    """Test save_entryscript_copy function."""

    @classmethod
    def setUpClass(cls):
        cls.utils = utils_module

    def test_saves_entryscript_copy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            uid_dir = os.path.join(tmpdir, "test-123")
            os.makedirs(uid_dir, exist_ok=True)

            self.utils.save_entryscript_copy(tmpdir, "test-123", "prefix", "#!/bin/bash\necho hello")

            script_path = os.path.join(uid_dir, "prefix_entryscript.sh")
            self.assertTrue(os.path.exists(script_path))
            with open(script_path) as f:
                self.assertIn("echo hello", f.read())

    def test_handles_none_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            uid_dir = os.path.join(tmpdir, "test-123")
            os.makedirs(uid_dir, exist_ok=True)

            self.utils.save_entryscript_copy(tmpdir, "test-123", "prefix", None)

            script_path = os.path.join(uid_dir, "prefix_entryscript.sh")
            self.assertTrue(os.path.exists(script_path))


class TestLoadLocalScript(unittest.TestCase):
    """Test load_local_script function."""

    @classmethod
    def setUpClass(cls):
        cls.utils = utils_module

    def test_loads_existing_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script_dir = os.path.join(tmpdir, "test-123")
            os.makedirs(script_dir, exist_ok=True)
            script_path = os.path.join(script_dir, "test.sh")
            with open(script_path, "w") as f:
                f.write("echo hello")

            result = self.utils.load_local_script(tmpdir, "test-123", "test.sh")
            self.assertEqual(result, "echo hello")

    def test_raises_for_missing_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                self.utils.load_local_script(tmpdir, "test-123", "nonexistent.sh")


class TestListSwebenchProImages(unittest.TestCase):
    """Test list_swebench_pro_images function."""

    @classmethod
    def setUpClass(cls):
        cls.utils = utils_module

    def test_lists_matching_images(self):
        mock_client = MagicMock()
        mock_image1 = MagicMock()
        mock_image1.tags = ["jefzda/sweap-images:test1", "other:tag"]
        mock_image2 = MagicMock()
        mock_image2.tags = ["jefzda/sweap-images:test2"]
        mock_image3 = MagicMock()
        mock_image3.tags = ["different:image"]
        mock_client.images.list.return_value = [mock_image1, mock_image2, mock_image3]

        result = self.utils.list_swebench_pro_images(mock_client)

        self.assertEqual(result, {"jefzda/sweap-images:test1", "jefzda/sweap-images:test2"})

    def test_handles_exception(self):
        mock_client = MagicMock()
        mock_client.images.list.side_effect = Exception("Docker error")

        result = self.utils.list_swebench_pro_images(mock_client)

        self.assertEqual(result, set())


class TestDockerImageExistsLocally(unittest.TestCase):
    """Test docker_image_exists_locally function."""

    @classmethod
    def setUpClass(cls):
        cls.utils = utils_module

    @patch('subprocess.run')
    def test_returns_true_when_image_exists(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        result = self.utils.docker_image_exists_locally("test-image")

        self.assertTrue(result)

    @patch('subprocess.run')
    def test_returns_false_when_image_not_exists(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result

        result = self.utils.docker_image_exists_locally("test-image")

        self.assertFalse(result)

    @patch('subprocess.run')
    def test_returns_false_on_exception(self, mock_run):
        mock_run.side_effect = Exception("Docker error")

        result = self.utils.docker_image_exists_locally("test-image")

        self.assertFalse(result)


class TestDockerPullImage(unittest.TestCase):
    """Test docker_pull_image function."""

    @classmethod
    def setUpClass(cls):
        cls.utils = utils_module

    @patch('subprocess.run')
    def test_returns_true_on_success(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        mock_logger = MagicMock()
        result = self.utils.docker_pull_image("test-image", mock_logger)

        self.assertTrue(result)

    @patch('subprocess.run')
    def test_returns_false_on_failure(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result

        mock_logger = MagicMock()
        result = self.utils.docker_pull_image("test-image", mock_logger)

        self.assertFalse(result)


class TestEnsureSwebenchProDockerImages(unittest.TestCase):
    """Test ensure_swebench_pro_docker_images function."""

    @classmethod
    def setUpClass(cls):
        cls.utils = utils_module

    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.docker_pull_image')
    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.docker_image_exists_locally')
    def test_skips_existing_images(self, mock_exists, mock_pull):
        mock_exists.return_value = True
        mock_logger = MagicMock()

        items = [{"image": "image1"}, {"image": "image2"}]
        get_image_name = lambda x: x["image"]

        self.utils.ensure_swebench_pro_docker_images(items, mock_logger, get_image_name)

        self.assertEqual(mock_exists.call_count, 2)
        mock_pull.assert_not_called()

    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.docker_pull_image')
    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.docker_image_exists_locally')
    def test_pulls_missing_images(self, mock_exists, mock_pull):
        mock_exists.side_effect = [False, True, True]
        mock_pull.return_value = True
        mock_logger = MagicMock()

        items = [{"image": "image1"}, {"image": "image2"}]
        get_image_name = lambda x: x["image"]

        self.utils.ensure_swebench_pro_docker_images(items, mock_logger, get_image_name)

        mock_pull.assert_called_once()

    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.docker_pull_image')
    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.docker_image_exists_locally')
    def test_raises_on_pull_failure(self, mock_exists, mock_pull):
        mock_exists.return_value = False
        mock_pull.return_value = False
        mock_logger = MagicMock()

        items = [{"image": "image1"}]
        get_image_name = lambda x: x["image"]

        with self.assertRaises(Exception):
            self.utils.ensure_swebench_pro_docker_images(items, mock_logger, get_image_name)

    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.docker_pull_image')
    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.docker_image_exists_locally')
    def test_handles_duplicate_images(self, mock_exists, mock_pull):
        mock_exists.return_value = True
        mock_logger = MagicMock()

        items = [{"image": "image1"}, {"image": "image1"}, {"image": "image2"}]
        get_image_name = lambda x: x["image"]

        self.utils.ensure_swebench_pro_docker_images(items, mock_logger, get_image_name)

        self.assertEqual(mock_exists.call_count, 2)


class TestRemoveSwebenchProImage(unittest.TestCase):
    """Test remove_swebench_pro_image function."""

    @classmethod
    def setUpClass(cls):
        cls.utils = utils_module

    def test_removes_image_successfully(self):
        mock_client = MagicMock()
        mock_logger = MagicMock()

        self.utils.remove_swebench_pro_image(mock_client, "test-image", mock_logger)

        mock_client.images.remove.assert_called_once_with("test-image", force=True)
        mock_logger.debug.assert_called_once()

    def test_handles_remove_exception(self):
        mock_client = MagicMock()
        mock_client.images.remove.side_effect = Exception("Remove error")
        mock_logger = MagicMock()

        self.utils.remove_swebench_pro_image(mock_client, "test-image", mock_logger)

        mock_logger.warning.assert_called_once()


class TestCleanSwebenchProImages(unittest.TestCase):
    """Test clean_swebench_pro_images function."""

    @classmethod
    def setUpClass(cls):
        cls.utils = utils_module

    def test_no_new_images_to_clean(self):
        mock_client = MagicMock()
        mock_logger = MagicMock()

        with patch('ais_bench.benchmark.tasks.swebench_pro.utils.list_swebench_pro_images') as mock_list:
            with patch('ais_bench.benchmark.tasks.swebench_pro.utils.remove_swebench_pro_image') as mock_remove:
                mock_list.return_value = {"existing1", "existing2"}

                self.utils.clean_swebench_pro_images(mock_client, {"existing1", "existing2"}, mock_logger)

                mock_remove.assert_not_called()
                mock_logger.debug.assert_called_once()


class TestCleanupSwebenchProContainers(unittest.TestCase):
    """Test cleanup_swebench_pro_containers function."""

    @classmethod
    def setUpClass(cls):
        cls.utils = utils_module

    @patch('subprocess.run')
    def test_cleanup_containers_success(self, mock_run):
        mock_result1 = MagicMock()
        mock_result1.returncode = 0
        mock_result1.stdout = "container1\ncontainer2\n"
        mock_result2 = MagicMock()
        mock_result2.returncode = 0
        mock_result2.stdout = ""
        mock_run.side_effect = [mock_result1, MagicMock(), mock_result2]

        self.utils.cleanup_swebench_pro_containers()

        self.assertEqual(mock_run.call_count, 3)

    @patch('subprocess.run')
    def test_cleanup_no_containers(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_run.return_value = mock_result

        self.utils.cleanup_swebench_pro_containers()

    @patch('subprocess.run')
    def test_cleanup_handles_docker_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError("docker not found")

        self.utils.cleanup_swebench_pro_containers()

    @patch('subprocess.run')
    def test_cleanup_handles_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker", timeout=10)

        self.utils.cleanup_swebench_pro_containers()


class TestCollectOutputsLocal(unittest.TestCase):
    """Test collect_outputs_local function."""

    @classmethod
    def setUpClass(cls):
        cls.utils = utils_module

    def test_collects_output_successfully(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_dir = os.path.join(tmpdir, "workspace")
            os.makedirs(workspace_dir, exist_ok=True)
            uid_dir = os.path.join(tmpdir, "test-123")
            os.makedirs(uid_dir, exist_ok=True)

            with open(os.path.join(workspace_dir, "stdout.log"), "w") as f:
                f.write("stdout content")
            with open(os.path.join(workspace_dir, "stderr.log"), "w") as f:
                f.write("stderr content")

            expected_output = {"tests": [], "result": "pass"}
            with open(os.path.join(workspace_dir, "output.json"), "w") as f:
                json.dump(expected_output, f)

            mock_logger = MagicMock()
            result = self.utils.collect_outputs_local(
                workspace_dir, tmpdir, "test-123", "prefix", mock_logger
            )

            self.assertEqual(result, expected_output)
            self.assertTrue(os.path.exists(os.path.join(uid_dir, "prefix_output.json")))

    def test_handles_missing_output_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_dir = os.path.join(tmpdir, "workspace")
            os.makedirs(workspace_dir, exist_ok=True)
            uid_dir = os.path.join(tmpdir, "test-123")
            os.makedirs(uid_dir, exist_ok=True)

            mock_logger = MagicMock()
            result = self.utils.collect_outputs_local(
                workspace_dir, tmpdir, "test-123", "prefix", mock_logger
            )

            self.assertIsNone(result)
            mock_logger.error.assert_called_once()


class TestLoadBaseDocker(unittest.TestCase):
    """Test load_base_docker function."""

    @classmethod
    def setUpClass(cls):
        cls.utils = utils_module

    def test_loads_base_dockerfile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docker_dir = Path(tmpdir)
            iid_dir = docker_dir / "base_dockerfile" / "test-123"
            iid_dir.mkdir(parents=True)
            dockerfile_path = iid_dir / "Dockerfile"
            dockerfile_path.write_text("FROM python:3.9\nRUN echo hello")

            result = self.utils.load_base_docker(str(docker_dir), "test-123")

            self.assertIn("FROM python:3.9", result)

    def test_raises_for_missing_dockerfile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                self.utils.load_base_docker(tmpdir, "test-123")


class TestInstanceDocker(unittest.TestCase):
    """Test instance_docker function."""

    @classmethod
    def setUpClass(cls):
        cls.utils = utils_module

    def test_loads_instance_dockerfile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docker_dir = Path(tmpdir)
            iid_dir = docker_dir / "instance_dockerfile" / "test-123"
            iid_dir.mkdir(parents=True)
            dockerfile_path = iid_dir / "Dockerfile"
            dockerfile_path.write_text("FROM base\nRUN test")

            result = self.utils.instance_docker(str(docker_dir), "test-123")

            self.assertIn("FROM base", result)

    def test_raises_for_missing_dockerfile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                self.utils.instance_docker(tmpdir, "test-123")


class TestCreateEntryscript(unittest.TestCase):
    """Test create_entryscript function."""

    @classmethod
    def setUpClass(cls):
        cls.utils = utils_module

    def test_creates_entryscript(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docker_dir = Path(tmpdir)

            base_docker_dir = docker_dir / "base_dockerfile" / "test-123"
            base_docker_dir.mkdir(parents=True)
            (base_docker_dir / "Dockerfile").write_text("ENV BASE_VAR=1\nFROM python:3.9")

            instance_docker_dir = docker_dir / "instance_dockerfile" / "test-123"
            instance_docker_dir.mkdir(parents=True)
            (instance_docker_dir / "Dockerfile").write_text("ENV INSTANCE_VAR=2\nRUN test")

            sample = {
                "instance_id": "test-123",
                "before_repo_set_cmd": "\ncd /app\npip install",
                "selected_test_files_to_run": '["test_file1.py", "test_file2.py"]',
                "base_commit": "abc123",
            }

            result = self.utils.create_entryscript(str(docker_dir), sample)

            self.assertIn("export BASE_VAR=1", result)
            self.assertIn("export INSTANCE_VAR=2", result)
            self.assertIn("git reset --hard abc123", result)
            self.assertIn("git checkout abc123", result)
            self.assertIn("git apply -v /workspace/patch.diff", result)
            self.assertIn("pip install", result)
            self.assertIn("test_file1.py,test_file2.py", result)


class TestAssembleWorkspaceFiles(unittest.TestCase):
    """Test assemble_workspace_files function."""

    @classmethod
    def setUpClass(cls):
        cls.utils = utils_module

    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.create_entryscript')
    def test_assembles_all_files(self, mock_create_entryscript):
        with tempfile.TemporaryDirectory() as tmpdir:
            scripts_dir = Path(tmpdir) / "scripts"
            docker_dir = Path(tmpdir) / "docker"

            uid = "test-123"
            instance_scripts_dir = scripts_dir / uid
            instance_scripts_dir.mkdir(parents=True)
            (instance_scripts_dir / "run_script.sh").write_text("#!/bin/bash\necho run")
            (instance_scripts_dir / "parser.py").write_text("print('parser')")

            (docker_dir / "base_dockerfile" / uid).mkdir(parents=True)
            (docker_dir / "base_dockerfile" / uid / "Dockerfile").write_text("FROM python")
            (docker_dir / "instance_dockerfile" / uid).mkdir(parents=True)
            (docker_dir / "instance_dockerfile" / uid / "Dockerfile").write_text("RUN test")

            mock_create_entryscript.return_value = "entryscript content"

            sample = {"instance_id": uid}
            files, entryscript = self.utils.assemble_workspace_files(
                uid, str(scripts_dir), str(docker_dir), "test patch", sample
            )

            self.assertIn("patch.diff", files)
            self.assertIn("run_script.sh", files)
            self.assertIn("parser.py", files)
            self.assertIn("entryscript.sh", files)
            self.assertEqual(entryscript, "entryscript content")

    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.create_entryscript')
    def test_strips_binary_from_patch(self, mock_create_entryscript):
        with tempfile.TemporaryDirectory() as tmpdir:
            scripts_dir = Path(tmpdir) / "scripts"
            docker_dir = Path(tmpdir) / "docker"

            uid = "test-123"
            instance_scripts_dir = scripts_dir / uid
            instance_scripts_dir.mkdir(parents=True)
            (instance_scripts_dir / "run_script.sh").write_text("#!/bin/bash")
            (instance_scripts_dir / "parser.py").write_text("print('parser')")

            (docker_dir / "base_dockerfile" / uid).mkdir(parents=True)
            (docker_dir / "base_dockerfile" / uid / "Dockerfile").write_text("FROM python")
            (docker_dir / "instance_dockerfile" / uid).mkdir(parents=True)
            (docker_dir / "instance_dockerfile" / uid / "Dockerfile").write_text("RUN test")

            mock_create_entryscript.return_value = "entryscript content"

            patch_with_binary = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1 +1 @@
-old
+new
diff --git a/bin.bin b/bin.bin
Binary files a/bin.bin and b/bin.bin differ
"""

            sample = {"instance_id": uid}
            files, _ = self.utils.assemble_workspace_files(
                uid, str(scripts_dir), str(docker_dir), patch_with_binary, sample
            )

            self.assertNotIn("Binary files", files["patch.diff"])
            self.assertIn("diff --git a/file.py", files["patch.diff"])


class TestEvalWithDockerFull(unittest.TestCase):
    """Test eval_with_docker function with more scenarios."""

    @classmethod
    def setUpClass(cls):
        cls.utils = utils_module

    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.collect_outputs_local')
    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.save_entryscript_copy')
    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.get_dockerhub_image_uri')
    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.assemble_workspace_files')
    def test_runs_container_successfully(self, mock_assemble, mock_get_uri, mock_save, mock_collect):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_assemble.return_value = ({"patch.diff": "patch"}, "entryscript")
            mock_get_uri.return_value = "test-image"
            mock_collect.return_value = {"result": "success"}

            mock_container = MagicMock()
            mock_container.wait.return_value = {"StatusCode": 0}

            mock_client = MagicMock()
            mock_client.containers.run.return_value = mock_container

            sample = {"instance_id": "test-123"}
            result = self.utils.eval_with_docker(
                "test patch", sample, tmpdir, Path(tmpdir), Path(tmpdir),
                MagicMock(), "prefix", mock_client, 3600
            )

            self.assertEqual(result, {"result": "success"})
            mock_client.containers.run.assert_called_once()

    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.get_dockerhub_image_uri')
    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.assemble_workspace_files')
    def test_handles_container_timeout(self, mock_assemble, mock_get_uri):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_assemble.return_value = ({"patch.diff": "patch"}, "entryscript")
            mock_get_uri.return_value = "test-image"

            mock_container = MagicMock()
            timeout_error = Exception("Timeout")
            mock_container.wait.side_effect = timeout_error
            mock_container.stop = MagicMock()
            mock_container.kill = MagicMock()

            mock_client = MagicMock()
            mock_client.containers.run.return_value = mock_container
            mock_client.errors = MagicMock()
            mock_client.errors.Timeout = type('Timeout', (Exception,), {})
            mock_container.wait.side_effect = mock_client.errors.Timeout("Timeout")

            mock_logger = MagicMock()
            sample = {"instance_id": "test-123"}

            result = self.utils.eval_with_docker(
                "test patch", sample, tmpdir, Path(tmpdir), Path(tmpdir),
                mock_logger, "prefix", mock_client, 3600
            )

            self.assertEqual(result, {
                "tests": [],
                "error": "timeout",
                "message": "Evaluation timed out after 3600 seconds",
                "instance_id": "test-123"
            })

    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.collect_outputs_local')
    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.get_dockerhub_image_uri')
    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.assemble_workspace_files')
    def test_handles_container_non_zero_exit(self, mock_assemble, mock_get_uri, mock_collect):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_assemble.return_value = ({"patch.diff": "patch"}, "entryscript")
            mock_get_uri.return_value = "test-image"
            mock_collect.return_value = {"result": "partial"}

            mock_container = MagicMock()
            mock_container.wait.return_value = {"StatusCode": 1}

            mock_client = MagicMock()
            mock_client.containers.run.return_value = mock_container

            mock_logger = MagicMock()
            sample = {"instance_id": "test-123"}

            result = self.utils.eval_with_docker(
                "test patch", sample, tmpdir, Path(tmpdir), Path(tmpdir),
                mock_logger, "prefix", mock_client, 3600
            )

            self.assertEqual(result, {"result": "partial"})
            mock_logger.error.assert_called()


class TestEvalWithDockerMoreScenarios(unittest.TestCase):
    """Test eval_with_docker function with more scenarios."""

    @classmethod
    def setUpClass(cls):
        cls.utils = utils_module

    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.collect_outputs_local')
    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.save_entryscript_copy')
    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.get_dockerhub_image_uri')
    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.assemble_workspace_files')
    def test_raises_when_docker_not_installed(self, mock_assemble, mock_get_uri, mock_save, mock_collect):
        from ais_bench.benchmark.utils.logging.exceptions import AISBenchImportError
        
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_assemble.return_value = ({"patch.diff": "patch"}, "entryscript")
            mock_get_uri.return_value = "test-image"

            sample = {"instance_id": "test-123"}

            with patch.dict('sys.modules', {'docker': None}):
                with self.assertRaises(AISBenchImportError):
                    self.utils.eval_with_docker(
                        "test patch", sample, tmpdir, Path(tmpdir), Path(tmpdir),
                        MagicMock(), "prefix", None, 3600
                    )

    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.collect_outputs_local')
    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.save_entryscript_copy')
    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.get_dockerhub_image_uri')
    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.assemble_workspace_files')
    def test_handles_none_output(self, mock_assemble, mock_get_uri, mock_save, mock_collect):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_assemble.return_value = ({"patch.diff": "patch"}, "entryscript")
            mock_get_uri.return_value = "test-image"
            mock_collect.return_value = None

            mock_container = MagicMock()
            mock_container.wait.return_value = {"StatusCode": 0}

            mock_client = MagicMock()
            mock_client.containers.run.return_value = mock_container
            mock_client.errors = MagicMock()
            mock_client.errors.Timeout = Exception

            sample = {"instance_id": "test-123"}
            result = self.utils.eval_with_docker(
                "test patch", sample, tmpdir, Path(tmpdir), Path(tmpdir),
                MagicMock(), "prefix", mock_client, 3600
            )

            self.assertIsNone(result)
            mock_save.assert_not_called()

    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.collect_outputs_local')
    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.save_entryscript_copy')
    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.get_dockerhub_image_uri')
    @patch('ais_bench.benchmark.tasks.swebench_pro.utils.assemble_workspace_files')
    def test_handles_general_exception(self, mock_assemble, mock_get_uri, mock_save, mock_collect):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_assemble.return_value = ({"patch.diff": "patch"}, "entryscript")
            mock_get_uri.return_value = "test-image"

            mock_client = MagicMock()
            mock_client.containers.run.side_effect = Exception("General error")

            mock_logger = MagicMock()
            sample = {"instance_id": "test-123"}

            result = self.utils.eval_with_docker(
                "test patch", sample, tmpdir, Path(tmpdir), Path(tmpdir),
                mock_logger, "prefix", mock_client, 3600
            )

            self.assertIsNone(result)
            mock_logger.error.assert_called()


if __name__ == '__main__':
    unittest.main()
