import os
import json
import stat
import shutil
import tempfile
import unittest
from unittest.mock import patch
import importlib

# Module under test
from ais_bench.benchmark.utils.file.file import (
    write_status,
    read_and_clear_statuses,
    match_files,
    match_cfg_file,
)
from ais_bench.benchmark.utils.logging.exceptions import FileMatchError


file_module = importlib.import_module("ais_bench.benchmark.utils.file.file")


class TestWriteStatus(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.status_file = os.path.join(self.tmpdir, "status.json")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_write_status_new_and_append(self):
        ok = write_status(self.status_file, {"task": 1})
        self.assertTrue(ok)
        with open(self.status_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["task"], 1)

        ok2 = write_status(self.status_file, {"task": 2})
        self.assertTrue(ok2)
        with open(self.status_file, "r", encoding="utf-8") as f:
            data2 = json.load(f)
        self.assertEqual([s["task"] for s in data2], [1, 2])

    def test_write_status_invalid_json_recovers(self):
        # Pre-create file with invalid JSON
        with open(self.status_file, "w", encoding="utf-8") as f:
            f.write("{invalid json}")

        with patch.object(file_module, "logger") as mock_logger:
            ok = write_status(self.status_file, {"ok": True})
            self.assertTrue(ok)
            mock_logger.warning.assert_called()  # logged recovery

        with open(self.status_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data, [{"ok": True}])

    def test_write_status_ioerror_on_write_returns_false(self):
        # Use a directory path to trigger IOError on write
        bad_path = os.path.join(self.tmpdir, "dir_as_file")
        os.makedirs(bad_path)

        with patch.object(file_module, "logger") as mock_logger:
            ok = write_status(bad_path, {"x": 1})
            self.assertFalse(ok)
            mock_logger.warning.assert_called()


class TestReadAndClearStatuses(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_dir_not_exist_returns_empty(self):
        nonexist_dir = os.path.join(self.tmpdir, "nope")
        out = read_and_clear_statuses(nonexist_dir, ["a.json"])
        self.assertEqual(out, [])

    def test_reads_and_clears_multiple(self):
        files = ["a.json", "b.json"]
        paths = []
        all_vals = []
        for i, name in enumerate(files, start=1):
            p = os.path.join(self.tmpdir, name)
            with open(p, "w", encoding="utf-8") as f:
                json.dump([{"i": i}, {"i": i + 10}], f)
            paths.append(p)
            all_vals.extend([{"i": i}, {"i": i + 10}])

        out = read_and_clear_statuses(self.tmpdir, files)
        self.assertEqual(out, all_vals)

        # files should be cleared to []
        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f), [])

    def test_invalid_json_cleared_and_continues(self):
        good = os.path.join(self.tmpdir, "good.json")
        bad = os.path.join(self.tmpdir, "bad.json")
        with open(good, "w", encoding="utf-8") as f:
            json.dump([{"ok": 1}], f)
        with open(bad, "w", encoding="utf-8") as f:
            f.write("{broken}")

        with patch.object(file_module, "logger") as mock_logger:
            out = read_and_clear_statuses(self.tmpdir, ["good.json", "bad.json"])
            # only the good entries are returned
            self.assertEqual(out, [{"ok": 1}])
            # bad file cleared
            with open(bad, "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f), [])
            mock_logger.warning.assert_called()

    def test_ioerror_on_clear_logs_warning(self):
        name = "ro.json"
        p = os.path.join(self.tmpdir, name)
        with open(p, "w", encoding="utf-8") as f:
            json.dump([{"k": 1}], f)

        # Make file read-only to cause failure when opening with 'w'
        os.chmod(p, stat.S_IREAD)
        try:
            with patch.object(file_module, "logger") as mock_logger:
                out = read_and_clear_statuses(self.tmpdir, [name])
                # It should still return the content read
                self.assertEqual(out, [{"k": 1}])
                mock_logger.warning.assert_called()
        finally:
            # Restore permissions so tearDown can clean up
            os.chmod(p, stat.S_IWUSR | stat.S_IREAD)


class TestMatchFiles(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # create files
        self.files = ["alpha.py", "beta_test.py", "Gamma.PY", "note.txt"]
        for name in self.files:
            path = os.path.join(self.tmpdir, name)
            with open(path, "w", encoding="utf-8") as f:
                f.write("x")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_exact_pattern_and_sorting(self):
        res = match_files(self.tmpdir, "alpha.py")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][0], "alpha")
        self.assertTrue(res[0][1].endswith("alpha.py"))

        res_many = match_files(self.tmpdir, ["alpha.py", "beta_test.py"])
        self.assertEqual([r[0] for r in res_many], ["alpha", "beta_test"])  # sorted by name

    def test_case_insensitive_and_fuzzy(self):
        # case-insensitive .PY
        res = match_files(self.tmpdir, "gamma.py")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][0].lower(), "gamma")

        res_fuzzy = match_files(self.tmpdir, "beta", fuzzy=True)
        self.assertEqual(len(res_fuzzy), 1)
        self.assertEqual(res_fuzzy[0][0], "beta_test")


class TestMatchCfgFile(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.wd1 = os.path.join(self.tmpdir, "wd1")
        self.wd2 = os.path.join(self.tmpdir, "wd2")
        os.makedirs(self.wd1)
        os.makedirs(self.wd2)

        # unique
        with open(os.path.join(self.wd1, "cfg1.py"), "w", encoding="utf-8") as f:
            f.write("x")
        # ambiguous across workdirs
        for wd in (self.wd1, self.wd2):
            with open(os.path.join(wd, "same.py"), "w", encoding="utf-8") as f:
                f.write("x")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_match_exact(self):
        res = match_cfg_file([self.wd1, self.wd2], "cfg1")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][0], "cfg1")
        self.assertTrue(res[0][1].endswith("cfg1.py"))

    def test_not_found_raises(self):
        with self.assertRaises(FileMatchError):
            match_cfg_file([self.wd1, self.wd2], "does_not_exist")

    def test_ambiguous_returns_first_and_warns(self):
        with patch.object(file_module, "logger") as mock_logger:
            res = match_cfg_file([self.wd1, self.wd2], "same")
            mock_logger.warning.assert_called()
        # first match should come from wd1 due to order
        self.assertEqual(len(res), 1)
        self.assertTrue(res[0][1].startswith(self.wd1))


if __name__ == "__main__":
    unittest.main()
