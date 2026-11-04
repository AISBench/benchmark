import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("log_format", Path(__file__).resolve().parents[2]/"assets/aisbench-prefix/log_format.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class LogFormatTests(unittest.TestCase):
    def test_preserves_table_numbers_and_separates_next_message(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"aisbench.log"
            path.write_text("│ TTFT │ 123.4 │\n╘══════╛[" + chr(27) + "[32mINFO" + chr(27) + "[0m] done", encoding="utf-8")
            module.normalize_log(path)
            self.assertEqual(path.read_text(encoding="utf-8"), "│ TTFT │ 123.4 │\n╘══════╛\n[INFO] done\n")
            first = path.read_bytes()
            module.normalize_log(path)
            self.assertEqual(path.read_bytes(), first)
