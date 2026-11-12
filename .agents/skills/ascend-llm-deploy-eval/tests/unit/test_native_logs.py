import io
from contextlib import redirect_stdout
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import watch_logs
import discover_tool

class LogTests(unittest.TestCase):
    def test_new_attempt_logs_and_incremental_reads(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cursors={}
            def read():
                stream=io.StringIO()
                with redirect_stdout(stream):watch_logs.poll(root,'test',cursors)
                return stream.getvalue()
            self.assertEqual(read(),'')
            log=root/'nodes/node-a/sub_tasks/c1/attempt-01/outputs/ts/logs/infer/model/dataset.out'
            log.parent.mkdir(parents=True);log.write_text('first line\n')
            self.assertIn('first line',read());self.assertEqual(read(),'')
            with log.open('a') as f:f.write('second line\n')
            text=read();self.assertIn('second line',text);self.assertNotIn('first line',text)
            other=root/'nodes/node-a/sub_tasks/c1/attempt-02/benchmark.log'
            other.parent.mkdir(parents=True);other.write_text('second attempt')
            self.assertIn('second attempt',read())
            log.write_text('reset\n');self.assertIn('reset',read())

    def test_discovery_needs_only_aisbench(self):
        with patch.object(discover_tool.importlib.util,'find_spec',return_value=SimpleNamespace(submodule_search_locations=['/installed/ais_bench'])) as find, patch.object(discover_tool.shutil,'which',return_value='/bin/ais_bench') as which, patch.object(discover_tool.importlib.metadata,'version',return_value='1.0'):
            result=discover_tool.discover()
            self.assertEqual(result['executable'],'/bin/ais_bench')
            find.assert_called_once_with('ais_bench');which.assert_called_once_with('ais_bench')
