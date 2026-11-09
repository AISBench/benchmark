from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import disk_guard
import finalize
import stage_runtime
from plan_run import normalize, validate
from test_serial_task import fixture

class StorageAndReleaseTests(unittest.TestCase):
    def test_budget_admission_and_runtime(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(disk_guard.shutil,'disk_usage',return_value=SimpleNamespace(free=12*disk_guard.GIB)):
                disk_guard.check(d,2)
                with self.assertRaises(disk_guard.DiskLimitError): disk_guard.check(d,2,admission=True)
            with patch.object(disk_guard,'used_bytes',return_value=5*disk_guard.GIB):
                with self.assertRaises(disk_guard.DiskLimitError): disk_guard.check(d)

    def test_full_disk_does_not_prevent_container_stop(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'logs').mkdir()
            stage_runtime.update(root,service='READY',container_id='cid',image_id='image')
            calls=[]
            def command(argv,**kw):
                calls.append(argv)
                return 'PID START COMMAND\n42 date model' if argv[:2]==['docker','top'] else ''
            running={'Id':'cid','Image':'image','State':{'Running':True,'Pid':42}}
            stopped={'Id':'cid','Image':'image','State':{'Running':False,'Pid':0}}
            with patch.object(finalize,'read_plan',return_value={}), patch.object(finalize,'inspect_owned',side_effect=[running,stopped]), patch.object(finalize,'command',side_effect=command), patch.object(finalize,'ticks',return_value='123'), patch.object(Path,'write_text',side_effect=OSError(28,'full')):
                result=finalize.release(root,'stop_container','job')
            self.assertIn(['docker','stop','-t','30','cid'],calls)
            self.assertTrue(result['recording_errors'])

    def test_residual_pid_reuse_is_not_killed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'logs').mkdir()
            stage_runtime.update(root,service='READY',container_id='cid',image_id='image')
            def command(argv,**kw):
                if argv[:2]==['docker','top']: return '42 date model'
                if argv[:2]==['npu-smi','info']: return '| 0 | 42 | model |'
                return ''
            running={'Id':'cid','Image':'image','State':{'Running':True,'Pid':42}}
            stopped={'Id':'cid','Image':'image','State':{'Running':False,'Pid':0}}
            with patch.object(finalize,'read_plan',return_value={}), patch.object(finalize,'inspect_owned',side_effect=[running,stopped]), patch.object(finalize,'command',side_effect=command), patch.object(finalize,'ticks',side_effect=['old','new']), patch.object(finalize.os,'kill') as kill:
                with self.assertRaisesRegex(RuntimeError,'identity'): finalize.release(root,'stop_container','job')
                kill.assert_not_called()

    def test_verbose_logging_rejected(self):
        task=normalize(fixture())
        task['deployments'][0]['instances'][0]['launch'].setdefault('env',{})['ASCEND_SLOG_PRINT_TO_STDOUT']='1'
        self.assertTrue(any('operator logging' in e for e in validate(task)))
