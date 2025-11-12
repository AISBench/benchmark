import unittest
from unittest import mock
import tempfile
import sqlite3
import os
import queue
import numpy as np

from ais_bench.benchmark.openicl.icl_inferencer.output_handler.base_handler import BaseInferencerOutputHandler
from ais_bench.benchmark.models.output import Output
from ais_bench.benchmark.utils.logging.exceptions import AISBenchImplementationError, ParameterValueError, FileOperationError


class ConcreteOutputHandler(BaseInferencerOutputHandler):
    """Concrete implementation for testing"""
    def get_result(self, conn, input, output, gold=None):
        return {"success": True, "result": "test"}


class TestBaseInferencerOutputHandler(unittest.TestCase):
    def test_init(self):
        """测试BaseInferencerOutputHandler的初始化"""
        handler = ConcreteOutputHandler(save_every=10)
        self.assertEqual(handler.save_every, 10)
        self.assertTrue(handler.all_success)

    def test_get_result_abstract(self):
        """测试抽象方法get_result未实现时抛出异常"""
        handler = BaseInferencerOutputHandler()
        with self.assertRaises(AISBenchImplementationError):
            handler.get_result(None, "input", "output")

    def test_write_to_json_invalid_save_dir(self):
        """测试write_to_json在无效保存目录时抛出异常"""
        handler = ConcreteOutputHandler()
        with self.assertRaises(ParameterValueError):
            handler.write_to_json("", False)
        with self.assertRaises(ParameterValueError):
            handler.write_to_json(None, False)

    def test_write_to_json_success(self):
        """测试write_to_json成功写入JSON文件"""
        handler = ConcreteOutputHandler()
        handler.results_dict["test"] = {"uid1": {"id": 0, "result": "test1"}}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            handler.write_to_json(tmpdir, False)
            file_path = os.path.join(tmpdir, "test.jsonl")
            self.assertTrue(os.path.exists(file_path))

    def test_write_to_json_empty_results_dict(self):
        """测试write_to_json在results_dict为空时不创建文件"""
        handler = ConcreteOutputHandler()
        handler.results_dict["test"] = {}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            handler.write_to_json(tmpdir, False)
            file_path = os.path.join(tmpdir, "test.jsonl")
            self.assertFalse(os.path.exists(file_path))

    def test_write_to_json_exception_handling(self):
        """测试write_to_json在发生异常时的错误处理"""
        handler = ConcreteOutputHandler()
        handler.results_dict["test"] = {"uid1": {"id": 0, "result": "test1"}}
        
        with mock.patch('pathlib.Path.mkdir', side_effect=OSError("Permission denied")):
            with tempfile.TemporaryDirectory() as tmpdir:
                with self.assertRaises(FileOperationError):
                    handler.write_to_json(tmpdir, False)

    def test_write_to_json_perf_mode(self):
        """测试write_to_json在perf_mode=True时写入详细信息文件"""
        handler = ConcreteOutputHandler()
        handler.results_dict["test"] = {"uid1": {"id": 0, "result": "test1"}}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            handler.write_to_json(tmpdir, True)
            file_path = os.path.join(tmpdir, "test_details.jsonl")
            self.assertTrue(os.path.exists(file_path))

    def test_report_cache_info_async(self):
        """测试report_cache_info异步方法成功报告缓存信息"""
        handler = ConcreteOutputHandler()
        
        async def run_test():
            result = await handler.report_cache_info(0, "input", "output", "test", "gold")
            self.assertTrue(result)
            
        import asyncio
        asyncio.run(run_test())

    def test_report_cache_info_async_failure(self):
        """测试report_cache_info异步方法在队列满时返回False"""
        handler = ConcreteOutputHandler()
        handler.cache_queue.async_q.put_nowait = mock.Mock(side_effect=Exception("Queue full"))
        
        async def run_test():
            result = await handler.report_cache_info(0, "input", "output", "test", "gold")
            self.assertFalse(result)
            
        import asyncio
        asyncio.run(run_test())

    def test_report_cache_info_sync(self):
        """测试report_cache_info_sync同步方法成功报告缓存信息"""
        handler = ConcreteOutputHandler()
        result = handler.report_cache_info_sync(0, "input", "output", "test", "gold")
        self.assertTrue(result)

    def test_report_cache_info_sync_failure(self):
        """测试report_cache_info_sync同步方法在队列满时返回False"""
        handler = ConcreteOutputHandler()
        handler.cache_queue.sync_q.put_nowait = mock.Mock(side_effect=Exception("Queue full"))
        result = handler.report_cache_info_sync(0, "input", "output", "test", "gold")
        self.assertFalse(result)

    def test_extract_and_write_arrays_atomic_types(self):
        """测试_extract_and_write_arrays处理基本数据类型"""
        handler = ConcreteOutputHandler()
        conn = sqlite3.connect(":memory:")
        
        self.assertEqual(handler._extract_and_write_arrays(None, conn), None)
        self.assertEqual(handler._extract_and_write_arrays(True, conn), True)
        self.assertEqual(handler._extract_and_write_arrays(42, conn), 42)
        self.assertEqual(handler._extract_and_write_arrays(3.14, conn), 3.14)
        self.assertEqual(handler._extract_and_write_arrays("test", conn), "test")
        self.assertEqual(handler._extract_and_write_arrays([1, 2], conn), [1, 2])
        self.assertEqual(handler._extract_and_write_arrays((1, 2), conn), (1, 2))
        
        conn.close()

    def test_extract_and_write_arrays_numpy(self):
        """测试_extract_and_write_arrays处理numpy数组并保存到数据库"""
        from ais_bench.benchmark.openicl.icl_inferencer.output_handler.db_utils import init_db
        
        handler = ConcreteOutputHandler()
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            conn = init_db(db_path)
            arr = np.array([1, 2, 3])
            result = handler._extract_and_write_arrays(arr, conn)
            self.assertIn("__db_ref__", result)
            conn.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_extract_and_write_arrays_numpy_error(self):
        """测试_extract_and_write_arrays在numpy数组保存失败时返回None"""
        handler = ConcreteOutputHandler()
        conn = sqlite3.connect(":memory:")
        
        with mock.patch('ais_bench.benchmark.openicl.icl_inferencer.output_handler.base_handler.save_numpy_to_db', 
                       side_effect=Exception("DB error")):
            arr = np.array([1, 2, 3])
            result = handler._extract_and_write_arrays(arr, conn)
            self.assertIsNone(result)
        
        conn.close()

    def test_extract_and_write_arrays_dict(self):
        """测试_extract_and_write_arrays处理字典类型数据"""
        handler = ConcreteOutputHandler()
        conn = sqlite3.connect(":memory:")
        
        data = {"key1": "value1", "key2": 42}
        result = handler._extract_and_write_arrays(data, conn)
        self.assertEqual(result["key1"], "value1")
        self.assertEqual(result["key2"], 42)
        
        conn.close()

    def test_extract_and_write_arrays_json_serializable(self):
        """测试_extract_and_write_arrays处理JSON可序列化对象"""
        handler = ConcreteOutputHandler()
        conn = sqlite3.connect(":memory:")
        
        obj = {"nested": {"key": "value"}}
        result = handler._extract_and_write_arrays(obj, conn)
        self.assertEqual(result, obj)
        
        conn.close()

    def test_extract_and_write_arrays_fallback_to_string(self):
        """测试_extract_and_write_arrays在对象无法JSON序列化时转换为字符串"""
        handler = ConcreteOutputHandler()
        conn = sqlite3.connect(":memory:")
        
        class CustomObj:
            def __str__(self):
                return "custom_object"
        
        obj = CustomObj()
        result = handler._extract_and_write_arrays(obj, conn)
        self.assertEqual(result, "custom_object")
        
        conn.close()

    def test_extract_and_write_arrays_fallback_to_none(self):
        """测试_extract_and_write_arrays在对象无法转换为字符串时返回None"""
        handler = ConcreteOutputHandler()
        conn = sqlite3.connect(":memory:")
        
        class BadObj:
            def __str__(self):
                raise Exception("Cannot convert to string")
        
        obj = BadObj()
        result = handler._extract_and_write_arrays(obj, conn)
        self.assertIsNone(result)
        
        conn.close()

    def test_run_cache_consumer_basic(self):
        """测试run_cache_consumer基本功能，从队列读取并处理缓存信息"""
        handler = ConcreteOutputHandler(save_every=2)
        handler.all_success = False
        
        with tempfile.TemporaryDirectory() as tmpdir:
            handler.cache_queue.sync_q.put((0, "test", "input1", "output1", "gold1"))
            handler.cache_queue.sync_q.put((1, "test", "input2", "output2", "gold2"))
            handler.cache_queue.sync_q.put(None)
            
            handler.get_result = mock.Mock(return_value={"success": True, "result": "test"})
            
            file_name = "tmp_test.jsonl"
            handler.run_cache_consumer(tmpdir, file_name, False, save_every=2)
            
            file_path = os.path.join(tmpdir, file_name)
            self.assertTrue(os.path.exists(file_path))

    def test_run_cache_consumer_perf_mode(self):
        """测试run_cache_consumer在perf_mode=True时将数据库文件移动到db_data目录"""
        handler = ConcreteOutputHandler(save_every=1)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            handler.cache_queue.sync_q.put((0, "test", "input1", "output1", "gold1"))
            handler.cache_queue.sync_q.put(None)
            
            handler.get_result = mock.Mock(return_value={"success": True, "result": "test"})
            
            file_name = "tmp_test.jsonl"
            handler.run_cache_consumer(tmpdir, file_name, True, save_every=1)
            
            db_data_dir = os.path.join(os.path.dirname(tmpdir), "db_data")
            if os.path.exists(db_data_dir):
                db_files = [f for f in os.listdir(db_data_dir) if f.endswith('.db')]
                self.assertGreaterEqual(len(db_files), 0)

    def test_run_cache_consumer_all_success_cleanup(self):
        """测试run_cache_consumer在all_success=True时删除输出文件"""
        handler = ConcreteOutputHandler(save_every=1)
        handler.all_success = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            handler.cache_queue.sync_q.put((0, "test", "input1", "output1", "gold1"))
            handler.cache_queue.sync_q.put(None)
            
            handler.get_result = mock.Mock(return_value={"success": True, "result": "test"})
            
            file_name = "tmp_test.jsonl"
            handler.run_cache_consumer(tmpdir, file_name, False, save_every=1)
            
            file_path = os.path.join(tmpdir, file_name)
            self.assertFalse(os.path.exists(file_path))

    def test_run_cache_consumer_batch_write(self):
        """测试run_cache_consumer批量写入功能"""
        handler = ConcreteOutputHandler(save_every=2)
        handler.all_success = False
        
        with tempfile.TemporaryDirectory() as tmpdir:
            handler.cache_queue.sync_q.put((0, "test", "input1", "output1", "gold1"))
            handler.cache_queue.sync_q.put((1, "test", "input2", "output2", "gold2"))
            handler.cache_queue.sync_q.put((2, "test", "input3", "output3", "gold3"))
            handler.cache_queue.sync_q.put(None)
            
            handler.get_result = mock.Mock(return_value={"success": True, "result": "test"})
            
            file_name = "tmp_test.jsonl"
            handler.run_cache_consumer(tmpdir, file_name, False, save_every=2)
            
            file_path = os.path.join(tmpdir, file_name)
            self.assertTrue(os.path.exists(file_path))
            with open(file_path, "r") as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 3)

    def test_run_cache_consumer_with_exception(self):
        """测试run_cache_consumer在发生异常时的异常处理"""
        handler = ConcreteOutputHandler(save_every=1)
        handler.all_success = False
        
        with tempfile.TemporaryDirectory() as tmpdir:
            handler.cache_queue.sync_q.put((0, "test", "input1", "output1", "gold1"))
            handler.cache_queue.sync_q.put(None)
            
            handler.get_result = mock.Mock(side_effect=Exception("Processing error"))
            
            file_name = "tmp_test.jsonl"
            handler.run_cache_consumer(tmpdir, file_name, False, save_every=1)
            
            file_path = os.path.join(tmpdir, file_name)

    def test_run_cache_consumer_perf_mode_db_handling(self):
        """测试run_cache_consumer在perf_mode下将数据库文件移动到db_data目录"""
        handler = ConcreteOutputHandler(save_every=1)
        handler.all_success = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            handler.cache_queue.sync_q.put((0, "test", "input1", "output1", "gold1"))
            handler.cache_queue.sync_q.put(None)
            
            handler.get_result = mock.Mock(return_value={"success": True, "result": "test"})
            
            file_name = "tmp_test.jsonl"
            handler.run_cache_consumer(tmpdir, file_name, True, save_every=1)
            
            db_path = os.path.join(tmpdir, "test.db")
            db_data_dir = os.path.join(os.path.dirname(tmpdir), "db_data")
            if os.path.exists(db_data_dir):
                self.assertFalse(os.path.exists(db_path))

    def test_run_cache_consumer_accuracy_mode_db_cleanup(self):
        """测试run_cache_consumer在accuracy模式下删除数据库文件"""
        handler = ConcreteOutputHandler(save_every=1)
        handler.all_success = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            handler.cache_queue.sync_q.put((0, "test", "input1", "output1", "gold1"))
            handler.cache_queue.sync_q.put(None)
            
            handler.get_result = mock.Mock(return_value={"success": True, "result": "test"})
            
            file_name = "tmp_test.jsonl"
            handler.run_cache_consumer(tmpdir, file_name, False, save_every=1)
            
            db_path = os.path.join(tmpdir, "test.db")
            self.assertFalse(os.path.exists(db_path))

    def test_stop_cache_consumer(self):
        """测试stop_cache_consumer方法向队列发送停止信号"""
        handler = ConcreteOutputHandler()
        handler.stop_cache_consumer()
        item = handler.cache_queue.sync_q.get(timeout=0.1)
        self.assertIsNone(item)

    def test_stop_cache_consumer_failure(self):
        """测试stop_cache_consumer在发生异常时抛出AISBenchRuntimeError"""
        handler = ConcreteOutputHandler()
        handler.cache_queue.sync_q.put = mock.Mock(side_effect=Exception("Queue error"))
        from ais_bench.benchmark.utils.logging.exceptions import AISBenchRuntimeError
        with self.assertRaises(AISBenchRuntimeError):
            handler.stop_cache_consumer()


if __name__ == '__main__':
    unittest.main()

