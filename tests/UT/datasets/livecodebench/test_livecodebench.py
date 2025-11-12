import unittest
import json
import sys
import os
from unittest.mock import MagicMock, patch, mock_open

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

try:
    from ais_bench.benchmark.datasets.livecodebench.livecodebench import (
        LCBCodeGenerationDataset,
        LCBCodeExecutionDataset,
        LCBTestOutputPredictionDataset,
        LCBSelfRepairDataset,
        CompassBenchCodeExecutionDataset,
        Platform,
        Difficulty,
        TestType,
        Test
    )
    LCB_AVAILABLE = True
except ImportError:
    LCB_AVAILABLE = False


class LiveCodeBenchTestBase(unittest.TestCase):
    """LiveCodeBench测试的基础类，提供共享功能"""
    @classmethod
    def setUpClass(cls):
        """如果LiveCodeBench模块不可用，跳过所有测试"""
        if not LCB_AVAILABLE:
            cls.skipTest(cls, "LiveCodeBench modules not available")


class TestEnumClasses(LiveCodeBenchTestBase):
    """测试枚举类"""
    
    def test_platform_enum(self):
        """测试Platform枚举"""
        self.assertEqual(Platform.LEETCODE.value, 'leetcode')
        self.assertEqual(Platform.CODEFORCES.value, 'codeforces')
        self.assertEqual(Platform.ATCODER.value, 'atcoder')
    
    def test_difficulty_enum(self):
        """测试Difficulty枚举"""
        self.assertEqual(Difficulty.EASY.value, 'easy')
        self.assertEqual(Difficulty.MEDIUM.value, 'medium')
        self.assertEqual(Difficulty.HARD.value, 'hard')
    
    def test_test_type_enum(self):
        """测试TestType枚举"""
        self.assertEqual(TestType.STDIN.value, 'stdin')
        self.assertEqual(TestType.FUNCTIONAL.value, 'functional')


class TestTestDataclass(LiveCodeBenchTestBase):
    """测试Test数据类"""
    
    def test_test_dataclass(self):
        """测试Test数据类初始化"""
        test = Test(
            input='test_input',
            output='test_output',
            testtype=TestType.STDIN
        )
        self.assertEqual(test.input, 'test_input')
        self.assertEqual(test.output, 'test_output')
        self.assertEqual(test.testtype, TestType.STDIN)
        
        # 测试字符串转枚举
        test2 = Test(
            input='test_input2',
            output='test_output2',
            testtype='functional'
        )
        self.assertEqual(test2.testtype, TestType.FUNCTIONAL)


class TestLCBCodeGenerationDataset(LiveCodeBenchTestBase):
    """测试LCBCodeGenerationDataset类"""
    
    def test_load_with_starter_code(self):
        """测试加载包含starter_code的数据集"""
        from unittest.mock import patch
        from ais_bench.benchmark.datasets.livecodebench import livecodebench
        
        mock_get_path = MagicMock(return_value='/fake/path')
        
        # 模拟数据集
        mock_item = {
            'starter_code': 'def test(): pass',
            'public_test_cases': json.dumps([{'input': 'in1', 'output': 'out1'}]),
            'private_test_cases': json.dumps([{'input': 'in2', 'output': 'out2'}]),
            'metadata': json.dumps({'func_name': 'test_func'})
        }
        
        def transform_func(item):
            item['format_prompt'] = 'test'
            item['evaluation_sample'] = 'test'
            return item
        
        mock_dataset = MagicMock()
        mock_dataset.map = MagicMock(side_effect=lambda func: mock_dataset)
        mock_load_dataset = MagicMock(return_value=mock_dataset)
        
        with patch.object(livecodebench, 'get_data_path', mock_get_path), \
             patch.object(livecodebench, 'load_dataset', mock_load_dataset):
            result = LCBCodeGenerationDataset.load(
                path='/test/path',
                local_mode=True,
                release_version='release_v1'
            )
            
            self.assertIn('test', result)
            self.assertIn('train', result)
            mock_get_path.assert_called_once_with('/test/path', local_mode=True)
    
    def test_load_with_compressed_private_test_cases(self):
        """测试加载压缩的private_test_cases"""
        from unittest.mock import patch
        from ais_bench.benchmark.datasets.livecodebench import livecodebench
        import base64
        import zlib
        import pickle
        
        mock_get_path = MagicMock(return_value='/fake/path')
        
        # 创建压缩的测试用例数据
        test_data = json.dumps([{'input': 'in2', 'output': 'out2'}])
        compressed = base64.b64encode(zlib.compress(pickle.dumps(test_data))).decode('utf-8')
        
        # 模拟一个会调用transform的数据集项
        def transform_func(item):
            item['format_prompt'] = 'test'
            # 测试压缩的private_test_cases处理逻辑
            if item.get('private_test_cases') == compressed:
                try:
                    json.loads(item['private_test_cases'])
                except:
                    # 应该进入解压缩逻辑
                    pass
            item['evaluation_sample'] = 'test'
            return item
        
        mock_dataset = MagicMock()
        mock_dataset.map = MagicMock(side_effect=lambda func: mock_dataset)
        mock_load_dataset = MagicMock(return_value=mock_dataset)
        
        with patch.object(livecodebench, 'get_data_path', mock_get_path), \
             patch.object(livecodebench, 'load_dataset', mock_load_dataset):
            result = LCBCodeGenerationDataset.load(
                path='/test/path',
                local_mode=True,
                release_version='release_v1'
            )
            
            self.assertIn('test', result)
            self.assertIn('train', result)
    
    def test_load_transform_with_starter_code(self):
        """测试transform函数处理有starter_code的情况"""
        from unittest.mock import patch
        from ais_bench.benchmark.datasets.livecodebench import livecodebench
        
        mock_get_path = MagicMock(return_value='/fake/path')
        mock_dataset = MagicMock()
        
        # 模拟transform函数会被调用，并处理有starter_code的情况
        def map_func(transform_func):
            # 模拟transform被调用
            test_item = {
                'starter_code': 'def test(): pass',
                'public_test_cases': json.dumps([{'input': 'in1', 'output': 'out1'}]),
                'private_test_cases': json.dumps([{'input': 'in2', 'output': 'out2'}]),
                'metadata': json.dumps({'func_name': 'test_func'})
            }
            transformed = transform_func(test_item)
            return mock_dataset
        
        mock_dataset.map = map_func
        mock_load_dataset = MagicMock(return_value=mock_dataset)
        
        with patch.object(livecodebench, 'get_data_path', mock_get_path), \
             patch.object(livecodebench, 'load_dataset', mock_load_dataset):
            result = LCBCodeGenerationDataset.load(path='/test/path')
            self.assertIn('test', result)
    
    def test_load_transform_with_compressed_private_cases(self):
        """测试transform函数处理压缩的private_test_cases"""
        from unittest.mock import patch
        from ais_bench.benchmark.datasets.livecodebench import livecodebench
        import base64
        import zlib
        import pickle
        
        mock_get_path = MagicMock(return_value='/fake/path')
        
        # 创建压缩的测试用例数据
        test_data = json.dumps([{'input': 'in2', 'output': 'out2'}])
        compressed = base64.b64encode(zlib.compress(pickle.dumps(test_data))).decode('utf-8')
        
        mock_dataset = MagicMock()
        
        def map_func(transform_func):
            # 模拟transform被调用，private_test_cases是压缩的
            test_item = {
                'starter_code': '',
                'public_test_cases': json.dumps([{'input': 'in1', 'output': 'out1'}]),
                'private_test_cases': compressed,  # 压缩的数据
                'metadata': json.dumps({'func_name': 'test_func'})
            }
            transformed = transform_func(test_item)
            return mock_dataset
        
        mock_dataset.map = map_func
        mock_load_dataset = MagicMock(return_value=mock_dataset)
        
        with patch.object(livecodebench, 'get_data_path', mock_get_path), \
             patch.object(livecodebench, 'load_dataset', mock_load_dataset):
            result = LCBCodeGenerationDataset.load(path='/test/path')
            self.assertIn('test', result)
    
    def test_load_without_starter_code(self):
        """测试加载不包含starter_code的数据集"""
        from unittest.mock import patch
        from ais_bench.benchmark.datasets.livecodebench import livecodebench
        
        mock_get_path = MagicMock(return_value='/fake/path')
        mock_dataset = MagicMock()
        mock_dataset.map = MagicMock(return_value=mock_dataset)
        mock_load_dataset = MagicMock(return_value=mock_dataset)
        
        with patch.object(livecodebench, 'get_data_path', mock_get_path), \
             patch.object(livecodebench, 'load_dataset', mock_load_dataset):
            result = LCBCodeGenerationDataset.load(
                path='/test/path',
                local_mode=False
            )
            
            self.assertIn('test', result)
            self.assertIn('train', result)


class TestLCBCodeExecutionDataset(LiveCodeBenchTestBase):
    """测试LCBCodeExecutionDataset类"""
    
    def test_load(self):
        """测试加载代码执行数据集"""
        from unittest.mock import patch
        from ais_bench.benchmark.datasets.livecodebench import livecodebench
        
        mock_get_path = MagicMock(return_value='/fake/path')
        
        # 模拟transform函数被调用
        def map_func(transform_func):
            test_item = {
                'code': 'def test(): return 1',
                'input': 'test()',
                'output': '1'
            }
            transformed = transform_func(test_item)
            return mock_dataset
        
        mock_dataset = MagicMock()
        mock_dataset.map = map_func
        mock_load_dataset = MagicMock(return_value=mock_dataset)
        
        with patch.object(livecodebench, 'get_data_path', mock_get_path), \
             patch.object(livecodebench, 'load_dataset', mock_load_dataset):
            result = LCBCodeExecutionDataset.load(
                path='/test/path',
                local_mode=True,
                cot=False
            )
            
            self.assertIn('test', result)
            self.assertIn('train', result)
            mock_get_path.assert_called_once_with('/test/path', local_mode=True)
    
    def test_load_with_cot(self):
        """测试加载带COT的代码执行数据集"""
        from unittest.mock import patch
        from ais_bench.benchmark.datasets.livecodebench import livecodebench
        
        mock_get_path = MagicMock(return_value='/fake/path')
        mock_dataset = MagicMock()
        mock_dataset.map = MagicMock(return_value=mock_dataset)
        mock_load_dataset = MagicMock(return_value=mock_dataset)
        
        with patch.object(livecodebench, 'get_data_path', mock_get_path), \
             patch.object(livecodebench, 'load_dataset', mock_load_dataset):
            result = LCBCodeExecutionDataset.load(
                path='/test/path',
                local_mode=False,
                cot=True
            )
            
            self.assertIn('test', result)
            self.assertIn('train', result)


class TestLCBTestOutputPredictionDataset(LiveCodeBenchTestBase):
    """测试LCBTestOutputPredictionDataset类"""
    
    def test_load(self):
        """测试加载测试输出预测数据集"""
        from unittest.mock import patch
        from ais_bench.benchmark.datasets.livecodebench import livecodebench
        
        mock_get_path = MagicMock(return_value='/fake/path')
        
        # 模拟transform函数被调用
        def map_func(transform_func):
            test_item = {
                'question_content': 'Test question',
                'starter_code': 'def test(): pass',
                'test': json.dumps([{'input': 'in1', 'output': 'out1'}])
            }
            transformed = transform_func(test_item)
            return mock_dataset
        
        mock_dataset = MagicMock()
        mock_dataset.map = map_func
        mock_load_dataset = MagicMock(return_value=mock_dataset)
        
        with patch.object(livecodebench, 'get_data_path', mock_get_path), \
             patch.object(livecodebench, 'load_dataset', mock_load_dataset):
            result = LCBTestOutputPredictionDataset.load(
                path='/test/path',
                local_mode=True
            )
            
            self.assertIn('test', result)
            self.assertIn('train', result)


class TestLCBSelfRepairDataset(LiveCodeBenchTestBase):
    """测试LCBSelfRepairDataset类"""
    
    def test_load(self):
        """测试加载自我修复数据集"""
        from unittest.mock import patch
        from ais_bench.benchmark.datasets.livecodebench import livecodebench
        
        mock_dataset = MagicMock()
        
        # 模拟transform函数被调用
        def map_func(transform_func):
            test_item = {
                'question_content': 'Test question',
                'code_list': ['def test(): return 1'],
                'metadata': json.dumps({'error_code': -1, 'error': 'test error'})
            }
            transformed = transform_func(test_item)
            return mock_dataset
        
        mock_dataset.map = map_func
        mock_load_dataset = MagicMock(return_value=mock_dataset)
        
        with patch.object(livecodebench, 'load_dataset', mock_load_dataset):
            result = LCBSelfRepairDataset.load(
                path='livecodebench/code_generation_lite',
                local_mode=False,
                release_version='release_v1'
            )
            
            self.assertIn('test', result)
            self.assertIn('train', result)


class TestCompassBenchCodeExecutionDataset(LiveCodeBenchTestBase):
    """测试CompassBenchCodeExecutionDataset类"""
    
    def test_load(self):
        """测试加载CompassBench代码执行数据集"""
        from unittest.mock import patch
        from ais_bench.benchmark.datasets.livecodebench import livecodebench
        
        mock_get_path = MagicMock(return_value='/fake/path')
        
        # 模拟transform函数被调用
        def map_func(transform_func):
            test_item = {
                'code': 'def test(): return 1',
                'input': 'test()',
                'output': '1'
            }
            transformed = transform_func(test_item)
            return mock_dataset
        
        mock_dataset = MagicMock()
        mock_dataset.map = map_func
        mock_load_from_disk = MagicMock(return_value={'test': mock_dataset})
        
        with patch.object(livecodebench, 'get_data_path', mock_get_path), \
             patch.object(livecodebench, 'load_from_disk', mock_load_from_disk):
            result = CompassBenchCodeExecutionDataset.load(
                path='/test/path',
                local_mode=True,
                cot=False
            )
            
            self.assertIn('test', result)
            self.assertIn('train', result)
            mock_get_path.assert_called_once_with('/test/path', local_mode=True)
    
    def test_load_with_cot(self):
        """测试加载带COT的CompassBench代码执行数据集"""
        from unittest.mock import patch
        from ais_bench.benchmark.datasets.livecodebench import livecodebench
        
        mock_get_path = MagicMock(return_value='/fake/path')
        mock_dataset = MagicMock()
        mock_dataset.map = MagicMock(return_value=mock_dataset)
        mock_load_from_disk = MagicMock(return_value={'test': mock_dataset})
        
        with patch.object(livecodebench, 'get_data_path', mock_get_path), \
             patch.object(livecodebench, 'load_from_disk', mock_load_from_disk):
            result = CompassBenchCodeExecutionDataset.load(
                path='/test/path',
                local_mode=False,
                cot=True
            )
            
            self.assertIn('test', result)
            self.assertIn('train', result)


if __name__ == '__main__':
    unittest.main()

