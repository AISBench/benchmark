import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from ais_bench.benchmark.calculators.default_perf_metric_calculator import DefaultPerfMetricCalculator
from ais_bench.benchmark.utils.logging.exceptions import AISBenchDataContentError


class TestDefaultPerfMetricCalculator(unittest.TestCase):
    def setUp(self):
        # 设置测试数据
        self.perf_details = {
            'id': ['req1', 'req2', 'req3'],
            'start_time': [1000, 1001, 1002],
            'end_time': [1100, 1105, 1110],
            'success': [True, True, False],
            'itl': [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]],
            'latency': [0.1, 0.2, 0.3],
            'ttft': [0.05, 0.06, 0.07],
            'tpot': [0.02, 0.03, 0.04],
            'input_tokens': [10, 20, 30],
            'output_tokens': [50, 60, 70],
            'generate_tokens_speed': [100, 200, 300]
        }
        self.max_concurrency = 10

    @patch('ais_bench.benchmark.calculators.base_perf_metric_calculator.AISLogger')
    def test_init_datas_success(self, mock_logger_class):
        # 测试成功初始化数据
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        calculator = DefaultPerfMetricCalculator()
        calculator._init_datas(self.perf_details, self.max_concurrency)
        
        # 验证数据结构初始化
        self.assertEqual(calculator.stage_dict, {'total': [0, 1, 2]})
        self.assertEqual(calculator.max_concurrency, self.max_concurrency)
        self.assertIn('total', calculator.data_count)
        self.assertIn('total', calculator.decode_latencies)
        self.assertIn('total', calculator.success_count)
        self.assertIn('total', calculator.infer_time)
        self.assertIn('total', calculator.result)
        
        # 验证数据计数
        self.assertEqual(calculator.data_count['total'], 3)
        self.assertEqual(calculator.success_count['total'], 2)
        
        # 验证推理时间计算
        expected_infer_time = max(self.perf_details['end_time']) - min(self.perf_details['start_time'])
        self.assertEqual(calculator.infer_time['total'], expected_infer_time)

    @patch('ais_bench.benchmark.calculators.default_perf_metric_calculator.AISBenchDataContentError')
    @patch('ais_bench.benchmark.calculators.base_perf_metric_calculator.AISLogger')
    def test_init_datas_all_failed(self, mock_logger_class, mock_error_class):
        # 测试所有请求失败的情况
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        # 配置模拟异常
        mock_error_instance = ValueError("All requests failed")
        mock_error_class.side_effect = mock_error_instance
        
        # 创建所有请求都失败的测试数据
        failed_perf_details = self.perf_details.copy()
        failed_perf_details['success'] = [False, False, False]
        
        calculator = DefaultPerfMetricCalculator()
        
        # 验证抛出异常
        with self.assertRaises(ValueError) as context:
            calculator._init_datas(failed_perf_details, self.max_concurrency)
        
        # 验证异常内容
        self.assertEqual(context.exception, mock_error_instance)
        self.assertIn('All requests failed', str(context.exception))
        
        # 验证异常类被正确调用
        mock_error_class.assert_called_once()

    @patch('ais_bench.benchmark.calculators.base_perf_metric_calculator.AISLogger')
    def test_get_requests_id(self, mock_logger_class):
        # 测试获取请求ID
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        calculator = DefaultPerfMetricCalculator()
        request_ids = calculator._get_requests_id(self.perf_details)
        
        # 验证返回的是正确的ID列表
        self.assertEqual(request_ids, [0, 1, 2])
        
        # 测试空数据情况
        empty_perf_details = {'id': []}
        empty_request_ids = calculator._get_requests_id(empty_perf_details)
        self.assertEqual(empty_request_ids, [])

    @patch('ais_bench.benchmark.calculators.base_perf_metric_calculator.AISLogger')
    def test_process_result(self, mock_logger_class):
        # 测试处理结果
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        calculator = DefaultPerfMetricCalculator()
        calculator.stage_dict = {'total': [0, 1, 2]}
        # 初始化必要的数据结构
        calculator.data_count = {}
        calculator.decode_latencies = {}
        calculator.success_count = {}
        calculator.infer_time = {}
        calculator.result = {}
        calculator.convert_result = MagicMock(return_value={'converted_result': True})
        
        calculator._process_result(self.perf_details, 'total')
        
        # 验证数据存储
        self.assertEqual(calculator.data_count['total'], 3)
        self.assertEqual(calculator.decode_latencies['total'], self.perf_details['itl'])
        self.assertEqual(calculator.success_count['total'], 2)
        
        # 验证推理时间计算
        expected_infer_time = max(self.perf_details['end_time']) - min(self.perf_details['start_time'])
        self.assertEqual(calculator.infer_time['total'], expected_infer_time)
        
        # 验证调用了convert_result方法
        calculator.convert_result.assert_called_once()

    @patch('ais_bench.benchmark.calculators.base_perf_metric_calculator.AISLogger')
    def test_process_result_with_none_values(self, mock_logger_class):
        # 测试处理包含None值的结果
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        # 创建包含None值的测试数据
        perf_details_with_none = self.perf_details.copy()
        perf_details_with_none['generate_tokens_speed'] = None
        
        calculator = DefaultPerfMetricCalculator()
        calculator.stage_dict = {'total': [0, 1, 2]}
        # 初始化必要的数据结构
        calculator.data_count = {}
        calculator.decode_latencies = {}
        calculator.success_count = {}
        calculator.infer_time = {}
        calculator.result = {}
        calculator.convert_result = MagicMock(return_value={'converted_result': True})
        
        calculator._process_result(perf_details_with_none, 'total')
        
        # 验证仍然能正常处理
        self.assertEqual(calculator.data_count['total'], 3)

    @patch('ais_bench.benchmark.calculators.base_perf_metric_calculator.AISLogger')
    def test_full_calculation_flow(self, mock_logger_class):
        # 测试完整的计算流程
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        calculator = DefaultPerfMetricCalculator()
        
        # 模拟基类的方法
        calculator._calc_metrics = MagicMock()
        calculator._calc_common_metrics = MagicMock()
        
        # 初始化数据
        calculator._init_datas(self.perf_details, self.max_concurrency)
        
        # 验证数据初始化后可以调用计算方法
        calculator._calc_metrics()
        calculator._calc_common_metrics()
        
        calculator._calc_metrics.assert_called_once()
        calculator._calc_common_metrics.assert_called_once()

    @patch('ais_bench.benchmark.calculators.base_perf_metric_calculator.AISLogger')
    def test_metrics_calculation_integration(self, mock_logger_class):
        # 测试指标计算的集成
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        calculator = DefaultPerfMetricCalculator()
        
        # 由于convert_result在基类中已经测试过，这里我们模拟它的行为
        # 以便测试DefaultPerfMetricCalculator的特定逻辑
        mock_converted_result = {
            'E2EL': [0.1, 0.2, 0.3],
            'TTFT': [0.05, 0.06, 0.07],
            'TPOT': [0.02, 0.03, 0.04],
            'ITL': [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]],
            'InputTokens': [10, 20, 30],
            'OutputTokens': [50, 60, 70]
        }
        
        # 临时保存原始方法
        original_convert_result = calculator.convert_result
        try:
            # 替换为模拟方法
            calculator.convert_result = MagicMock(return_value=mock_converted_result)
            
            # 初始化数据
            calculator._init_datas(self.perf_details, self.max_concurrency)
            
            # 验证结果正确存储
            self.assertEqual(calculator.result['total'], mock_converted_result)
        finally:
            # 恢复原始方法
            calculator.convert_result = original_convert_result


if __name__ == '__main__':
    unittest.main()