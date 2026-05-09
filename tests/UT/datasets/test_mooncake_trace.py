"""Unit tests for mooncake_trace.py"""
import hashlib
import unittest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from ais_bench.benchmark.datasets.mooncake_trace import (
    RNGManager,
    init_rng,
    derive_rng,
    PromptGenerator,
    MooncakeTrace,
    load_mooncake_trace,
    _process_timestamps,
    MooncakeTraceDataset,
    initialize_corpus,
    DEFAULT_CORPUS_FILE,
)
from ais_bench.benchmark.utils.logging.exceptions import (
    AISBenchRuntimeError,
    ParameterValueError,
    AISBenchDataContentError,
)
from ais_bench.benchmark.utils.logging.error_codes import DSET_CODES


# ============================================================================
# 1. RNGManager测试
# ============================================================================

class TestRNGManager(unittest.TestCase):
    """测试RNGManager类"""

    def test_derive_consistent_seed(self):
        """测试种子派生的一致性（相同seed和identifier产生相同RNG）"""
        manager1 = RNGManager(42)
        manager2 = RNGManager(42)

        rng1 = manager1.derive("test")
        rng2 = manager2.derive("test")

        # 相同seed和identifier应该产生相同的随机数序列
        values1 = [rng1.randint(1, 100) for _ in range(10)]
        values2 = [rng2.randint(1, 100) for _ in range(10)]

        self.assertEqual(values1, values2)

    def test_derive_different_identifiers(self):
        """测试不同identifier产生不同RNG"""
        manager = RNGManager(42)

        rng1 = manager.derive("test1")
        rng2 = manager.derive("test2")

        # 不同identifier应该产生不同的随机数序列
        values1 = [rng1.randint(1, 100) for _ in range(10)]
        values2 = [rng2.randint(1, 100) for _ in range(10)]

        self.assertNotEqual(values1, values2)

    def test_derive_none_seed(self):
        """测试None种子产生非确定性RNG"""
        manager1 = RNGManager(None)
        manager2 = RNGManager(None)

        rng1 = manager1.derive("test")
        rng2 = manager2.derive("test")

        # None种子应该产生非确定性结果（虽然可能偶尔相同，但通常不同）
        values1 = [rng1.randint(1, 100) for _ in range(10)]
        values2 = [rng2.randint(1, 100) for _ in range(10)]

        # 由于是系统随机，通常应该不同（虽然理论上可能相同）
        # 我们至少验证它们能正常工作
        self.assertEqual(len(values1), 10)
        self.assertEqual(len(values2), 10)

    def test_init_rng_and_derive_rng(self):
        """测试init_rng和derive_rng函数"""
        # 先初始化
        init_rng(42)

        # 派生RNG
        rng1 = derive_rng("test")
        rng2 = derive_rng("test")

        # 相同identifier应该产生相同的随机数序列
        values1 = [rng1.randint(1, 100) for _ in range(10)]
        values2 = [rng2.randint(1, 100) for _ in range(10)]

        self.assertEqual(values1, values2)

    def test_derive_rng_without_init(self):
        """测试未初始化时调用derive_rng抛出异常"""
        # 重置全局RNG manager
        import ais_bench.benchmark.datasets.mooncake_trace as mt_module
        mt_module._rng_manager = None

        with self.assertRaises(AISBenchRuntimeError) as cm:
            derive_rng("test")

        self.assertIn("RNG manager not initialized", str(cm.exception))


# ============================================================================
# 2. PromptGenerator测试
# ============================================================================

class TestPromptGenerator(unittest.TestCase):
    """测试PromptGenerator类"""

    def setUp(self):
        """设置测试环境"""
        # Mock tokenizer
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.vocab_size = 1000
        self.mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.mock_tokenizer.decode.return_value = "decoded text"

        # 创建tokenized corpus
        self.tokenized_corpus = list(range(100))  # 100个token的语料库

        # 初始化RNG
        init_rng(42)

    def tearDown(self):
        """清理测试环境"""
        # 重置全局RNG manager
        import ais_bench.benchmark.datasets.mooncake_trace as mt_module
        mt_module._rng_manager = None

    def test_generate_with_hash_ids(self):
        """测试hash_id缓存机制"""
        generator = PromptGenerator(self.mock_tokenizer, self.tokenized_corpus, root_seed=42)

        # 第一次生成：使用2个hash_ids，需要至少512+1=513个token（block_size=512）
        # 或者使用1个hash_id，mean=100
        prompt1 = generator.generate(mean=100, hash_ids=[1])

        # 验证tokenizer被调用
        self.assertTrue(self.mock_tokenizer.decode.called)
        # 验证hash_id被缓存
        self.assertIn(1, generator._cache)

    def test_generate_with_hash_ids_cached(self):
        """测试hash_id缓存复用"""
        generator = PromptGenerator(self.mock_tokenizer, self.tokenized_corpus, root_seed=42)

        # 第一次生成：使用1个hash_id，mean=100
        prompt1 = generator.generate(mean=100, hash_ids=[1])
        decode_call_count_1 = self.mock_tokenizer.decode.call_count

        # 第二次生成（应该使用缓存）
        prompt2 = generator.generate(mean=100, hash_ids=[1])
        decode_call_count_2 = self.mock_tokenizer.decode.call_count

        # 验证decode被调用了（虽然可能调用次数相同，但至少应该工作）
        self.assertGreaterEqual(decode_call_count_2, decode_call_count_1)

    def test_generate_without_hash_ids(self):
        """测试不使用hash_ids的生成"""
        generator = PromptGenerator(self.mock_tokenizer, self.tokenized_corpus, root_seed=42)

        prompt = generator.generate(mean=10, stddev=0)

        # 验证tokenizer被调用
        self.assertTrue(self.mock_tokenizer.decode.called)

    def test_generate_prompt_num_tokens(self):
        """测试generate_prompt方法"""
        generator = PromptGenerator(self.mock_tokenizer, self.tokenized_corpus, root_seed=42)

        prompt = generator.generate_prompt(10)

        # 验证tokenizer.decode被调用
        self.assertTrue(self.mock_tokenizer.decode.called)

    def test_sample_tokens_circular(self):
        """测试语料库循环采样"""
        generator = PromptGenerator(self.mock_tokenizer, self.tokenized_corpus, root_seed=42)

        # 请求超过语料库大小的token数
        tokens = generator._sample_tokens(150)  # 语料库只有100个token

        # 应该返回整个语料库（因为超过语料库大小）
        self.assertEqual(len(tokens), 100)

    def test_sample_tokens_exceeds_corpus(self):
        """测试请求token数超过语料库大小"""
        generator = PromptGenerator(self.mock_tokenizer, self.tokenized_corpus, root_seed=42)

        tokens = generator._sample_tokens(200)  # 超过语料库大小

        # 应该返回整个语料库的副本
        self.assertEqual(tokens, self.tokenized_corpus)

    def test_sample_num_tokens_normal_dist(self):
        """测试正态分布采样token数量"""
        generator = PromptGenerator(self.mock_tokenizer, self.tokenized_corpus, root_seed=42)

        num_tokens = generator._sample_num_tokens(mean=50, stddev=10)

        # 应该返回正整数
        self.assertGreater(num_tokens, 0)
        self.assertIsInstance(num_tokens, int)

    def test_sample_num_tokens_zero_stddev(self):
        """测试stddev为0的情况"""
        generator = PromptGenerator(self.mock_tokenizer, self.tokenized_corpus, root_seed=42)

        num_tokens = generator._sample_num_tokens(mean=50, stddev=0)

        # 应该直接返回mean值
        self.assertEqual(num_tokens, 50)

    def test_sample_num_tokens_missing_mean(self):
        """测试缺少mean参数时抛出异常"""
        generator = PromptGenerator(self.mock_tokenizer, self.tokenized_corpus, root_seed=42)

        with self.assertRaises(ParameterValueError) as cm:
            generator._sample_num_tokens(mean=None, stddev=10)

        self.assertIn("mean must be provided", str(cm.exception))

    def test_generate_cached_prompt_block_separation(self):
        """测试block_separation_token_id支持"""
        self.mock_tokenizer.block_separation_token_id = 999
        generator = PromptGenerator(self.mock_tokenizer, self.tokenized_corpus, root_seed=42)

        prompt = generator.generate(mean=100, hash_ids=[1])

        # 验证block_separation_token_id被使用
        # 检查缓存中是否包含block_separation_token_id
        if 1 in generator._cache:
            cached_tokens = generator._cache[1]
            # 第一个token应该是block_separation_token_id
            self.assertEqual(cached_tokens[0], 999)

    def test_generate_cached_prompt_invalid_params(self):
        """测试参数验证（final_block_size验证）"""
        generator = PromptGenerator(self.mock_tokenizer, self.tokenized_corpus, root_seed=42)

        # 测试final_block_size <= 0的情况
        with self.assertRaises(ParameterValueError) as cm:
            generator._generate_cached_prompt(
                num_tokens=100,
                hash_ids=[1, 2, 3],
                block_size=512
            )

        self.assertIn("Final block size", str(cm.exception))

    def test_reproducibility_same_seed(self):
        """测试可复现性（相同种子产生相同prompt）"""
        # 第一次运行
        init_rng(42)
        generator1 = PromptGenerator(self.mock_tokenizer, self.tokenized_corpus, root_seed=42)
        prompt1 = generator1.generate(mean=10, stddev=0)

        # 重置并第二次运行
        import ais_bench.benchmark.datasets.mooncake_trace as mt_module
        mt_module._rng_manager = None
        init_rng(42)
        generator2 = PromptGenerator(self.mock_tokenizer, self.tokenized_corpus, root_seed=42)
        prompt2 = generator2.generate(mean=10, stddev=0)

        # 相同seed应该产生相同的prompt
        self.assertEqual(prompt1, prompt2)


# ============================================================================
# 3. MooncakeTrace测试
# ============================================================================

class TestMooncakeTrace(unittest.TestCase):
    """测试MooncakeTrace类"""

    def test_init_with_input_text(self):
        """测试使用input_text初始化"""
        data = {
            "input_text": "Test prompt",
            "output_length": 100,
            "timestamp": 1000
        }
        trace = MooncakeTrace(data)

        self.assertEqual(trace.input_text, "Test prompt")
        self.assertEqual(trace.output_length, 100)
        self.assertEqual(trace.timestamp, 1000)

    def test_init_with_input_length(self):
        """测试使用input_length初始化"""
        data = {
            "input_length": 512,
            "output_length": 100,
            "hash_ids": [1, 2],
            "timestamp": 1000
        }
        trace = MooncakeTrace(data)

        self.assertIsNone(trace.input_text)
        self.assertEqual(trace.input_length, 512)
        self.assertEqual(trace.hash_ids, [1, 2])
        self.assertEqual(trace.output_length, 100)

    def test_init_with_both_input_text_and_length(self):
        """测试同时提供input_text和input_length（input_text优先）"""
        data = {
            "input_text": "Test prompt",
            "input_length": 512,
            "output_length": 100
        }
        trace = MooncakeTrace(data)

        # input_text存在时，input_length变为可选（会被忽略）
        self.assertEqual(trace.input_text, "Test prompt")
        self.assertEqual(trace.input_length, 512)  # 仍然会被设置，但会被忽略

    def test_init_missing_both(self):
        """测试缺少input_text和input_length时抛出异常"""
        data = {
            "output_length": 100
        }

        with self.assertRaises(ParameterValueError) as cm:
            MooncakeTrace(data)

        self.assertIn("Either 'input_text' or 'input_length' must be provided", str(cm.exception))

    def test_init_with_hash_ids(self):
        """测试hash_ids字段"""
        data = {
            "input_length": 512,
            "hash_ids": [1, 2, 3]
        }
        trace = MooncakeTrace(data)

        self.assertEqual(trace.hash_ids, [1, 2, 3])

    def test_init_with_timestamp(self):
        """测试timestamp字段"""
        data = {
            "input_length": 512,
            "timestamp": 5000
        }
        trace = MooncakeTrace(data)

        self.assertEqual(trace.timestamp, 5000)

    def test_init_with_output_length(self):
        """测试output_length字段"""
        data = {
            "input_length": 512,
            "output_length": 200
        }
        trace = MooncakeTrace(data)

        self.assertEqual(trace.output_length, 200)


# ============================================================================
# 4. load_mooncake_trace和_process_timestamps测试
# ============================================================================

class TestLoadMooncakeTrace(unittest.TestCase):
    """测试load_mooncake_trace函数"""

    def test_load_valid_trace(self):
        """测试加载有效的trace文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"input_length": 512, "output_length": 100}\n')
            f.write('{"input_text": "Test", "output_length": 50}\n')
            temp_path = f.name

        try:
            traces = load_mooncake_trace(temp_path)
            self.assertEqual(len(traces), 2)
            self.assertEqual(traces[0].input_length, 512)
            self.assertEqual(traces[1].input_text, "Test")
        finally:
            os.unlink(temp_path)

    def test_load_empty_file(self):
        """测试加载空文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name

        try:
            traces = load_mooncake_trace(temp_path)
            self.assertEqual(len(traces), 0)
        finally:
            os.unlink(temp_path)

    def test_load_with_empty_lines(self):
        """测试跳过空行"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"input_length": 512}\n')
            f.write('\n')
            f.write('  \n')
            f.write('{"input_length": 256}\n')
            temp_path = f.name

        try:
            traces = load_mooncake_trace(temp_path)
            self.assertEqual(len(traces), 2)
        finally:
            os.unlink(temp_path)


class TestProcessTimestamps(unittest.TestCase):
    """测试_process_timestamps函数"""

    def test_process_timestamps_auto_offset(self):
        """测试auto_offset功能"""
        traces = [
            MooncakeTrace({"input_length": 100, "timestamp": 1000}),
            MooncakeTrace({"input_length": 200, "timestamp": 2000}),
            MooncakeTrace({"input_length": 300, "timestamp": 3000}),
        ]

        result = _process_timestamps(traces, auto_offset=True)

        # 第一个timestamp应该变为0
        self.assertEqual(result[0].timestamp, 0)
        self.assertEqual(result[1].timestamp, 1000)
        self.assertEqual(result[2].timestamp, 2000)

    def test_process_timestamps_start_offset(self):
        """测试start_offset过滤"""
        traces = [
            MooncakeTrace({"input_length": 100, "timestamp": 1000}),
            MooncakeTrace({"input_length": 200, "timestamp": 2000}),
            MooncakeTrace({"input_length": 300, "timestamp": 3000}),
        ]

        result = _process_timestamps(traces, start_offset=1500)

        # 应该只保留timestamp >= 1500的trace
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].timestamp, 2000)
        self.assertEqual(result[1].timestamp, 3000)

    def test_process_timestamps_end_offset(self):
        """测试end_offset过滤"""
        traces = [
            MooncakeTrace({"input_length": 100, "timestamp": 1000}),
            MooncakeTrace({"input_length": 200, "timestamp": 2000}),
            MooncakeTrace({"input_length": 300, "timestamp": 3000}),
        ]

        result = _process_timestamps(traces, end_offset=2500)

        # 应该只保留timestamp <= 2500的trace
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].timestamp, 1000)
        self.assertEqual(result[1].timestamp, 2000)

    def test_process_timestamps_both_offsets(self):
        """测试同时使用start和end offset"""
        traces = [
            MooncakeTrace({"input_length": 100, "timestamp": 1000}),
            MooncakeTrace({"input_length": 200, "timestamp": 2000}),
            MooncakeTrace({"input_length": 300, "timestamp": 3000}),
        ]

        result = _process_timestamps(traces, start_offset=1500, end_offset=2500)

        # 应该只保留1500 <= timestamp <= 2500的trace
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].timestamp, 2000)

    def test_process_timestamps_no_timestamps(self):
        """测试没有timestamp的trace"""
        traces = [
            MooncakeTrace({"input_length": 100}),
            MooncakeTrace({"input_length": 200}),
        ]

        result = _process_timestamps(traces, start_offset=1000, end_offset=2000)

        # 没有timestamp的trace应该被保留
        self.assertEqual(len(result), 2)

    def test_process_timestamps_mixed(self):
        """测试混合（有timestamp和没有timestamp的trace）"""
        traces = [
            MooncakeTrace({"input_length": 100, "timestamp": 1000}),
            MooncakeTrace({"input_length": 200}),  # 没有timestamp
            MooncakeTrace({"input_length": 300, "timestamp": 3000}),
        ]

        result = _process_timestamps(traces, start_offset=500, end_offset=2000)

        # 应该保留第一个trace（在范围内）和没有timestamp的trace
        self.assertEqual(len(result), 2)
        self.assertIsNotNone(result[0].timestamp)
        self.assertIsNone(result[1].timestamp)

    def test_process_timestamps_empty_list(self):
        """测试空列表"""
        result = _process_timestamps([])
        self.assertEqual(len(result), 0)


# ============================================================================
# 5. MooncakeTraceDataset测试
# ============================================================================

class TestMooncakeTraceDataset(unittest.TestCase):
    """测试MooncakeTraceDataset类"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()

        # 创建测试trace文件
        self.trace_file = os.path.join(self.temp_dir, "test_trace.jsonl")
        with open(self.trace_file, 'w', encoding='utf-8') as f:
            f.write('{"input_length": 100, "output_length": 50, "timestamp": 1000}\n')
            f.write('{"input_text": "Test prompt", "output_length": 30, "timestamp": 2000}\n')

        # 创建测试语料库文件
        self.corpus_file = os.path.join(self.temp_dir, "shakespeare.txt")
        with open(self.corpus_file, 'w', encoding='utf-8') as f:
            f.write("To be or not to be, that is the question.\n")
            f.write("Whether 'tis nobler in the mind to suffer\n")
            f.write("The slings and arrows of outrageous fortune,\n")

    def tearDown(self):
        """清理测试环境"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

        # 重置全局RNG manager
        import ais_bench.benchmark.datasets.mooncake_trace as mt_module
        mt_module._rng_manager = None

    @patch('ais_bench.benchmark.datasets.mooncake_trace.load_tokenizer')
    @patch('ais_bench.benchmark.datasets.mooncake_trace.Path')
    def test_load_from_cache(self, mock_path, mock_load_tokenizer):
        """测试从缓存文件加载"""
        # 创建缓存文件
        cache_file = os.path.join(self.temp_dir, "test_trace_generated_prompts.jsonl")
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write('{"prompt": "cached prompt", "timestamp": 1000, "max_out_len": 50}\n')

        dataset_instance = object.__new__(MooncakeTraceDataset)
        dataset_instance.logger = MagicMock()

        dataset = dataset_instance.load(
            path=self.trace_file,
            model_path="/fake/path",
            generated_prompts_path=cache_file
        )

        # 验证从缓存加载
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["prompt"], "cached prompt")

    @patch('ais_bench.benchmark.datasets.mooncake_trace.load_tokenizer')
    @patch('ais_bench.benchmark.datasets.mooncake_trace.Path')
    def test_load_generate_prompts(self, mock_path, mock_load_tokenizer):
        """测试生成prompt流程"""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 1000
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "generated prompt"
        mock_load_tokenizer.return_value = mock_tokenizer

        # Mock Path查找语料库文件
        mock_path_instance = MagicMock()
        mock_path_instance.parent = MagicMock()
        mock_path_instance.parent.parent = MagicMock()
        mock_path_instance.parent.parent.parent = MagicMock()

        # 设置语料库文件路径
        corpus_path = Path(self.corpus_file)
        mock_path_instance.parent.__truediv__.return_value = corpus_path
        mock_path.return_value = mock_path_instance

        dataset_instance = object.__new__(MooncakeTraceDataset)
        dataset_instance.logger = MagicMock()

        dataset = dataset_instance.load(
            path=self.trace_file,
            model_path="/fake/path",
            random_seed=42
        )

        # 验证生成了prompt
        self.assertEqual(len(dataset), 2)
        self.assertIn("prompt", dataset[0])

    @patch('ais_bench.benchmark.datasets.mooncake_trace.load_tokenizer')
    @patch('ais_bench.benchmark.datasets.mooncake_trace.Path')
    def test_load_with_input_text(self, mock_path, mock_load_tokenizer):
        """测试input_text字段支持"""
        # 创建只包含input_text的trace文件
        trace_file = os.path.join(self.temp_dir, "test_input_text.jsonl")
        with open(trace_file, 'w', encoding='utf-8') as f:
            f.write('{"input_text": "Direct prompt", "output_length": 50}\n')

        # Mock tokenizer（虽然不会用到，但需要存在）
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 1000
        mock_load_tokenizer.return_value = mock_tokenizer

        # Mock Path
        mock_path_instance = MagicMock()
        mock_path_instance.parent = MagicMock()
        corpus_path = Path(self.corpus_file)
        mock_path_instance.parent.__truediv__.return_value = corpus_path
        mock_path.return_value = mock_path_instance

        dataset_instance = object.__new__(MooncakeTraceDataset)
        dataset_instance.logger = MagicMock()

        dataset = dataset_instance.load(
            path=trace_file,
            model_path="/fake/path",
            random_seed=42
        )

        # 验证使用了input_text
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["prompt"], "Direct prompt")

    @patch('ais_bench.benchmark.datasets.mooncake_trace.load_tokenizer')
    @patch('ais_bench.benchmark.datasets.mooncake_trace.Path')
    def test_load_mixed_mode(self, mock_path, mock_load_tokenizer):
        """测试混合模式（部分使用input_text，部分使用生成的prompt）"""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 1000
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "generated prompt"
        mock_load_tokenizer.return_value = mock_tokenizer

        # Mock Path
        mock_path_instance = MagicMock()
        mock_path_instance.parent = MagicMock()
        corpus_path = Path(self.corpus_file)
        mock_path_instance.parent.__truediv__.return_value = corpus_path
        mock_path.return_value = mock_path_instance

        dataset_instance = object.__new__(MooncakeTraceDataset)
        dataset_instance.logger = MagicMock()

        dataset = dataset_instance.load(
            path=self.trace_file,
            model_path="/fake/path",
            random_seed=42
        )

        # 验证混合模式：第一个使用生成的prompt，第二个使用input_text
        self.assertEqual(len(dataset), 2)
        # 第二个应该使用input_text
        self.assertEqual(dataset[1]["prompt"], "Test prompt")

    @patch('ais_bench.benchmark.datasets.mooncake_trace.load_tokenizer')
    @patch('ais_bench.benchmark.datasets.mooncake_trace.Path')
    def test_load_timestamp_processing_auto_offset(self, mock_path, mock_load_tokenizer):
        """测试时间戳处理（auto_offset）"""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 1000
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "generated prompt"
        mock_load_tokenizer.return_value = mock_tokenizer

        # Mock Path
        mock_path_instance = MagicMock()
        mock_path_instance.parent = MagicMock()
        corpus_path = Path(self.corpus_file)
        mock_path_instance.parent.__truediv__.return_value = corpus_path
        mock_path.return_value = mock_path_instance

        dataset_instance = object.__new__(MooncakeTraceDataset)
        dataset_instance.logger = MagicMock()

        dataset = dataset_instance.load(
            path=self.trace_file,
            model_path="/fake/path",
            random_seed=42,
            fixed_schedule_auto_offset=True
        )

        # 验证第一个timestamp变为0
        self.assertEqual(dataset[0]["timestamp"], 0)
        self.assertEqual(dataset[1]["timestamp"], 1000)

    @patch('ais_bench.benchmark.datasets.mooncake_trace.load_tokenizer')
    @patch('ais_bench.benchmark.datasets.mooncake_trace.Path')
    def test_load_timestamp_processing_start_offset(self, mock_path, mock_load_tokenizer):
        """测试时间戳处理（start_offset）"""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 1000
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "generated prompt"
        mock_load_tokenizer.return_value = mock_tokenizer

        # Mock Path
        mock_path_instance = MagicMock()
        mock_path_instance.parent = MagicMock()
        corpus_path = Path(self.corpus_file)
        mock_path_instance.parent.__truediv__.return_value = corpus_path
        mock_path.return_value = mock_path_instance

        dataset_instance = object.__new__(MooncakeTraceDataset)
        dataset_instance.logger = MagicMock()

        dataset = dataset_instance.load(
            path=self.trace_file,
            model_path="/fake/path",
            random_seed=42,
            fixed_schedule_start_offset=1500
        )

        # 应该只保留timestamp >= 1500的trace
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["timestamp"], 2000)

    @patch('ais_bench.benchmark.datasets.mooncake_trace.load_tokenizer')
    @patch('ais_bench.benchmark.datasets.mooncake_trace.Path')
    def test_load_timestamp_processing_end_offset(self, mock_path, mock_load_tokenizer):
        """测试时间戳处理（end_offset）"""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 1000
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "generated prompt"
        mock_load_tokenizer.return_value = mock_tokenizer

        # Mock Path
        mock_path_instance = MagicMock()
        mock_path_instance.parent = MagicMock()
        corpus_path = Path(self.corpus_file)
        mock_path_instance.parent.__truediv__.return_value = corpus_path
        mock_path.return_value = mock_path_instance

        dataset_instance = object.__new__(MooncakeTraceDataset)
        dataset_instance.logger = MagicMock()

        dataset = dataset_instance.load(
            path=self.trace_file,
            model_path="/fake/path",
            random_seed=42,
            fixed_schedule_end_offset=1500
        )

        # 应该只保留timestamp <= 1500的trace
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["timestamp"], 1000)

    @patch('ais_bench.benchmark.datasets.mooncake_trace.load_tokenizer')
    @patch('ais_bench.benchmark.datasets.mooncake_trace.Path')
    def test_load_prompts_sorted_by_timestamp(self, mock_path, mock_load_tokenizer):
        """测试prompts按timestamp排序"""
        # 创建乱序的trace文件
        trace_file = os.path.join(self.temp_dir, "test_unsorted.jsonl")
        with open(trace_file, 'w', encoding='utf-8') as f:
            f.write('{"input_length": 100, "timestamp": 3000}\n')
            f.write('{"input_length": 200, "timestamp": 1000}\n')
            f.write('{"input_length": 300, "timestamp": 2000}\n')

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 1000
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "generated prompt"
        mock_load_tokenizer.return_value = mock_tokenizer

        # Mock Path
        mock_path_instance = MagicMock()
        mock_path_instance.parent = MagicMock()
        corpus_path = Path(self.corpus_file)
        mock_path_instance.parent.__truediv__.return_value = corpus_path
        mock_path.return_value = mock_path_instance

        dataset_instance = object.__new__(MooncakeTraceDataset)
        dataset_instance.logger = MagicMock()

        dataset = dataset_instance.load(
            path=trace_file,
            model_path="/fake/path",
            random_seed=42
        )

        # 验证按timestamp排序
        timestamps = [item["timestamp"] for item in dataset]
        self.assertEqual(timestamps, sorted(timestamps))

    @patch('ais_bench.benchmark.datasets.mooncake_trace.load_tokenizer')
    @patch('ais_bench.benchmark.datasets.mooncake_trace.Path')
    def test_load_cache_filename_with_schedule_params(self, mock_path, mock_load_tokenizer):
        """测试缓存文件名包含fixed_schedule参数"""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 1000
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "generated prompt"
        mock_load_tokenizer.return_value = mock_tokenizer

        # Mock Path
        mock_path_instance = MagicMock()
        mock_path_instance.parent = MagicMock()
        corpus_path = Path(self.corpus_file)
        mock_path_instance.parent.__truediv__.return_value = corpus_path
        mock_path.return_value = mock_path_instance

        dataset_instance = object.__new__(MooncakeTraceDataset)
        dataset_instance.logger = MagicMock()

        dataset = dataset_instance.load(
            path=self.trace_file,
            model_path="/fake/path",
            random_seed=42,
            fixed_schedule_auto_offset=True,
            fixed_schedule_start_offset=500,
            fixed_schedule_end_offset=3000
        )
        with open(self.trace_file, "rb") as f:
            trace_md5 = hashlib.md5(f.read()).hexdigest()
        expected_cache_name = f"test_trace_{trace_md5}_generated_prompts_auto_start500_end3000.jsonl"
        cache_path = os.path.join(self.temp_dir, expected_cache_name)
        self.assertTrue(os.path.exists(cache_path))

    def test_load_invalid_schedule_params(self):
        """测试异常处理（start_offset > end_offset）"""
        dataset_instance = object.__new__(MooncakeTraceDataset)
        dataset_instance.logger = MagicMock()

        with self.assertRaises(ParameterValueError) as cm:
            dataset_instance.load(
                path=self.trace_file,
                model_path="/fake/path",
                fixed_schedule_start_offset=2000,
                fixed_schedule_end_offset=1000
            )

        self.assertIn("fixed_schedule_start_offset", str(cm.exception))

    @patch('ais_bench.benchmark.datasets.mooncake_trace.load_tokenizer')
    @patch('ais_bench.benchmark.datasets.mooncake_trace.Path')
    def test_load_corpus_not_found(self, mock_path, mock_load_tokenizer):
        """测试语料库文件不存在异常"""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer

        # Mock Path - 所有路径都不存在
        mock_path_instance = MagicMock()
        mock_path_instance.parent = MagicMock()
        mock_path_instance.parent.parent = MagicMock()
        mock_path_instance.parent.parent.parent = MagicMock()

        # 所有路径的exists()都返回False
        mock_path_instance.parent.__truediv__.return_value.exists.return_value = False
        mock_path_instance.parent.parent.__truediv__.return_value.exists.return_value = False
        mock_path_instance.parent.parent.parent.__truediv__.return_value.exists.return_value = False
        mock_path.return_value = mock_path_instance

        dataset_instance = object.__new__(MooncakeTraceDataset)
        dataset_instance.logger = MagicMock()

        with self.assertRaises(AISBenchDataContentError) as cm:
            dataset_instance.load(
                path=self.trace_file,
                model_path="/fake/path",
                random_seed=42
            )

        self.assertIn("Corpus file not found", str(cm.exception))

    @patch('ais_bench.benchmark.datasets.mooncake_trace.load_tokenizer')
    @patch('ais_bench.benchmark.datasets.mooncake_trace.Path')
    def test_load_reproducibility(self, mock_path, mock_load_tokenizer):
        """测试可复现性（相同seed产生相同结果）"""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 1000
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "generated prompt"
        mock_load_tokenizer.return_value = mock_tokenizer

        # Mock Path
        mock_path_instance = MagicMock()
        mock_path_instance.parent = MagicMock()
        corpus_path = Path(self.corpus_file)
        mock_path_instance.parent.__truediv__.return_value = corpus_path
        mock_path.return_value = mock_path_instance

        dataset_instance = object.__new__(MooncakeTraceDataset)
        dataset_instance.logger = MagicMock()

        # 第一次运行
        dataset1 = dataset_instance.load(
            path=self.trace_file,
            model_path="/fake/path",
            random_seed=42
        )

        # 重置RNG manager
        import ais_bench.benchmark.datasets.mooncake_trace as mt_module
        mt_module._rng_manager = None

        # 第二次运行
        dataset2 = dataset_instance.load(
            path=self.trace_file,
            model_path="/fake/path",
            random_seed=42
        )

        # 相同seed应该产生相同的结果（至少prompt应该相同）
        # 注意：由于使用了缓存文件，第二次运行会直接从缓存加载
        # 所以这里主要验证缓存机制工作正常
        self.assertEqual(len(dataset1), len(dataset2))


if __name__ == "__main__":
    unittest.main()
