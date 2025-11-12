import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

try:
    from ais_bench.benchmark.datasets.longbench.longbench_lsht import lsht_postprocess
    from ais_bench.benchmark.datasets.longbench.longbench_trec import trec_postprocess
    from ais_bench.benchmark.datasets.longbench.longbench_samsum import samsum_postprocess
    from ais_bench.benchmark.datasets.longbench.longbench_trivia_qa import triviaqa_postprocess
    LONGBENCH_POSTPROCESSORS_AVAILABLE = True
except ImportError:
    LONGBENCH_POSTPROCESSORS_AVAILABLE = False


class LongBenchPostprocessorsTestBase(unittest.TestCase):
    """LongBench后处理函数测试的基础类"""
    @classmethod
    def setUpClass(cls):
        if not LONGBENCH_POSTPROCESSORS_AVAILABLE:
            cls.skipTest(cls, "LongBench postprocessors modules not available")


class TestLshtPostprocess(LongBenchPostprocessorsTestBase):
    """测试lsht_postprocess函数"""
    
    def test_lsht_postprocess_basic(self):
        """测试基本后处理"""
        text = '\n\nfirst line\nsecond line'
        result = lsht_postprocess(text)
        self.assertEqual(result, 'first line')
    
    def test_lsht_postprocess_no_newline(self):
        """测试无换行的情况"""
        text = 'single line'
        result = lsht_postprocess(text)
        self.assertEqual(result, 'single line')
    
    def test_lsht_postprocess_empty(self):
        """测试空字符串"""
        text = ''
        result = lsht_postprocess(text)
        self.assertEqual(result, '')


class TestTrecPostprocess(LongBenchPostprocessorsTestBase):
    """测试trec_postprocess函数"""
    
    def test_trec_postprocess_basic(self):
        """测试基本后处理"""
        text = '\n\nfirst line\nsecond line'
        result = trec_postprocess(text)
        self.assertEqual(result, 'first line')
    
    def test_trec_postprocess_no_newline(self):
        """测试无换行的情况"""
        text = 'single line'
        result = trec_postprocess(text)
        self.assertEqual(result, 'single line')
    
    def test_trec_postprocess_empty(self):
        """测试空字符串"""
        text = ''
        result = trec_postprocess(text)
        self.assertEqual(result, '')


class TestSamsumPostprocess(LongBenchPostprocessorsTestBase):
    """测试samsum_postprocess函数"""
    
    def test_samsum_postprocess_basic(self):
        """测试基本后处理"""
        text = '\n\nfirst line\nsecond line'
        result = samsum_postprocess(text)
        self.assertEqual(result, 'first line')
    
    def test_samsum_postprocess_no_newline(self):
        """测试无换行的情况"""
        text = 'single line'
        result = samsum_postprocess(text)
        self.assertEqual(result, 'single line')
    
    def test_samsum_postprocess_empty(self):
        """测试空字符串"""
        text = ''
        result = samsum_postprocess(text)
        self.assertEqual(result, '')


class TestTriviaqaPostprocess(LongBenchPostprocessorsTestBase):
    """测试triviaqa_postprocess函数"""
    
    def test_triviaqa_postprocess_basic(self):
        """测试基本后处理"""
        text = '\n\nfirst line\nsecond line'
        result = triviaqa_postprocess(text)
        self.assertEqual(result, 'first line')
    
    def test_triviaqa_postprocess_no_newline(self):
        """测试无换行的情况"""
        text = 'single line'
        result = triviaqa_postprocess(text)
        self.assertEqual(result, 'single line')
    
    def test_triviaqa_postprocess_empty(self):
        """测试空字符串"""
        text = ''
        result = triviaqa_postprocess(text)
        self.assertEqual(result, '')


if __name__ == '__main__':
    unittest.main()

