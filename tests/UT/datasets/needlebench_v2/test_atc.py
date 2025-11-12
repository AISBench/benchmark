import unittest
import sys
import os
from unittest.mock import patch, MagicMock, mock_open

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

try:
    from ais_bench.benchmark.datasets.needlebench_v2.atc import (
        QuestionType,
        NeedleBenchATCDataset,
        relationship_generation_map_zh,
        relationship_generation_map_en,
        relationship_terms_zh_CN,
        relationship_terms_en,
        relationship_templates_zh_CN,
        relationship_templates_en
    )
    NEEDLEBENCH_ATC_AVAILABLE = True
except ImportError:
    NEEDLEBENCH_ATC_AVAILABLE = False


class NeedleBenchATCTestBase(unittest.TestCase):
    """NeedleBenchATC测试的基础类"""
    @classmethod
    def setUpClass(cls):
        if not NEEDLEBENCH_ATC_AVAILABLE:
            cls.skipTest(cls, "NeedleBenchATC modules not available")


class TestQuestionType(NeedleBenchATCTestBase):
    """测试QuestionType枚举"""
    
    def test_question_type_enum(self):
        """测试问题类型枚举值"""
        self.assertEqual(QuestionType.ELDEST_ANCESTOR.value, 0)
        self.assertEqual(QuestionType.NTH_ANCESTOR.value, 1)
        self.assertEqual(QuestionType.NTH_DESCENDANT.value, 2)
        self.assertEqual(QuestionType.RELATIONSHIP_DISTANCE.value, 3)


class TestConstants(NeedleBenchATCTestBase):
    """测试常量"""
    
    def test_relationship_maps(self):
        """测试关系映射"""
        self.assertIn('父亲', relationship_generation_map_zh)
        self.assertIn('father', relationship_generation_map_en)
        self.assertEqual(relationship_generation_map_zh['父亲'], 1)
        self.assertEqual(relationship_generation_map_en['father'], 1)
    
    def test_relationship_terms(self):
        """测试关系术语"""
        self.assertIsInstance(relationship_terms_zh_CN, list)
        self.assertIsInstance(relationship_terms_en, list)
        self.assertIn('父亲', relationship_terms_zh_CN)
        self.assertIn('father', relationship_terms_en)
    
    def test_relationship_templates(self):
        """测试关系模板"""
        self.assertIsInstance(relationship_templates_zh_CN, list)
        self.assertIsInstance(relationship_templates_en, list)
        self.assertGreater(len(relationship_templates_zh_CN), 0)
        self.assertGreater(len(relationship_templates_en), 0)


class TestNeedleBenchATCDataset(NeedleBenchATCTestBase):
    """测试NeedleBenchATCDataset类"""
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc.get_data_path')
    @patch('os.path.join')
    @patch('builtins.open')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc.random')
    def test_load_eldest_ancestor(self, mock_random, mock_open, mock_join, mock_get_path):
        """测试加载最年长祖先问题数据集"""
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.read.return_value = '{"English": "Name1,Name2,Name3,Name4", "Chinese": "名字1,名字2,名字3,名字4"}'
        mock_open.return_value = mock_file_handle
        
        # mock需要返回真实的关系术语和模板
        # choice会被多次调用：每次循环先选模板，再选关系术语
        # 对于num_needles=2（3个名字），会有2次循环，每次调用2次choice
        choice_count = [0]
        def choice_side_effect(items):
            if not items:
                return None
            choice_count[0] += 1
            # 奇数次返回模板，偶数次返回关系术语
            if choice_count[0] % 2 == 1:
                # 返回第一个模板（包含{relationship}占位符）
                return relationship_templates_en[0]
            else:
                # 返回一个在relationship_generation_map_en中的术语
                return 'father'
        
        mock_random.choice.side_effect = choice_side_effect
        mock_random.sample.return_value = ['Name1', 'Name2', 'Name3']  # num_needles=2需要3个名字
        mock_random.shuffle = lambda x: None
        
        result = NeedleBenchATCDataset.load(
            path='/test/path',
            file_name='names.json',
            num_needles=2,
            language='English',
            repeats=1,
            question_types=[QuestionType.ELDEST_ANCESTOR]
        )
        
        self.assertIsNotNone(result)
        # Dataset对象使用column_names属性
        self.assertTrue(hasattr(result, 'column_names'))
        self.assertIn('prompt', result.column_names)
        self.assertIn('answer', result.column_names)
        self.assertIn('question_type', result.column_names)
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc.get_data_path')
    @patch('os.path.join')
    @patch('builtins.open')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc.random')
    def test_load_nth_ancestor(self, mock_random, mock_open, mock_join, mock_get_path):
        """测试加载N代祖先问题数据集"""
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.read.return_value = '{"English": "Name1,Name2,Name3,Name4", "Chinese": "名字1,名字2,名字3,名字4"}'
        mock_open.return_value = mock_file_handle
        
        mock_random.choice.return_value = 'father'
        mock_random.sample.return_value = ['Name1', 'Name2', 'Name3']
        mock_random.shuffle = lambda x: None
        
        result = NeedleBenchATCDataset.load(
            path='/test/path',
            file_name='names.json',
            num_needles=2,
            language='English',
            repeats=1,
            question_types=[QuestionType.NTH_ANCESTOR]
        )
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'column_names'))
        self.assertIn('prompt', result.column_names)
        self.assertIn('answer', result.column_names)
        self.assertIn('question_type', result.column_names)
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc.get_data_path')
    @patch('os.path.join')
    @patch('builtins.open')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc.random')
    def test_load_nth_descendant(self, mock_random, mock_open, mock_join, mock_get_path):
        """测试加载N代子孙问题数据集"""
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.read.return_value = '{"English": "Name1,Name2,Name3,Name4", "Chinese": "名字1,名字2,名字3,名字4"}'
        mock_open.return_value = mock_file_handle
        
        mock_random.choice.return_value = 'father'
        mock_random.sample.return_value = ['Name1', 'Name2', 'Name3']
        mock_random.shuffle = lambda x: None
        
        result = NeedleBenchATCDataset.load(
            path='/test/path',
            file_name='names.json',
            num_needles=2,
            language='English',
            repeats=1,
            question_types=[QuestionType.NTH_DESCENDANT]
        )
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'column_names'))
        self.assertIn('prompt', result.column_names)
        self.assertIn('answer', result.column_names)
        self.assertIn('question_type', result.column_names)
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc.get_data_path')
    @patch('os.path.join')
    @patch('builtins.open')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc.random')
    def test_load_relationship_distance(self, mock_random, mock_open, mock_join, mock_get_path):
        """测试加载关系距离问题数据集"""
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.read.return_value = '{"English": "Name1,Name2,Name3,Name4", "Chinese": "名字1,名字2,名字3,名字4"}'
        mock_open.return_value = mock_file_handle
        
        mock_random.choice.return_value = 'father'
        mock_random.sample.return_value = ['Name1', 'Name2', 'Name3']
        mock_random.shuffle = lambda x: None
        
        result = NeedleBenchATCDataset.load(
            path='/test/path',
            file_name='names.json',
            num_needles=2,
            language='English',
            repeats=1,
            question_types=[QuestionType.RELATIONSHIP_DISTANCE]
        )
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'column_names'))
        self.assertIn('prompt', result.column_names)
        self.assertIn('answer', result.column_names)
        self.assertIn('question_type', result.column_names)
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc.get_data_path')
    @patch('os.path.join')
    @patch('builtins.open')
    def test_load_empty_question_types(self, mock_open, mock_join, mock_get_path):
        """测试空的问题类型列表"""
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        # Ensure json.load reads a proper JSON string
        mock_open.return_value.__enter__.return_value.read.return_value = '{"English": "Name1,Name2", "Chinese": "名字1,名字2"}'
        
        with self.assertRaises(ValueError):
            NeedleBenchATCDataset.load(
                path='/test/path',
                file_name='names.json',
                num_needles=2,
                language='English',
                repeats=1,
                question_types=[]
            )
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc.get_data_path')
    @patch('os.path.join')
    @patch('builtins.open')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc.random')
    def test_load_chinese(self, mock_random, mock_open, mock_join, mock_get_path):
        """测试加载中文数据集"""
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.read.return_value = '{"English": "Name1,Name2", "Chinese": "名字1,名字2,名字3,名字4"}'
        mock_open.return_value = mock_file_handle
        
        mock_random.choice.return_value = '父亲'
        mock_random.sample.return_value = ['名字1', '名字2', '名字3']
        mock_random.shuffle = lambda x: None
        
        result = NeedleBenchATCDataset.load(
            path='/test/path',
            file_name='names.json',
            num_needles=2,
            language='Chinese',
            repeats=1,
            question_types=[QuestionType.ELDEST_ANCESTOR]
        )
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'column_names'))
        self.assertIn('prompt', result.column_names)
        self.assertIn('answer', result.column_names)
        self.assertIn('question_type', result.column_names)
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc.get_data_path')
    @patch('os.path.join')
    @patch('builtins.open')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc.random')
    def test_load_nth_ancestor_chinese(self, mock_random, mock_open, mock_join, mock_get_path):
        """测试加载N代祖先问题数据集（中文）"""
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.read.return_value = '{"English": "Name1,Name2,Name3,Name4", "Chinese": "名字1,名字2,名字3,名字4"}'
        mock_open.return_value = mock_file_handle
        
        choice_count = [0]
        def choice_side_effect(items):
            if not items:
                return None
            choice_count[0] += 1
            if choice_count[0] % 2 == 1:
                return relationship_templates_zh_CN[0]
            else:
                return '父亲'
        
        mock_random.choice.side_effect = choice_side_effect
        mock_random.sample.return_value = ['名字1', '名字2', '名字3']
        mock_random.shuffle = lambda x: None
        
        result = NeedleBenchATCDataset.load(
            path='/test/path',
            file_name='names.json',
            num_needles=2,
            language='Chinese',
            repeats=1,
            question_types=[QuestionType.NTH_ANCESTOR]
        )
        
        self.assertIsNotNone(result)
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc.get_data_path')
    @patch('os.path.join')
    @patch('builtins.open')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc.random')
    def test_load_nth_descendant_chinese(self, mock_random, mock_open, mock_join, mock_get_path):
        """测试加载N代子孙问题数据集（中文）"""
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.read.return_value = '{"English": "Name1,Name2,Name3,Name4", "Chinese": "名字1,名字2,名字3,名字4"}'
        mock_open.return_value = mock_file_handle
        
        choice_count = [0]
        def choice_side_effect(items):
            if not items:
                return None
            choice_count[0] += 1
            if choice_count[0] % 2 == 1:
                return relationship_templates_zh_CN[0]
            else:
                return '父亲'
        
        mock_random.choice.side_effect = choice_side_effect
        mock_random.sample.return_value = ['名字1', '名字2', '名字3']
        mock_random.shuffle = lambda x: None
        
        result = NeedleBenchATCDataset.load(
            path='/test/path',
            file_name='names.json',
            num_needles=2,
            language='Chinese',
            repeats=1,
            question_types=[QuestionType.NTH_DESCENDANT]
        )
        
        self.assertIsNotNone(result)
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc.get_data_path')
    @patch('os.path.join')
    @patch('builtins.open')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc.random')
    def test_load_relationship_distance_chinese(self, mock_random, mock_open, mock_join, mock_get_path):
        """测试加载关系距离问题数据集（中文）"""
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.read.return_value = '{"English": "Name1,Name2,Name3,Name4", "Chinese": "名字1,名字2,名字3,名字4"}'
        mock_open.return_value = mock_file_handle
        
        choice_count = [0]
        def choice_side_effect(items):
            if not items:
                return None
            choice_count[0] += 1
            if choice_count[0] % 2 == 1:
                return relationship_templates_zh_CN[0]
            else:
                return '父亲'
        
        mock_random.choice.side_effect = choice_side_effect
        mock_random.sample.return_value = ['名字1', '名字2', '名字3']
        mock_random.shuffle = lambda x: None
        
        result = NeedleBenchATCDataset.load(
            path='/test/path',
            file_name='names.json',
            num_needles=2,
            language='Chinese',
            repeats=1,
            question_types=[QuestionType.RELATIONSHIP_DISTANCE]
        )
        
        self.assertIsNotNone(result)
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc.get_data_path')
    @patch('os.path.join')
    @patch('builtins.open')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc.random')
    def test_load_unsupported_language(self, mock_random, mock_open, mock_join, mock_get_path):
        """测试不支持的语言"""
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        # Include French key so we get past json lookup and reach language check
        mock_open.return_value.__enter__.return_value.read.return_value = '{"English": "Name1", "Chinese": "名字1", "French": "Jean,Paul"}'
        
        with self.assertRaises(ValueError):
            NeedleBenchATCDataset.load(
                path='/test/path',
                file_name='names.json',
                num_needles=1,
                language='French',
                repeats=1,
                question_types=[QuestionType.ELDEST_ANCESTOR]
            )


if __name__ == '__main__':
    unittest.main()

