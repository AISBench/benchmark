import unittest
import sys
import os
from unittest.mock import patch, MagicMock
from datasets import Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

try:
    from ais_bench.benchmark.datasets.longbench.longbench_2wikim_qa import LongBench2wikimqaDataset
    from ais_bench.benchmark.datasets.longbench.longbench_dureader import LongBenchdureaderDataset
    from ais_bench.benchmark.datasets.longbench.longbench_gov_report import LongBenchgov_reportDataset
    from ais_bench.benchmark.datasets.longbench.longbench_hotpot_qa import LongBenchhotpotqaDataset
    from ais_bench.benchmark.datasets.longbench.longbench_lcc import LongBenchlccDataset
    from ais_bench.benchmark.datasets.longbench.longbench_lsht import LongBenchlshtDataset
    from ais_bench.benchmark.datasets.longbench.longbench_multi_news import LongBenchmulti_newsDataset
    from ais_bench.benchmark.datasets.longbench.longbench_multifieldqa_en import LongBenchmultifieldqa_enDataset
    from ais_bench.benchmark.datasets.longbench.longbench_multifieldqa_zh import LongBenchmultifieldqa_zhDataset
    from ais_bench.benchmark.datasets.longbench.longbench_musique import LongBenchmusiqueDataset
    from ais_bench.benchmark.datasets.longbench.longbench_narrative_qa import LongBenchnarrativeqaDataset
    from ais_bench.benchmark.datasets.longbench.longbench_passage_count import LongBenchpassage_countDataset
    from ais_bench.benchmark.datasets.longbench.longbench_passage_retrieval_en import LongBenchpassage_retrieval_enDataset
    from ais_bench.benchmark.datasets.longbench.longbench_passage_retrieval_zh import LongBenchpassage_retrieval_zhDataset
    from ais_bench.benchmark.datasets.longbench.longbench_qasper import LongBenchqasperDataset
    from ais_bench.benchmark.datasets.longbench.longbench_qmsum import LongBenchqmsumDataset
    from ais_bench.benchmark.datasets.longbench.longbench_repobench import LongBenchrepobenchDataset
    from ais_bench.benchmark.datasets.longbench.longbench_samsum import LongBenchsamsumDataset
    from ais_bench.benchmark.datasets.longbench.longbench_trec import LongBenchtrecDataset
    from ais_bench.benchmark.datasets.longbench.longbench_trivia_qa import LongBenchtriviaqaDataset
    from ais_bench.benchmark.datasets.longbench.longbench_vcsum import LongBenchvcsumDataset
    LONGBENCH_DATASETS_AVAILABLE = True
except ImportError:
    LONGBENCH_DATASETS_AVAILABLE = False


class LongBenchDatasetsTestBase(unittest.TestCase):
    """LongBench数据集测试的基础类"""
    @classmethod
    def setUpClass(cls):
        if not LONGBENCH_DATASETS_AVAILABLE:
            cls.skipTest(cls, "LongBench datasets modules not available")


class TestLongBench2wikimqaDataset(LongBenchDatasetsTestBase):
    """测试LongBench2wikimqaDataset类"""
    
    @patch('ais_bench.benchmark.datasets.longbench.longbench_2wikim_qa.load_dataset')
    @patch('ais_bench.benchmark.datasets.longbench.longbench_2wikim_qa.get_data_path')
    def test_load(self, mock_get_path, mock_load_dataset):
        """测试加载数据集"""
        mock_get_path.return_value = '/fake/path'
        
        # 创建模拟的数据集split对象，模拟dataset[split]['field'][i]的访问
        mock_test_split = MagicMock()
        type(mock_test_split).__len__ = MagicMock(return_value=2)
        
        # 模拟字段访问：dataset[split]['input'][i] 返回列表
        mock_test_split.__getitem__ = MagicMock(side_effect=lambda key: {
            'input': ['question1', 'question2'],
            'context': ['context1', 'context2'],
            'answers': [['answer1'], ['answer2']]
        }[key])
        
        # 创建模拟的数据集对象
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = mock_test_split
        # 模拟dataset[split] = Dataset.from_list(...)
        mock_dataset.__setitem__ = MagicMock()
        
        mock_load_dataset.return_value = mock_dataset
        
        result = LongBench2wikimqaDataset.load(path='/test/path', name='test_name')
        
        self.assertTrue(mock_load_dataset.called)
        self.assertTrue(mock_dataset.__setitem__.called)


class TestLongBenchdureaderDataset(LongBenchDatasetsTestBase):
    """测试LongBenchdureaderDataset类"""
    
    @patch('ais_bench.benchmark.datasets.longbench.longbench_dureader.load_dataset')
    @patch('ais_bench.benchmark.datasets.longbench.longbench_dureader.get_data_path')
    def test_load(self, mock_get_path, mock_load_dataset):
        """测试加载数据集"""
        mock_get_path.return_value = '/fake/path'
        
        # 模拟split对象
        mock_test_split = MagicMock()
        type(mock_test_split).__len__ = MagicMock(return_value=1)
        mock_test_split.__getitem__ = MagicMock(side_effect=lambda key: {
            'input': ['question1'],
            'context': ['context1'],
            'answers': [['answer1']]
        }[key])
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = mock_test_split
        mock_dataset.__setitem__ = MagicMock()
        
        mock_load_dataset.return_value = mock_dataset
        
        result = LongBenchdureaderDataset.load(path='/test/path', name='test_name')
        
        self.assertTrue(mock_load_dataset.called)
        self.assertTrue(mock_dataset.__setitem__.called)


class TestLongBenchgov_reportDataset(LongBenchDatasetsTestBase):
    """测试LongBenchgov_reportDataset类"""
    
    @patch('ais_bench.benchmark.datasets.longbench.longbench_gov_report.load_dataset')
    @patch('ais_bench.benchmark.datasets.longbench.longbench_gov_report.get_data_path')
    def test_load(self, mock_get_path, mock_load_dataset):
        """测试加载数据集（无input字段）"""
        mock_get_path.return_value = '/fake/path'
        
        # 模拟split对象
        mock_test_split = MagicMock()
        type(mock_test_split).__len__ = MagicMock(return_value=1)
        mock_test_split.__getitem__ = MagicMock(side_effect=lambda key: {
            'context': ['context1'],
            'answers': [['answer1']]
        }[key])
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = mock_test_split
        mock_dataset.__setitem__ = MagicMock()
        
        mock_load_dataset.return_value = mock_dataset
        
        result = LongBenchgov_reportDataset.load(path='/test/path', name='test_name')
        
        self.assertTrue(mock_load_dataset.called)
        self.assertTrue(mock_dataset.__setitem__.called)


class TestLongBenchhotpotqaDataset(LongBenchDatasetsTestBase):
    """测试LongBenchhotpotqaDataset类"""
    
    @patch('ais_bench.benchmark.datasets.longbench.longbench_hotpot_qa.load_dataset')
    @patch('ais_bench.benchmark.datasets.longbench.longbench_hotpot_qa.get_data_path')
    def test_load(self, mock_get_path, mock_load_dataset):
        """测试加载数据集"""
        mock_get_path.return_value = '/fake/path'
        
        # 模拟split对象
        mock_test_split = MagicMock()
        type(mock_test_split).__len__ = MagicMock(return_value=1)
        mock_test_split.__getitem__ = MagicMock(side_effect=lambda key: {
            'input': ['question1'],
            'context': ['context1'],
            'answers': [['answer1']]
        }[key])
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = mock_test_split
        mock_dataset.__setitem__ = MagicMock()
        
        mock_load_dataset.return_value = mock_dataset
        
        result = LongBenchhotpotqaDataset.load(path='/test/path', name='test_name')
        
        self.assertTrue(mock_load_dataset.called)
        self.assertTrue(mock_dataset.__setitem__.called)


class TestLongBenchlccDataset(LongBenchDatasetsTestBase):
    """测试LongBenchlccDataset类"""
    
    @patch('ais_bench.benchmark.datasets.longbench.longbench_lcc.load_dataset')
    @patch('ais_bench.benchmark.datasets.longbench.longbench_lcc.get_data_path')
    def test_load(self, mock_get_path, mock_load_dataset):
        """测试加载数据集（无input字段）"""
        mock_get_path.return_value = '/fake/path'
        
        # 模拟split对象
        mock_test_split = MagicMock()
        type(mock_test_split).__len__ = MagicMock(return_value=1)
        mock_test_split.__getitem__ = MagicMock(side_effect=lambda key: {
            'context': ['context1'],
            'answers': [['answer1']]
        }[key])
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = mock_test_split
        mock_dataset.__setitem__ = MagicMock()
        
        mock_load_dataset.return_value = mock_dataset
        
        result = LongBenchlccDataset.load(path='/test/path', name='test_name')
        
        self.assertTrue(mock_load_dataset.called)
        self.assertTrue(mock_dataset.__setitem__.called)


class TestLongBenchlshtDataset(LongBenchDatasetsTestBase):
    """测试LongBenchlshtDataset类"""
    
    @patch('ais_bench.benchmark.datasets.longbench.longbench_lsht.load_dataset')
    @patch('ais_bench.benchmark.datasets.longbench.longbench_lsht.get_data_path')
    def test_load(self, mock_get_path, mock_load_dataset):
        """测试加载数据集（包含all_labels）"""
        mock_get_path.return_value = '/fake/path'
        
        # 模拟split对象
        mock_test_split = MagicMock()
        type(mock_test_split).__len__ = MagicMock(return_value=1)
        mock_test_split.__getitem__ = MagicMock(side_effect=lambda key: {
            'input': ['question1'],
            'context': ['context1'],
            'answers': [['answer1']],
            'all_classes': [['class1', 'class2']]
        }[key])
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = mock_test_split
        mock_dataset.__setitem__ = MagicMock()
        
        mock_load_dataset.return_value = mock_dataset
        
        result = LongBenchlshtDataset.load(path='/test/path', name='test_name')
        
        self.assertTrue(mock_load_dataset.called)
        self.assertTrue(mock_dataset.__setitem__.called)


class TestLongBenchmulti_newsDataset(LongBenchDatasetsTestBase):
    """测试LongBenchmulti_newsDataset类"""
    
    @patch('ais_bench.benchmark.datasets.longbench.longbench_multi_news.load_dataset')
    @patch('ais_bench.benchmark.datasets.longbench.longbench_multi_news.get_data_path')
    def test_load(self, mock_get_path, mock_load_dataset):
        """测试加载数据集（无input字段）"""
        mock_get_path.return_value = '/fake/path'
        
        # 模拟split对象
        mock_test_split = MagicMock()
        type(mock_test_split).__len__ = MagicMock(return_value=1)
        mock_test_split.__getitem__ = MagicMock(side_effect=lambda key: {
            'context': ['context1'],
            'answers': [['answer1']]
        }[key])
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = mock_test_split
        mock_dataset.__setitem__ = MagicMock()
        
        mock_load_dataset.return_value = mock_dataset
        
        result = LongBenchmulti_newsDataset.load(path='/test/path', name='test_name')
        
        self.assertTrue(mock_load_dataset.called)
        self.assertTrue(mock_dataset.__setitem__.called)


class TestLongBenchmultifieldqa_enDataset(LongBenchDatasetsTestBase):
    """测试LongBenchmultifieldqa_enDataset类"""
    
    @patch('ais_bench.benchmark.datasets.longbench.longbench_multifieldqa_en.load_dataset')
    @patch('ais_bench.benchmark.datasets.longbench.longbench_multifieldqa_en.get_data_path')
    def test_load(self, mock_get_path, mock_load_dataset):
        """测试加载数据集"""
        mock_get_path.return_value = '/fake/path'
        
        # 模拟split对象
        mock_test_split = MagicMock()
        type(mock_test_split).__len__ = MagicMock(return_value=1)
        mock_test_split.__getitem__ = MagicMock(side_effect=lambda key: {
            'input': ['question1'],
            'context': ['context1'],
            'answers': [['answer1']]
        }[key])
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = mock_test_split
        mock_dataset.__setitem__ = MagicMock()
        
        mock_load_dataset.return_value = mock_dataset
        
        result = LongBenchmultifieldqa_enDataset.load(path='/test/path', name='test_name')
        
        self.assertTrue(mock_load_dataset.called)
        self.assertTrue(mock_dataset.__setitem__.called)


class TestLongBenchmultifieldqa_zhDataset(LongBenchDatasetsTestBase):
    """测试LongBenchmultifieldqa_zhDataset类"""
    
    @patch('ais_bench.benchmark.datasets.longbench.longbench_multifieldqa_zh.load_dataset')
    @patch('ais_bench.benchmark.datasets.longbench.longbench_multifieldqa_zh.get_data_path')
    def test_load(self, mock_get_path, mock_load_dataset):
        """测试加载数据集"""
        mock_get_path.return_value = '/fake/path'
        
        # 模拟split对象
        mock_test_split = MagicMock()
        type(mock_test_split).__len__ = MagicMock(return_value=1)
        mock_test_split.__getitem__ = MagicMock(side_effect=lambda key: {
            'input': ['question1'],
            'context': ['context1'],
            'answers': [['answer1']]
        }[key])
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = mock_test_split
        mock_dataset.__setitem__ = MagicMock()
        
        mock_load_dataset.return_value = mock_dataset
        
        result = LongBenchmultifieldqa_zhDataset.load(path='/test/path', name='test_name')
        
        self.assertTrue(mock_load_dataset.called)
        self.assertTrue(mock_dataset.__setitem__.called)


class TestLongBenchmusiqueDataset(LongBenchDatasetsTestBase):
    """测试LongBenchmusiqueDataset类"""
    
    @patch('ais_bench.benchmark.datasets.longbench.longbench_musique.load_dataset')
    @patch('ais_bench.benchmark.datasets.longbench.longbench_musique.get_data_path')
    def test_load(self, mock_get_path, mock_load_dataset):
        """测试加载数据集"""
        mock_get_path.return_value = '/fake/path'
        
        # 模拟split对象
        mock_test_split = MagicMock()
        type(mock_test_split).__len__ = MagicMock(return_value=1)
        mock_test_split.__getitem__ = MagicMock(side_effect=lambda key: {
            'input': ['question1'],
            'context': ['context1'],
            'answers': [['answer1']]
        }[key])
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = mock_test_split
        mock_dataset.__setitem__ = MagicMock()
        
        mock_load_dataset.return_value = mock_dataset
        
        result = LongBenchmusiqueDataset.load(path='/test/path', name='test_name')
        
        self.assertTrue(mock_load_dataset.called)
        self.assertTrue(mock_dataset.__setitem__.called)


class TestLongBenchnarrativeqaDataset(LongBenchDatasetsTestBase):
    """测试LongBenchnarrativeqaDataset类"""
    
    @patch('ais_bench.benchmark.datasets.longbench.longbench_narrative_qa.load_dataset')
    @patch('ais_bench.benchmark.datasets.longbench.longbench_narrative_qa.get_data_path')
    def test_load(self, mock_get_path, mock_load_dataset):
        """测试加载数据集"""
        mock_get_path.return_value = '/fake/path'
        
        # 模拟split对象
        mock_test_split = MagicMock()
        type(mock_test_split).__len__ = MagicMock(return_value=1)
        mock_test_split.__getitem__ = MagicMock(side_effect=lambda key: {
            'input': ['question1'],
            'context': ['context1'],
            'answers': [['answer1']]
        }[key])
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = mock_test_split
        mock_dataset.__setitem__ = MagicMock()
        
        mock_load_dataset.return_value = mock_dataset
        
        result = LongBenchnarrativeqaDataset.load(path='/test/path', name='test_name')
        
        self.assertTrue(mock_load_dataset.called)
        self.assertTrue(mock_dataset.__setitem__.called)


class TestLongBenchpassage_countDataset(LongBenchDatasetsTestBase):
    """测试LongBenchpassage_countDataset类"""
    
    @patch('ais_bench.benchmark.datasets.longbench.longbench_passage_count.load_dataset')
    @patch('ais_bench.benchmark.datasets.longbench.longbench_passage_count.get_data_path')
    def test_load(self, mock_get_path, mock_load_dataset):
        """测试加载数据集（无input字段）"""
        mock_get_path.return_value = '/fake/path'
        
        # 模拟split对象
        mock_test_split = MagicMock()
        type(mock_test_split).__len__ = MagicMock(return_value=1)
        mock_test_split.__getitem__ = MagicMock(side_effect=lambda key: {
            'context': ['context1'],
            'answers': [['answer1']]
        }[key])
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = mock_test_split
        mock_dataset.__setitem__ = MagicMock()
        
        mock_load_dataset.return_value = mock_dataset
        
        result = LongBenchpassage_countDataset.load(path='/test/path', name='test_name')
        
        self.assertTrue(mock_load_dataset.called)
        self.assertTrue(mock_dataset.__setitem__.called)


class TestLongBenchpassage_retrieval_enDataset(LongBenchDatasetsTestBase):
    """测试LongBenchpassage_retrieval_enDataset类"""
    
    @patch('ais_bench.benchmark.datasets.longbench.longbench_passage_retrieval_en.load_dataset')
    @patch('ais_bench.benchmark.datasets.longbench.longbench_passage_retrieval_en.get_data_path')
    def test_load(self, mock_get_path, mock_load_dataset):
        """测试加载数据集"""
        mock_get_path.return_value = '/fake/path'
        
        # 模拟split对象
        mock_test_split = MagicMock()
        type(mock_test_split).__len__ = MagicMock(return_value=1)
        mock_test_split.__getitem__ = MagicMock(side_effect=lambda key: {
            'input': ['question1'],
            'context': ['context1'],
            'answers': [['answer1']]
        }[key])
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = mock_test_split
        mock_dataset.__setitem__ = MagicMock()
        
        mock_load_dataset.return_value = mock_dataset
        
        result = LongBenchpassage_retrieval_enDataset.load(path='/test/path', name='test_name')
        
        self.assertTrue(mock_load_dataset.called)
        self.assertTrue(mock_dataset.__setitem__.called)


class TestLongBenchpassage_retrieval_zhDataset(LongBenchDatasetsTestBase):
    """测试LongBenchpassage_retrieval_zhDataset类"""
    
    @patch('ais_bench.benchmark.datasets.longbench.longbench_passage_retrieval_zh.load_dataset')
    @patch('ais_bench.benchmark.datasets.longbench.longbench_passage_retrieval_zh.get_data_path')
    def test_load(self, mock_get_path, mock_load_dataset):
        """测试加载数据集"""
        mock_get_path.return_value = '/fake/path'
        
        # 模拟split对象
        mock_test_split = MagicMock()
        type(mock_test_split).__len__ = MagicMock(return_value=1)
        mock_test_split.__getitem__ = MagicMock(side_effect=lambda key: {
            'input': ['question1'],
            'context': ['context1'],
            'answers': [['answer1']]
        }[key])
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = mock_test_split
        mock_dataset.__setitem__ = MagicMock()
        
        mock_load_dataset.return_value = mock_dataset
        
        result = LongBenchpassage_retrieval_zhDataset.load(path='/test/path', name='test_name')
        
        self.assertTrue(mock_load_dataset.called)
        self.assertTrue(mock_dataset.__setitem__.called)


class TestLongBenchqasperDataset(LongBenchDatasetsTestBase):
    """测试LongBenchqasperDataset类"""
    
    @patch('ais_bench.benchmark.datasets.longbench.longbench_qasper.load_dataset')
    @patch('ais_bench.benchmark.datasets.longbench.longbench_qasper.get_data_path')
    def test_load(self, mock_get_path, mock_load_dataset):
        """测试加载数据集"""
        mock_get_path.return_value = '/fake/path'
        
        # 模拟split对象
        mock_test_split = MagicMock()
        type(mock_test_split).__len__ = MagicMock(return_value=1)
        mock_test_split.__getitem__ = MagicMock(side_effect=lambda key: {
            'input': ['question1'],
            'context': ['context1'],
            'answers': [['answer1']]
        }[key])
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = mock_test_split
        mock_dataset.__setitem__ = MagicMock()
        
        mock_load_dataset.return_value = mock_dataset
        
        result = LongBenchqasperDataset.load(path='/test/path', name='test_name')
        
        self.assertTrue(mock_load_dataset.called)
        self.assertTrue(mock_dataset.__setitem__.called)


class TestLongBenchqmsumDataset(LongBenchDatasetsTestBase):
    """测试LongBenchqmsumDataset类"""
    
    @patch('ais_bench.benchmark.datasets.longbench.longbench_qmsum.load_dataset')
    @patch('ais_bench.benchmark.datasets.longbench.longbench_qmsum.get_data_path')
    def test_load(self, mock_get_path, mock_load_dataset):
        """测试加载数据集"""
        mock_get_path.return_value = '/fake/path'
        
        # 模拟split对象
        mock_test_split = MagicMock()
        type(mock_test_split).__len__ = MagicMock(return_value=1)
        mock_test_split.__getitem__ = MagicMock(side_effect=lambda key: {
            'input': ['question1'],
            'context': ['context1'],
            'answers': [['answer1']]
        }[key])
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = mock_test_split
        mock_dataset.__setitem__ = MagicMock()
        
        mock_load_dataset.return_value = mock_dataset
        
        result = LongBenchqmsumDataset.load(path='/test/path', name='test_name')
        
        self.assertTrue(mock_load_dataset.called)
        self.assertTrue(mock_dataset.__setitem__.called)


class TestLongBenchrepobenchDataset(LongBenchDatasetsTestBase):
    """测试LongBenchrepobenchDataset类"""
    
    @patch('ais_bench.benchmark.datasets.longbench.longbench_repobench.load_dataset')
    @patch('ais_bench.benchmark.datasets.longbench.longbench_repobench.get_data_path')
    def test_load(self, mock_get_path, mock_load_dataset):
        """测试加载数据集"""
        mock_get_path.return_value = '/fake/path'
        
        # 模拟split对象
        mock_test_split = MagicMock()
        type(mock_test_split).__len__ = MagicMock(return_value=1)
        mock_test_split.__getitem__ = MagicMock(side_effect=lambda key: {
            'input': ['question1'],
            'context': ['context1'],
            'answers': [['answer1']]
        }[key])
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = mock_test_split
        mock_dataset.__setitem__ = MagicMock()
        
        mock_load_dataset.return_value = mock_dataset
        
        result = LongBenchrepobenchDataset.load(path='/test/path', name='test_name')
        
        self.assertTrue(mock_load_dataset.called)
        self.assertTrue(mock_dataset.__setitem__.called)


class TestLongBenchsamsumDataset(LongBenchDatasetsTestBase):
    """测试LongBenchsamsumDataset类"""
    
    @patch('ais_bench.benchmark.datasets.longbench.longbench_samsum.load_dataset')
    @patch('ais_bench.benchmark.datasets.longbench.longbench_samsum.get_data_path')
    def test_load(self, mock_get_path, mock_load_dataset):
        """测试加载数据集"""
        mock_get_path.return_value = '/fake/path'
        
        # 模拟split对象
        mock_test_split = MagicMock()
        type(mock_test_split).__len__ = MagicMock(return_value=1)
        mock_test_split.__getitem__ = MagicMock(side_effect=lambda key: {
            'input': ['question1'],
            'context': ['context1'],
            'answers': [['answer1']]
        }[key])
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = mock_test_split
        mock_dataset.__setitem__ = MagicMock()
        
        mock_load_dataset.return_value = mock_dataset
        
        result = LongBenchsamsumDataset.load(path='/test/path', name='test_name')
        
        self.assertTrue(mock_load_dataset.called)
        self.assertTrue(mock_dataset.__setitem__.called)


class TestLongBenchtrecDataset(LongBenchDatasetsTestBase):
    """测试LongBenchtrecDataset类"""
    
    @patch('ais_bench.benchmark.datasets.longbench.longbench_trec.load_dataset')
    @patch('ais_bench.benchmark.datasets.longbench.longbench_trec.get_data_path')
    def test_load(self, mock_get_path, mock_load_dataset):
        """测试加载数据集（包含all_labels）"""
        mock_get_path.return_value = '/fake/path'
        
        # 模拟split对象
        mock_test_split = MagicMock()
        type(mock_test_split).__len__ = MagicMock(return_value=1)
        mock_test_split.__getitem__ = MagicMock(side_effect=lambda key: {
            'input': ['question1'],
            'context': ['context1'],
            'answers': [['answer1']],
            'all_classes': [['class1', 'class2']]
        }[key])
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = mock_test_split
        mock_dataset.__setitem__ = MagicMock()
        
        mock_load_dataset.return_value = mock_dataset
        
        result = LongBenchtrecDataset.load(path='/test/path', name='test_name')
        
        self.assertTrue(mock_load_dataset.called)
        self.assertTrue(mock_dataset.__setitem__.called)


class TestLongBenchtriviaqaDataset(LongBenchDatasetsTestBase):
    """测试LongBenchtriviaqaDataset类"""
    
    @patch('ais_bench.benchmark.datasets.longbench.longbench_trivia_qa.load_dataset')
    @patch('ais_bench.benchmark.datasets.longbench.longbench_trivia_qa.get_data_path')
    def test_load(self, mock_get_path, mock_load_dataset):
        """测试加载数据集"""
        mock_get_path.return_value = '/fake/path'
        
        # 模拟split对象
        mock_test_split = MagicMock()
        type(mock_test_split).__len__ = MagicMock(return_value=1)
        mock_test_split.__getitem__ = MagicMock(side_effect=lambda key: {
            'input': ['question1'],
            'context': ['context1'],
            'answers': [['answer1']]
        }[key])
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = mock_test_split
        mock_dataset.__setitem__ = MagicMock()
        
        mock_load_dataset.return_value = mock_dataset
        
        result = LongBenchtriviaqaDataset.load(path='/test/path', name='test_name')
        
        self.assertTrue(mock_load_dataset.called)
        self.assertTrue(mock_dataset.__setitem__.called)


class TestLongBenchvcsumDataset(LongBenchDatasetsTestBase):
    """测试LongBenchvcsumDataset类"""
    
    @patch('ais_bench.benchmark.datasets.longbench.longbench_vcsum.load_dataset')
    @patch('ais_bench.benchmark.datasets.longbench.longbench_vcsum.get_data_path')
    def test_load(self, mock_get_path, mock_load_dataset):
        """测试加载数据集（无input字段）"""
        mock_get_path.return_value = '/fake/path'
        
        # 模拟split对象
        mock_test_split = MagicMock()
        type(mock_test_split).__len__ = MagicMock(return_value=1)
        mock_test_split.__getitem__ = MagicMock(side_effect=lambda key: {
            'context': ['context1'],
            'answers': [['answer1']]
        }[key])
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = mock_test_split
        mock_dataset.__setitem__ = MagicMock()
        
        mock_load_dataset.return_value = mock_dataset
        
        result = LongBenchvcsumDataset.load(path='/test/path', name='test_name')
        
        self.assertTrue(mock_load_dataset.called)
        self.assertTrue(mock_dataset.__setitem__.called)


if __name__ == '__main__':
    unittest.main()

