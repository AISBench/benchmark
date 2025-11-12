import unittest
from unittest import mock
from datasets import Dataset

from ais_bench.benchmark.openicl.icl_retriever.icl_base_retriever import BaseRetriever
from ais_bench.benchmark.utils.logging.exceptions import AISBenchValueError, AISBenchImplementationError
from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template import PromptTemplate


class DummyDataset:
    def __init__(self):
        from datasets import DatasetDict
        reader = type("R", (), {"output_column": "label"})()
        reader.dataset = DatasetDict({
            "test": Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]})
        })
        self.reader = reader
        self.train = Dataset.from_dict({"text": ["t0", "t1"], "label": [0, 1]})
        self.test = Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]})
        self.abbr = "dummy"


class DummyRetriever(BaseRetriever):
    def retrieve(self):
        return [[0], [1]]


class TestBaseRetriever(unittest.TestCase):
    def test_get_gold_ans_with_output_column(self):
        """测试BaseRetriever在output_column存在时获取gold答案"""
        ds = DummyDataset()
        r = DummyRetriever(ds)
        gold = r.get_gold_ans()
        self.assertEqual(gold, [0, 1])

    def test_get_gold_ans_without_output_column(self):
        """测试BaseRetriever在output_column为None时返回None"""
        ds = DummyDataset()
        ds.reader.output_column = None
        r = DummyRetriever(ds)
        gold = r.get_gold_ans()
        self.assertIsNone(gold)

    def test_get_labels_from_prompt_template(self):
        """测试BaseRetriever从prompt_template字典中获取标签"""
        from ais_bench.benchmark.registry import ICL_PROMPT_TEMPLATES
        
        ds = DummyDataset()
        with mock.patch.object(ICL_PROMPT_TEMPLATES, 'build', return_value=PromptTemplate(template={"yes": "Q: {q}", "no": "Q: {q}"})):
            r = DummyRetriever(ds, prompt_template={"type": "PromptTemplate", "template": {"yes": "Q: {q}", "no": "Q: {q}"}})
            labels = r.get_labels()
            self.assertIn("yes", labels)
            self.assertIn("no", labels)

    def test_get_labels_from_ice_template(self):
        """测试BaseRetriever从ice_template字典中获取标签"""
        from ais_bench.benchmark.registry import ICL_PROMPT_TEMPLATES
        
        ds = DummyDataset()
        with mock.patch.object(ICL_PROMPT_TEMPLATES, 'build', return_value=PromptTemplate(template={"yes": "<ICE> Q: {q}", "no": "<ICE> Q: {q}"}, ice_token="<ICE>")):
            r = DummyRetriever(ds, ice_template={"type": "PromptTemplate", "template": {"yes": "<ICE> Q: {q}", "no": "<ICE> Q: {q}"}, "ice_token": "<ICE>"})
            labels = r.get_labels()
            self.assertIn("yes", labels)
            self.assertIn("no", labels)

    def test_get_labels_from_dataset(self):
        """测试BaseRetriever从数据集的output_column获取标签"""
        ds = DummyDataset()
        r = DummyRetriever(ds)
        labels = r.get_labels()
        self.assertEqual(set(labels), {0, 1})

    def test_generate_ice_no_template_error(self):
        """测试BaseRetriever在没有ice_template时生成ICE抛出AISBenchValueError"""
        ds = DummyDataset()
        r = DummyRetriever(ds, ice_template=None)
        with self.assertRaises(AISBenchValueError):
            r.generate_ice([0])

    def test_generate_ice_with_meta_template(self):
        """测试BaseRetriever使用meta模板生成ICE，使用空分隔符"""
        from ais_bench.benchmark.registry import ICL_PROMPT_TEMPLATES
        from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template import PromptTemplate
        
        ds = DummyDataset()
        meta_tmpl = PromptTemplate(template={"round": [{"role": "USER", "prompt": "{text}"}]})
        meta_tmpl.prompt_type = "meta"
        
        with mock.patch.object(ICL_PROMPT_TEMPLATES, 'build', return_value=meta_tmpl):
            r = DummyRetriever(ds, ice_template={"type": "PromptTemplate", "template": {"round": [{"role": "USER", "prompt": "{text}"}]}})
            ice = r.generate_ice([0])
            self.assertIsNotNone(ice)

    def test_generate_ice_with_promptlist(self):
        """测试BaseRetriever生成ICE返回PromptList或字符串"""
        from ais_bench.benchmark.utils.prompt import PromptList
        from ais_bench.benchmark.registry import ICL_PROMPT_TEMPLATES
        
        ds = DummyDataset()
        template = {"round": [{"role": "USER", "prompt": "{text}"}]}
        with mock.patch.object(ICL_PROMPT_TEMPLATES, 'build', return_value=PromptTemplate(template=template)):
            r = DummyRetriever(ds, ice_template={"type": "PromptTemplate", "template": template})
            ice = r.generate_ice([0])
            self.assertIsNotNone(ice)

    def test_generate_label_prompt_all_scenarios(self):
        """测试BaseRetriever的generate_label_prompt方法在不同模板配置下的各种场景"""
        from ais_bench.benchmark.registry import ICL_PROMPT_TEMPLATES
        
        ds = DummyDataset()
        ice_tmpl = PromptTemplate(template="<ICE> Ex: {text}", ice_token="<ICE>")
        prompt_tmpl = PromptTemplate(template="<ICE> Q: {text}", ice_token="<ICE>")
        
        with mock.patch.object(ICL_PROMPT_TEMPLATES, 'build', side_effect=[ice_tmpl, prompt_tmpl]):
            r = DummyRetriever(ds, ice_template={"type": "PromptTemplate", "template": "<ICE> Ex: {text}", "ice_token": "<ICE>"},
                              prompt_template={"type": "PromptTemplate", "template": "<ICE> Q: {text}", "ice_token": "<ICE>"})
            prompt = r.generate_label_prompt(0, "ice_example", label=0)
            self.assertIsInstance(prompt, str)

        with mock.patch.object(ICL_PROMPT_TEMPLATES, 'build', return_value=prompt_tmpl):
            r2 = DummyRetriever(ds, ice_template={"type": "PromptTemplate", "template": "<ICE> Q: {text}", "ice_token": "<ICE>"})
            prompt2 = r2.generate_label_prompt(0, "ice_example", label=0)
            self.assertIsInstance(prompt2, str)

        with mock.patch.object(ICL_PROMPT_TEMPLATES, 'build', return_value=prompt_tmpl):
            r3 = DummyRetriever(ds, prompt_template={"type": "PromptTemplate", "template": "<ICE> Q: {text}", "ice_token": "<ICE>"})
            prompt3 = r3.generate_label_prompt(0, "ice_example", label=0)
            self.assertIsInstance(prompt3, str)

        r4 = DummyRetriever(ds)
        with self.assertRaises(AISBenchImplementationError):
            r4.generate_label_prompt(0, "ice_example", label=0)

    def test_generate_label_prompt_no_ice_token_error(self):
        """测试BaseRetriever在模板没有ice_token时生成标签提示抛出AISBenchImplementationError"""
        from ais_bench.benchmark.registry import ICL_PROMPT_TEMPLATES
        
        ds = DummyDataset()
        tmpl = PromptTemplate(template="Q: {text}")
        with mock.patch.object(ICL_PROMPT_TEMPLATES, 'build', return_value=tmpl):
            r = DummyRetriever(ds, ice_template={"type": "PromptTemplate", "template": "Q: {text}"},
                              prompt_template={"type": "PromptTemplate", "template": "Q: {text}"})
            with self.assertRaises(AISBenchImplementationError):
                r.generate_label_prompt(0, "ice_example", label=0)

    def test_generate_prompt_for_generate_task_all_scenarios(self):
        """测试BaseRetriever的generate_prompt_for_generate_task方法在不同模板配置下的各种场景"""
        from ais_bench.benchmark.registry import ICL_PROMPT_TEMPLATES
        
        ds = DummyDataset()
        ice_tmpl = PromptTemplate(template="<ICE> Ex: {text}", ice_token="<ICE>")
        prompt_tmpl = PromptTemplate(template="<ICE> Q: {text}", ice_token="<ICE>")
        
        with mock.patch.object(ICL_PROMPT_TEMPLATES, 'build', side_effect=[ice_tmpl, prompt_tmpl]):
            r = DummyRetriever(ds, ice_template={"type": "PromptTemplate", "template": "<ICE> Ex: {text}", "ice_token": "<ICE>"},
                              prompt_template={"type": "PromptTemplate", "template": "<ICE> Q: {text}", "ice_token": "<ICE>"})
            prompt = r.generate_prompt_for_generate_task(0, "ice_example")
            self.assertIsNotNone(prompt)

        with mock.patch.object(ICL_PROMPT_TEMPLATES, 'build', return_value=prompt_tmpl):
            r2 = DummyRetriever(ds, ice_template={"type": "PromptTemplate", "template": "<ICE> Q: {text}", "ice_token": "<ICE>"})
            prompt2 = r2.generate_prompt_for_generate_task(0, "ice_example")
            self.assertIsNotNone(prompt2)

        with mock.patch.object(ICL_PROMPT_TEMPLATES, 'build', return_value=prompt_tmpl):
            r3 = DummyRetriever(ds, prompt_template={"type": "PromptTemplate", "template": "<ICE> Q: {text}", "ice_token": "<ICE>"})
            prompt3 = r3.generate_prompt_for_generate_task(0, "ice_example")
            self.assertIsNotNone(prompt3)

        r4 = DummyRetriever(ds)
        with self.assertRaises(AISBenchImplementationError):
            r4.generate_prompt_for_generate_task(0, "ice_example")

    def test_retrieve_abstract(self):
        """测试BaseRetriever的抽象方法retrieve未实现时抛出AISBenchImplementationError"""
        ds = DummyDataset()
        with self.assertRaises(AISBenchImplementationError):
            BaseRetriever(ds).retrieve()


if __name__ == '__main__':
    unittest.main()

