import unittest
from unittest import mock

from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.utils.prompt import PromptList


class TestPromptTemplate(unittest.TestCase):
    def test_generate_ice_item_str_template(self):
        """测试使用字符串模板生成ICE项"""
        tp = PromptTemplate(template="Q: {q} A: {a}", ice_token=None)
        out = tp.generate_ice_item({"q": "1+1?", "a": "2"}, label="any")
        self.assertEqual(out, "Q: 1+1? A: 2")

    def test_generate_ice_item_dict_template(self):
        """测试使用字典模板生成ICE项"""
        tp = PromptTemplate(template={"yes": "Q: {q} A: {a}"}, ice_token=None)
        out = tp.generate_ice_item({"q": "1+1?", "a": "2"}, label="yes")
        self.assertEqual(out, "Q: 1+1? A: 2")

    def test_generate_ice_item_with_sep_token(self):
        """测试使用sep_token生成ICE项时移除分隔符"""
        tp = PromptTemplate(template="Q: {q} A: {a}", ice_token=None, sep_token="<SEP>")
        out = tp.generate_ice_item({"q": "1+1?", "a": "2"}, label="any")
        self.assertNotIn("<SEP>", str(out))

    def test_generate_ice_item_promptlist_format(self):
        """测试使用PromptList格式生成ICE项"""
        template = {"round": [{"role": "USER", "prompt": "{q}"}]}
        tp = PromptTemplate(template=template, ice_token=None)
        out = tp.generate_ice_item({"q": "test"}, label="any")
        self.assertTrue(isinstance(out, PromptList) or isinstance(out, list))

    def test_generate_label_prompt_item_str_template(self):
        """测试使用字符串模板生成标签提示项"""
        tp = PromptTemplate(template="<ICE> Q: {q}", ice_token="<ICE>")
        ice = "example"
        out = tp.generate_label_prompt_item({"q": "1+1?"}, ice, label="any")
        self.assertIn("example", out)
        self.assertIn("1+1?", out)

    def test_generate_label_prompt_item_dict_template(self):
        """测试使用字典模板生成标签提示项"""
        tp = PromptTemplate(template={"yes": "A: {a}"}, ice_token=None)
        out = tp.generate_label_prompt_item({"a": "2"}, ice="", label="yes")
        self.assertIn("2", out)

    def test_generate_label_prompt_item_remain_sep(self):
        """测试generate_label_prompt_item在remain_sep=True时保留分隔符"""
        tp = PromptTemplate(template="Q: {q}<SEP>A: {a}", ice_token=None, sep_token="<SEP>")
        out = tp.generate_label_prompt_item({"q": "1+1?", "a": "2"}, ice="", label="any", remain_sep=True)
        self.assertIn("<SEP>", out)

    def test_generate_item_dict_template_origin_type(self):
        """测试使用字典模板生成项时保持原始类型"""
        tp = PromptTemplate(template={"label1": "Q: {q} A: {a}"}, ice_token=None)
        out = tp.generate_item({"q": "1+1?", "a": "2"}, output_field="a")
        self.assertTrue(isinstance(out, (str, list, PromptList)))

    def test_generate_item_with_output_field(self):
        """测试使用output_field生成项时替换为指定token"""
        tp = PromptTemplate(template="Q: {q} A: {a}", ice_token=None)
        out = tp.generate_item({"q": "1+1?", "a": "2"}, output_field="a", output_field_replace_token="<ANS>")
        self.assertIn("<ANS>", str(out))

    def test_generate_item_with_tokens(self):
        """测试使用多个token生成项"""
        template = {
            "round": [
                {"role": "SYS", "prompt": "{sys}"},
                {"role": "USER", "prompt": "{q}"},
                {"role": "BOT", "prompt": "{a}"},
            ]
        }
        tp = PromptTemplate(template=template, ice_token="<ICE>", sep_token="<SEP>")
        out = tp.generate_item(
            entry={"sys": "s", "q": "1+1?", "a": "2"},
            output_field="a",
            output_field_replace_token="<ANS>",
            ice_field_replace_token="<ICE>DATA",
        )
        self.assertTrue(isinstance(out, list))
        self.assertIn("<ANS>", str(out))


if __name__ == '__main__':
    unittest.main()


