import unittest
from ais_bench.benchmark.utils.logging.exceptions import AISBenchImplementationError, AISBenchValueError

from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template_base import BasePromptTemplate


class TestBasePromptTemplate(unittest.TestCase):
    def test_init_type_checks(self):
        """测试BasePromptTemplate初始化时的类型检查"""
        with self.assertRaises(AISBenchValueError):
            BasePromptTemplate(template=123)

        with self.assertRaises(AISBenchValueError):
            BasePromptTemplate(template="hello", ice_token="<ICE>")

    def test_abstract_methods_raise(self):
        """测试BasePromptTemplate的抽象方法未实现时抛出异常"""
        b = BasePromptTemplate(template="{x}")
        with self.assertRaises(AISBenchImplementationError):
            b.generate_ice_item({"x": 1}, label="y")
        with self.assertRaises(AISBenchImplementationError):
            b.generate_label_prompt_item({"x": 1}, ice="", label="y")
        with self.assertRaises(AISBenchImplementationError):
            b.generate_item({"x": 1})


    def test_template_value_type_error(self):
        """测试模板值类型错误时抛出异常"""
        with self.assertRaises(AISBenchValueError):
            BasePromptTemplate(template={"label": 123})

    def test_template_ice_token_not_in_value(self):
        """测试ice_token不在模板值中时抛出异常"""
        with self.assertRaises(AISBenchValueError):
            BasePromptTemplate(template={"label": "hello"}, ice_token="<ICE>")

    def test_check_prompt_template_static_method(self):
        """测试_check_prompt_template静态方法的类型检查"""
        template = BasePromptTemplate(template="test")
        result = BasePromptTemplate._check_prompt_template(template)
        self.assertEqual(result, template)
        
        with self.assertRaises(AISBenchValueError):
            BasePromptTemplate._check_prompt_template("not a template")

    def test_encode_template_with_begin_list(self):
        """测试_encode_template处理begin为列表的情况"""
        template = BasePromptTemplate(template={
            "begin": ["start1", "start2"],
            "round": ["round1"],
            "end": "end"
        })
        result = template._encode_template(template.template, ice=False)
        self.assertIsNotNone(result)

    def test_encode_template_with_begin_string(self):
        """测试_encode_template处理begin为字符串的情况"""
        template = BasePromptTemplate(template={
            "begin": "start",
            "round": ["round1"],
            "end": "end"
        })
        result = template._encode_template(template.template, ice=False)
        self.assertIsNotNone(result)

    def test_encode_template_with_end_list(self):
        """测试_encode_template处理end为列表的情况"""
        template = BasePromptTemplate(template={
            "begin": "start",
            "round": ["round1"],
            "end": ["end1", "end2"]
        })
        result = template._encode_template(template.template, ice=False)
        self.assertIsNotNone(result)

    def test_encode_template_with_end_string(self):
        """测试_encode_template处理end为字符串的情况"""
        template = BasePromptTemplate(template={
            "begin": "start",
            "round": ["round1"],
            "end": "end"
        })
        result = template._encode_template(template.template, ice=False)
        self.assertIsNotNone(result)

    def test_repr(self):
        """测试__repr__方法返回正确的字符串表示"""
        template = BasePromptTemplate(template="test <ICE>", ice_token="<ICE>")
        repr_str = repr(template)
        self.assertIn("BasePromptTemplate", repr_str)
        self.assertIn("test", repr_str)
        self.assertIn("<ICE>", repr_str)


if __name__ == '__main__':
    unittest.main()

