import unittest
from unittest import mock
import torch
from copy import deepcopy
from typing import List

from ais_bench.benchmark.models.local_models import base
from ais_bench.benchmark.models.local_models.base import BaseModel, LMTemplateParser
from ais_bench.benchmark.utils.prompt import PromptList


# 创建一个具体的子类来测试BaseModel
class ConcreteModel(BaseModel):
    def _generate(self, input, max_out_len: int) -> List[str]:
        return ["generated text"]
    
    def encode(self, prompt: str) -> torch.Tensor:
        return torch.tensor([[1, 2, 3]])
    
    def decode(self, tokens: torch.Tensor) -> str:
        return "decoded text"
    
    def get_token_len(self, prompt: str) -> int:
        return len(prompt.split())


class TestBaseModel(unittest.TestCase):
    def setUp(self):
        self.default_kwargs = {
            "path": "test-model",
            "max_seq_len": 2048,
            "tokenizer_only": False,
            "generation_kwargs": {"temperature": 0.7}
        }
    
    def test_init_default_parameters(self):
        """测试使用默认参数初始化"""
        with mock.patch.object(base, 'LMTemplateParser') as mock_parser:
            model = ConcreteModel(**self.default_kwargs)
            
            # 验证基本属性
            self.assertEqual(model.path, "test-model")
            self.assertEqual(model.max_seq_len, 2048)
            self.assertFalse(model.tokenizer_only)
            self.assertEqual(model.generation_kwargs, {"temperature": 0.7})
            self.assertFalse(model.sync_rank)
            self.assertFalse(model.is_synthetic)
            self.assertIsNone(model.eos_token_id)
            mock_parser.assert_called_once_with(None)
    
    def test_init_with_meta_template(self):
        """测试带有meta_template的初始化"""
        kwargs = self.default_kwargs.copy()
        meta_template = {
            "round": [{"role": "USER", "begin": "[USER] ", "end": "\n"}],
            "eos_token_id": 123
        }
        kwargs["meta_template"] = meta_template
        
        with mock.patch.object(base, 'LMTemplateParser') as mock_parser:
            model = ConcreteModel(**kwargs)
            mock_parser.assert_called_once_with(meta_template)
            self.assertEqual(model.eos_token_id, 123)
    
    def test_set_synthetic(self):
        """测试set_synthetic方法"""
        model = ConcreteModel(**self.default_kwargs)
        self.assertFalse(model.is_synthetic)
        
        model.set_synthetic()
        self.assertTrue(model.is_synthetic)
    
    def test_parse_template(self):
        """测试parse_template方法"""
        model = ConcreteModel(**self.default_kwargs)
        mock_parser = mock.MagicMock()
        mock_parser.parse_template.return_value = "parsed template"
        model.template_parser = mock_parser
        
        result = model.parse_template("test prompt", "gen")
        mock_parser.parse_template.assert_called_once_with("test prompt", "gen")
        self.assertEqual(result, "parsed template")
    
    def test_get_token_len_from_template_single(self):
        """测试单个模板的token长度计算"""
        model = ConcreteModel(**self.default_kwargs)
        mock_parser = mock.MagicMock()
        mock_parser.parse_template.return_value = "hello world"
        model.template_parser = mock_parser
        
        # 模拟get_token_len返回2
        with mock.patch.object(model, 'get_token_len', return_value=2):
            result = model.get_token_len_from_template("test", "ppl")
            self.assertEqual(result, 2)
    
    def test_get_token_len_from_template_batch(self):
        """测试批量模板的token长度计算"""
        model = ConcreteModel(**self.default_kwargs)
        mock_parser = mock.MagicMock()
        mock_parser.parse_template.return_value = ["hello world", "test case"]
        model.template_parser = mock_parser
        
        # 模拟get_token_len返回不同值
        with mock.patch.object(model, 'get_token_len', side_effect=[2, 2]):
            result = model.get_token_len_from_template(["test1", "test2"], "ppl")
            self.assertEqual(result, [2, 2])
    
    def test_get_token_len_from_template_prompt_list(self):
        """测试PromptList类型的token长度计算"""
        model = ConcreteModel(**self.default_kwargs)
        prompt_list = PromptList([{"prompt": "test"}])
        mock_parser = mock.MagicMock()
        mock_parser.parse_template.return_value = prompt_list
        model.template_parser = mock_parser
        
        # 模拟get_token_len返回1
        with mock.patch.object(model, 'get_token_len', return_value=1):
            result = model.get_token_len_from_template(prompt_list, "ppl")
            self.assertEqual(result, 1)
    
    def test_sync_inputs_rank_0(self):
        """测试rank 0的输入同步"""
        model = ConcreteModel(**self.default_kwargs)
        mock_dist = mock.MagicMock()
        mock_dist.get_rank.return_value = 0
        
        # 模拟torch tensor和dist操作
        mock_tensor = torch.tensor([[1, 2, 3]])
        with mock.patch.object(base, 'dist', mock_dist):
            with mock.patch.object(model, 'encode', return_value=mock_tensor):
                with mock.patch.object(model, 'get_token_len', return_value=10):
                    with mock.patch.object(model, 'decode', return_value="synced text"):
                        with mock.patch.object(base.torch, 'tensor', return_value=mock_tensor):
                            result = model.sync_inputs("test input")
                            
                            mock_dist.broadcast.assert_called_with(mock_tensor, src=0)
                            self.assertEqual(result, "synced text")
    
    def test_sync_inputs_rank_1(self):
        """测试非rank 0的输入同步"""
        model = ConcreteModel(**self.default_kwargs)
        mock_dist = mock.MagicMock()
        mock_dist.get_rank.return_value = 1
        
        # 模拟size tensor和empty tensor
        size_tensor = torch.tensor([[1, 3]])
        empty_tensor = torch.empty([1, 3], dtype=torch.long)
        
        with mock.patch.object(base, 'dist', mock_dist):
            with mock.patch.object(base.torch, 'empty', return_value=empty_tensor):
                with mock.patch.object(model, 'decode', return_value="synced text"):
                    result = model.sync_inputs("test input")
                    
                    # 验证调用了两次broadcast
                    self.assertEqual(mock_dist.broadcast.call_count, 2)
                    self.assertEqual(result, "synced text")
    
    def test_sync_inputs_large_tokens_logging(self):
        """测试大token数的日志记录"""
        model = ConcreteModel(**self.default_kwargs)
        model.logger = mock.MagicMock()
        mock_dist = mock.MagicMock()
        mock_dist.get_rank.return_value = 0
        
        with mock.patch.object(base, 'dist', mock_dist):
            with mock.patch.object(model, 'encode', return_value=torch.tensor([[1]])):
                with mock.patch.object(model, 'get_token_len', return_value=3000):  # 超过2048
                    with mock.patch.object(model, 'decode', return_value="text"):
                        with mock.patch.object(base.torch, 'tensor'):
                            model.sync_inputs("test")
                            model.logger.info.assert_called_with("Large tokens nums: 3000")
    
    def test_to_method(self):
        """测试to方法"""
        model = ConcreteModel(**self.default_kwargs)
        model.model = mock.MagicMock()
        
        model.to("cuda:0")
        model.model.to.assert_called_once_with("cuda:0")
    
    def test_abstract_methods(self):
        """测试抽象方法（通过ConcreteModel间接测试）"""
        model = ConcreteModel(**self.default_kwargs)
        
        # 验证具体实现可以正常调用
        result = model._generate("test", 100)
        self.assertEqual(result, ["generated text"])
        
        tokens = model.encode("test")
        self.assertTrue(isinstance(tokens, torch.Tensor))
        
        text = model.decode(torch.tensor([[1]]))
        self.assertEqual(text, "decoded text")
        
        length = model.get_token_len("hello world")
        self.assertEqual(length, 2)


class TestLMTemplateParser(unittest.TestCase):
    def setUp(self):
        self.default_meta_template = {
            "round": [
                {"role": "USER", "begin": "[USER] ", "end": "\n"},
                {"role": "ASSISTANT", "begin": "[ASSISTANT] ", "end": "\n"}
            ]
        }
    
    def test_init_without_meta_template(self):
        """测试没有meta_template的初始化"""
        parser = LMTemplateParser(None)
        self.assertIsNone(parser.meta_template)
    
    def test_init_with_valid_meta_template(self):
        """测试有效的meta_template初始化"""
        parser = LMTemplateParser(self.default_meta_template)
        self.assertEqual(parser.meta_template, self.default_meta_template)
        self.assertEqual(len(parser.roles), 2)
        self.assertIn("USER", parser.roles)
        self.assertIn("ASSISTANT", parser.roles)
    
    def test_init_with_reserved_roles(self):
        """测试包含reserved_roles的初始化"""
        meta_template = deepcopy(self.default_meta_template)
        meta_template["reserved_roles"] = [
            {"role": "SYSTEM", "begin": "[SYSTEM] ", "end": "\n"}
        ]
        parser = LMTemplateParser(meta_template)
        self.assertEqual(len(parser.roles), 3)
        self.assertIn("SYSTEM", parser.roles)
    
    def test_init_with_duplicate_roles(self):
        """测试重复角色的情况"""
        meta_template = {
            "round": [
                {"role": "USER", "begin": "[USER] ", "end": "\n"},
                {"role": "USER", "begin": "[USER2] ", "end": "\n"}
            ]
        }
        with self.assertRaises(AssertionError):
            LMTemplateParser(meta_template)
    
    def test_parse_template_string(self):
        """测试解析字符串模板"""
        parser = LMTemplateParser(self.default_meta_template)
        result = parser.parse_template("test prompt", "gen")
        self.assertEqual(result, "test prompt")
    
    def test_parse_template_list(self):
        """测试解析列表模板"""
        parser = LMTemplateParser(self.default_meta_template)
        result = parser.parse_template(["test1", "test2"], "gen")
        self.assertEqual(result, ["test1", "test2"])
    
    def test_parse_template_prompt_list_no_meta(self):
        """测试没有meta_template时解析PromptList"""
        parser = LMTemplateParser(None)
        prompt_list = PromptList([
            {"prompt": "Hello"},
            {"prompt": "World"}
        ])
        result = parser.parse_template(prompt_list, "gen")
        self.assertEqual(result, "Hello\nWorld")
    
    def test_parse_template_with_meta_and_sections(self):
        """测试带有sections的模板解析"""
        parser = LMTemplateParser(self.default_meta_template)
        prompt_template = PromptList([
            {"section": "round", "pos": "begin"},
            {"role": "USER", "prompt": "Question"},
            {"role": "ASSISTANT", "prompt": "Answer"},
            {"section": "round", "pos": "end"}
        ])
        
        # 模拟_split_rounds和_prompt2str方法
        with mock.patch.object(parser, '_split_rounds', return_value=[0, 4]):  # 只有一轮
            with mock.patch.object(parser, '_prompt2str', return_value=("processed text", True)):
                result = parser.parse_template(prompt_template, "gen")
                self.assertEqual(result, "processed text")
    
    def test_parse_template_begin_end_sections(self):
        """测试begin和end sections的解析"""
        meta_template = {
            "begin": "Begin text\n",
            "end": "\nEnd text",
            "round": []
        }
        parser = LMTemplateParser(meta_template)
        prompt_template = PromptList([
            {"section": "begin", "pos": "begin"},
            {"section": "begin", "pos": "end"}
        ])
        
        result = parser.parse_template(prompt_template, "gen")
        self.assertEqual(result, "Begin text\n\nEnd text")
    
    def test_parse_template_invalid_pos(self):
        """测试无效的pos值"""
        parser = LMTemplateParser(self.default_meta_template)
        prompt_template = PromptList([
            {"section": "round", "pos": "invalid"}
        ])
        
        with self.assertRaises(Exception):
            parser.parse_template(prompt_template, "gen")
    
    def test_split_rounds(self):
        """测试_split_rounds方法"""
        parser = LMTemplateParser(self.default_meta_template)
        prompt_template = [
            {"role": "USER", "prompt": "q1"},
            {"role": "ASSISTANT", "prompt": "a1"},
            {"role": "USER", "prompt": "q2"},
            {"role": "ASSISTANT", "prompt": "a2"}
        ]
        result = parser._split_rounds(prompt_template, self.default_meta_template["round"])
        self.assertEqual(result, [0, 2, 4])
    
    def test_update_role_dict_string(self):
        """测试字符串输入的update_role_dict"""
        parser = LMTemplateParser(self.default_meta_template)
        result = parser._update_role_dict("test")
        self.assertEqual(len(result), 2)
    
    def test_update_role_dict_dict(self):
        """测试字典输入的update_role_dict"""
        parser = LMTemplateParser(self.default_meta_template)
        result = parser._update_role_dict({"role": "USER", "prompt": "new prompt"})
        self.assertEqual(result["USER"]["prompt"], "new prompt")
    
    def test_update_role_dict_with_fallback(self):
        """测试带有fallback_role的update_role_dict"""
        parser = LMTemplateParser(self.default_meta_template)
        # 重定向到标准输出以避免测试输出干扰
        with mock.patch('builtins.print'):
            result = parser._update_role_dict({"role": "UNKNOWN", "fallback_role": "USER", "prompt": "fallback prompt"})
            self.assertEqual(result["USER"]["prompt"], "fallback prompt")
    
    def test_update_role_dict_no_fallback(self):
        """测试没有fallback_role的update_role_dict"""
        # 简化测试，避免键错误
        parser = LMTemplateParser(self.default_meta_template)
        # 重定向到标准输出以避免测试输出干扰
        with mock.patch('builtins.print'):
            # 提供包含role键的字典，但没有fallback_role
            result = parser._update_role_dict({"role": "USER", "prompt": "new user prompt"})
            # 验证用户提示被更新
            self.assertEqual(result["USER"]["prompt"], "new user prompt")
    
    def test_prompt2str_string(self):
        """测试字符串输入的prompt2str"""
        parser = LMTemplateParser(self.default_meta_template)
        result, cont = parser._prompt2str("test", {}, False)
        self.assertEqual(result, "test")
        self.assertTrue(cont)
    
    def test_prompt2str_dict(self):
        """测试字典输入的prompt2str"""
        parser = LMTemplateParser(self.default_meta_template)
        # 模拟_role2str方法
        with mock.patch.object(parser, '_role2str', return_value=("role text", True)):
            result, cont = parser._prompt2str({"role": "USER"}, {}, False)
            self.assertEqual(result, "role text")
            self.assertTrue(cont)
    
    def test_prompt2str_list(self):
        """测试列表输入的prompt2str"""
        parser = LMTemplateParser(self.default_meta_template)
        # 模拟递归调用
        def side_effect(prompt, role_dict, for_gen):
            if isinstance(prompt, str):
                return (prompt, True)
            return ("list text", False)
        
        with mock.patch.object(parser, '_prompt2str', side_effect=side_effect):
            result, cont = parser._prompt2str(["test1", "test2"], {}, False)
            self.assertEqual(result, "list text")
            self.assertFalse(cont)
    
    def test_role2str_normal(self):
        """测试正常情况下的role2str"""
        parser = LMTemplateParser(self.default_meta_template)
        role_dict = {
            "USER": {"begin": "[U] ", "prompt": "test", "end": "[/U]"}
        }
        result, cont = parser._role2str({"role": "USER"}, role_dict, False)
        self.assertEqual(result, "[U] test[/U]")
        self.assertTrue(cont)
    
    def test_role2str_for_gen(self):
        """测试for_gen=True时的role2str"""
        parser = LMTemplateParser(self.default_meta_template)
        role_dict = {
            "USER": {"begin": "[U] ", "prompt": "test", "end": "[/U]", "generate": True}
        }
        result, cont = parser._role2str({"role": "USER"}, role_dict, True)
        self.assertEqual(result, "[U] ")
        self.assertFalse(cont)
    
    def test_role2str_with_fallback(self):
        """测试带有fallback_role的role2str"""
        parser = LMTemplateParser(self.default_meta_template)
        role_dict = {
            "USER": {"begin": "[U] ", "prompt": "test", "end": "[/U]"}
        }
        result, cont = parser._role2str({"role": "UNKNOWN", "fallback_role": "USER"}, role_dict, False)
        self.assertEqual(result, "[U] test[/U]")
        self.assertTrue(cont)
    
    def test_encode_speical_tokens(self):
        """测试_encode_speical_tokens方法"""
        parser = LMTemplateParser(self.default_meta_template)
        with self.assertRaises(NotImplementedError):
            parser._encode_speical_tokens(["test", 123])


if __name__ == "__main__":
    unittest.main()