import unittest

from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template_multiturn import MultiTurnPromptTemplate


class TestMultiTurnPromptTemplate(unittest.TestCase):
    def test_generate_item_multiturn(self):
        """测试MultiTurnPromptTemplate生成多轮对话项，保留begin和end标记"""
        template = {
            "round": [
                {"role": "SYS", "prompt": "<B>"},
                {"role": "USER", "prompt": "Q: {question}"},
                {"role": "BOT", "prompt": "A: {answer}"},
                {"role": "SYS", "prompt": "</B>"},
            ]
        }
        tp = MultiTurnPromptTemplate(template=template)
        out = tp.generate_item({"question": ["q1", "q2"], "answer": ["a1", "a2"]})
        self.assertTrue(isinstance(out, list))
        begin_found = False
        for item in out:
            if isinstance(item, dict) and item.get("prompt") == "<B>":
                begin_found = True
                break
        self.assertTrue(begin_found)
        end_found = False
        for item in reversed(out):
            if isinstance(item, dict) and item.get("prompt") == "</B>":
                end_found = True
                break
        self.assertTrue(end_found)


if __name__ == '__main__':
    unittest.main()


