import unittest

from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template_mm import MMPromptTemplate


class TestMMPromptTemplate(unittest.TestCase):
    def test_generate_item_mm(self):
        """测试MMPromptTemplate生成多模态项，将prompt_mm转换为类型化条目列表"""
        template = {
            "round": [
                {"prompt_mm": {"text": "{q}", "image_url": "{img}"}}
            ]
        }
        tp = MMPromptTemplate(template=template)
        out = tp.generate_item({"q": "what?", "img": "http://example/a.jpg", "content": "xxx"})
        self.assertTrue(isinstance(out, list))
        mm_item = None
        for item in out:
            if isinstance(item, dict) and "prompt_mm" in item:
                mm_item = item["prompt_mm"]
                break
        self.assertIsNotNone(mm_item)
        self.assertTrue(isinstance(mm_item, list))


if __name__ == '__main__':
    unittest.main()


