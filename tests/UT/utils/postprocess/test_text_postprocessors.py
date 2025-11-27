import sys
import types
import unittest

from ais_bench.benchmark.utils.postprocess.text_postprocessors import (
    general_postprocess,
    general_cn_postprocess,
    first_capital_postprocess,
    last_capital_postprocess,
    first_option_postprocess,
    first_option_postprocess_v1,
    first_capital_postprocess_multi,
    last_option_postprocess,
    first_number_postprocess,
    multiple_select_postprocess,
    xml_tag_postprocessor,
    general_eval_wrapper_postprocess,
    match_answer_pattern,
)


class TestTextPostprocessors(unittest.TestCase):
    def test_general_postprocess_basic(self):
        text = "The quick. brown fox\n jumps over"
        # removes punctuation and articles, trims spaces, takes before first delimiter
        self.assertEqual(general_postprocess(text), "quick")

    def test_general_cn_postprocess_with_mocked_jieba(self):
        # Create a dummy jieba module with a simple cut implementation
        dummy_jieba = types.ModuleType("jieba")

        def _cut(s):
            return list(s)  # split into characters
        dummy_jieba.cut = _cut

        # Inject into sys.modules so the function-level import finds it
        prev = sys.modules.get("jieba")
        sys.modules["jieba"] = dummy_jieba
        try:
            text = "你好世界"
            # result joins characters with spaces
            self.assertEqual(general_cn_postprocess(text), "你 好 世 界")
        finally:
            if prev is not None:
                sys.modules["jieba"] = prev
            else:
                del sys.modules["jieba"]

    def test_first_capital_postprocess(self):
        self.assertEqual(first_capital_postprocess("abCDeF"), "C")
        self.assertEqual(first_capital_postprocess("abc"), "")

    def test_last_capital_postprocess(self):
        self.assertEqual(last_capital_postprocess("abCDeF"), "F")
        self.assertEqual(last_capital_postprocess("abc"), "")

    def test_first_option_postprocess_v1_and_base(self):
        text = "很多描述... 答案是： C。更多描述"
        self.assertEqual(first_option_postprocess(text, "ABCD"), "C")
        # v1 checks only the tail window, but still should find the same
        self.assertEqual(first_option_postprocess_v1(text, "ABCD"), "C")

        # English variants
        text2 = "The answer is: (B)"
        self.assertEqual(first_option_postprocess(text2, "ABCD"), "B")

        # Cushion patterns
        text3 = "Option: D: because ..."
        self.assertEqual(first_option_postprocess(
            text3, "ABCD", cushion=True), "D")

        # No match
        self.assertEqual(first_option_postprocess(
            "no answer here", "ABCD"), "")

    def test_first_capital_postprocess_multi(self):
        self.assertEqual(first_capital_postprocess_multi("xxABCDyy"), "ABCD")
        self.assertEqual(first_capital_postprocess_multi("none"), "")

    def test_last_option_postprocess(self):
        text = "... choose A, or maybe C. Final: B? No, C."
        self.assertEqual(last_option_postprocess(text, "ABCD"), "C")
        self.assertEqual(last_option_postprocess("no match", "ABCD"), "")

    def test_first_number_postprocess(self):
        self.assertEqual(first_number_postprocess(
            "pi is 3.14, e is 2.71"), 3.14)
        self.assertIsNone(first_number_postprocess("no numbers"))

    def test_multiple_select_postprocess(self):
        text = "select A and C and A again, plus b"
        # uppercase unique sorted
        self.assertEqual(multiple_select_postprocess(text), "AC")

    def test_xml_tag_postprocessor(self):
        text = "<conclude>First</conclude> something <conclude>Second</conclude>"
        self.assertEqual(xml_tag_postprocessor(text, "<conclude>"), "Second")
        self.assertEqual(xml_tag_postprocessor(
            "no tags", "<conclude>"), "NO ANSWER FOUND")

    def test_general_eval_wrapper_postprocess(self):
        # literal_eval success on quoted string
        text_repr = "'Hello, world.'"
        out = general_eval_wrapper_postprocess(
            text_repr, postprocess="general")
        self.assertEqual(out, "Hello")

        # literal_eval failure -> uses raw text, still applies postprocess
        raw = "not a repr"
        out2 = general_eval_wrapper_postprocess(raw, postprocess="general")
        self.assertEqual(out2, general_postprocess(raw))

        # No postprocess -> returns evaluated value (or raw on failure)
        self.assertEqual(general_eval_wrapper_postprocess(
            text_repr), "Hello, world.")
        self.assertEqual(general_eval_wrapper_postprocess(raw), raw)

    def test_match_answer_pattern(self):
        text = "Answer: [C]"
        pattern = r"\[(.)\]"
        self.assertEqual(match_answer_pattern(text, pattern), "C")
        self.assertEqual(match_answer_pattern("no", pattern), "")


if __name__ == "__main__":
    unittest.main()
