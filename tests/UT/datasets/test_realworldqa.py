import unittest

from ais_bench.benchmark.datasets.realworldqa import (
    OPEN_PROMPT,
    RealworldQAEvaluator,
)


class TestRealworldQAEvaluator(unittest.TestCase):

    def test_extract_answer_with_answer_tag(self):
        evaluator = RealworldQAEvaluator()
        result = evaluator._extract_answer("The car is red.\nANSWER: red")
        self.assertEqual(result, "red")

    def test_extract_answer_with_extra_spaces(self):
        evaluator = RealworldQAEvaluator()
        result = evaluator._extract_answer("Step 1: look\nANSWER:  Blue  ")
        self.assertEqual(result, "Blue")

    def test_extract_answer_no_answer_tag(self):
        evaluator = RealworldQAEvaluator()
        result = evaluator._extract_answer("The car is red.")
        self.assertEqual(result, "")

    def test_extract_answer_multiline(self):
        evaluator = RealworldQAEvaluator()
        result = evaluator._extract_answer("ANSWER: The car\nis red.")
        self.assertEqual(result, "The car")

    def test_extract_answer_empty(self):
        evaluator = RealworldQAEvaluator()
        result = evaluator._extract_answer("")
        self.assertEqual(result, "")

    def test_extract_answer_first_match_only(self):
        evaluator = RealworldQAEvaluator()
        result = evaluator._extract_answer("ANSWER: wrong\n...\nANSWER: correct")
        self.assertEqual(result, "wrong")

    def test_extract_answer_with_think_tags(self):
        evaluator = RealworldQAEvaluator()
        result = evaluator._extract_answer("<think>ANSWER: A, no wait...</think>\nANSWER: B")
        self.assertEqual(result, "A, no wait...</think>")

    def test_extract_answer_single_letter(self):
        evaluator = RealworldQAEvaluator()
        result = evaluator._extract_answer("ANSWER: C")
        self.assertEqual(result, "C")

    def test_normalize_answer_lowercase(self):
        evaluator = RealworldQAEvaluator()
        self.assertEqual(evaluator._normalize_answer("HeLLo"), "hello")

    def test_normalize_answer_strip(self):
        evaluator = RealworldQAEvaluator()
        self.assertEqual(evaluator._normalize_answer("  answer  "), "answer")

    def test_normalize_answer_empty(self):
        evaluator = RealworldQAEvaluator()
        self.assertEqual(evaluator._normalize_answer(""), "")

    def test_normalize_answer_with_newlines(self):
        evaluator = RealworldQAEvaluator()
        self.assertEqual(evaluator._normalize_answer("  Foo\nBar  "), "foo\nbar")

    def test_score_all_correct(self):
        evaluator = RealworldQAEvaluator()
        result = evaluator.score(
            ["ANSWER: A", "ANSWER: B", "ANSWER: C"],
            ["A", "B", "C"],
        )
        self.assertEqual(result["accuracy"], 100.0)
        self.assertEqual(len(result["details"]), 3)
        self.assertTrue(all(d["correct"] for d in result["details"]))

    def test_score_all_wrong(self):
        evaluator = RealworldQAEvaluator()
        result = evaluator.score(["ANSWER: A", "ANSWER: B"], ["C", "D"])
        self.assertEqual(result["accuracy"], 0.0)

    def test_score_mixed(self):
        evaluator = RealworldQAEvaluator()
        result = evaluator.score(
            ["ANSWER: A", "ANSWER: B", "ANSWER: C"],
            ["A", "D", "C"],
        )
        self.assertAlmostEqual(result["accuracy"], 200.0 / 3)
        correct_flags = [d["correct"] for d in result["details"]]
        self.assertEqual(correct_flags, [True, False, True])

    def test_score_case_insensitive(self):
        evaluator = RealworldQAEvaluator()
        result = evaluator.score(["ANSWER: ChildREN"], ["Children"])
        self.assertEqual(result["accuracy"], 100.0)

    def test_score_no_answer_format(self):
        evaluator = RealworldQAEvaluator()
        result = evaluator.score(["B"], ["B"])
        self.assertEqual(result["accuracy"], 0.0)

    def test_score_length_mismatch(self):
        evaluator = RealworldQAEvaluator()
        with self.assertRaises(ValueError):
            evaluator.score(["ANSWER: A", "ANSWER: B"], ["A"])

    def test_score_empty_lists(self):
        evaluator = RealworldQAEvaluator()
        result = evaluator.score([], [])
        self.assertEqual(result["accuracy"], 0)
        self.assertEqual(len(result["details"]), 0)

    def test_score_with_dict_references(self):
        evaluator = RealworldQAEvaluator()
        result = evaluator.score(["ANSWER: correct"], [{"answer": "correct"}])
        self.assertEqual(result["accuracy"], 100.0)

    def test_score_with_dict_references_wrong(self):
        evaluator = RealworldQAEvaluator()
        result = evaluator.score(["ANSWER: wrong"], [{"answer": "correct"}])
        self.assertEqual(result["accuracy"], 0.0)

    def test_score_whitespace_handling(self):
        evaluator = RealworldQAEvaluator()
        result = evaluator.score(["ANSWER:  foo bar  "], ["foo bar"])
        self.assertEqual(result["accuracy"], 100.0)

    def test_score_details_structure(self):
        evaluator = RealworldQAEvaluator()
        result = evaluator.score(["ANSWER: xyz"], ["xyz"])
        detail = result["details"][0]
        for key in ["origin_prediction", "pred", "answer",
                     "normalized_prediction", "normalized_answer", "correct"]:
            self.assertIn(key, detail)

    def test_score_with_malformed_dict_references(self):
        evaluator = RealworldQAEvaluator()
        result = evaluator.score(["ANSWER: x"], [{}])
        detail = result["details"][0]
        self.assertEqual(detail["answer"], "")
        self.assertFalse(detail["correct"])

    def test_open_prompt_format(self):
        formatted = OPEN_PROMPT.format(question="What color is the sky?")
        self.assertIn("What color is the sky?", formatted)
        self.assertIn("ANSWER: [ANSWER]", formatted)
        self.assertIn("step by step", formatted)


if __name__ == "__main__":
    unittest.main()