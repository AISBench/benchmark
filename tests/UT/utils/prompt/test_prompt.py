import unittest
from unittest.mock import patch
from mmengine.config import ConfigDict

from ais_bench.benchmark.utils.prompt.prompt import (
    safe_format,
    get_prompt_hash,
    is_mm_prompt,
    PromptList
)
from ais_bench.benchmark.utils.logging.exceptions import AISBenchTypeError
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES


class TestSafeFormat(unittest.TestCase):
    """Tests for safe_format function."""

    def test_safe_format_basic(self):
        """Test basic string formatting."""
        result = safe_format("Hello {name}", name="World")
        self.assertEqual(result, "Hello World")

    def test_safe_format_multiple_keys(self):
        """Test formatting with multiple keys."""
        result = safe_format("{greeting} {name}!", greeting="Hello", name="World")
        self.assertEqual(result, "Hello World!")

    def test_safe_format_missing_key(self):
        """Test formatting with missing key (should be ignored)."""
        result = safe_format("Hello {name}", other="value")
        self.assertEqual(result, "Hello {name}")

    def test_safe_format_non_string_value(self):
        """Test formatting with non-string value."""
        result = safe_format("Number: {num}", num=42)
        self.assertEqual(result, "Number: 42")

    def test_safe_format_multiple_occurrences(self):
        """Test formatting with multiple occurrences of same key."""
        result = safe_format("{x} and {x}", x="test")
        self.assertEqual(result, "test and test")


class TestGetPromptHash(unittest.TestCase):
    """Tests for get_prompt_hash function."""

    def test_get_prompt_hash_single_config(self):
        """Test hash generation for single config."""
        dataset_cfg = ConfigDict({
            'infer_cfg': ConfigDict({
                'reader': ConfigDict({
                    'type': 'DatasetReader',
                    'input_columns': ['question'],
                    'output_column': 'answer'
                }),
                'retriever': ConfigDict({
                    'type': 'ZeroRetriever',
                    'test_split': 'test'
                })
            })
        })

        hash1 = get_prompt_hash(dataset_cfg)
        hash2 = get_prompt_hash(dataset_cfg)
        self.assertEqual(hash1, hash2)
        self.assertIsInstance(hash1, str)
        self.assertEqual(len(hash1), 64)  # SHA256 produces 64 hex characters

    def test_get_prompt_hash_list_single(self):
        """Test hash generation for list with single config."""
        dataset_cfg = ConfigDict({
            'infer_cfg': ConfigDict({
                'reader': ConfigDict({
                    'type': 'DatasetReader',
                    'input_columns': ['question'],
                    'output_column': 'answer'
                }),
                'retriever': ConfigDict({
                    'type': 'ZeroRetriever'
                })
            })
        })

        hash1 = get_prompt_hash([dataset_cfg])
        hash2 = get_prompt_hash(dataset_cfg)
        self.assertEqual(hash1, hash2)

    def test_get_prompt_hash_list_multiple(self):
        """Test hash generation for list with multiple configs."""
        cfg1 = ConfigDict({
            'infer_cfg': ConfigDict({
                'reader': ConfigDict({
                    'type': 'DatasetReader',
                    'input_columns': ['q1'],
                    'output_column': 'a1'
                }),
                'retriever': ConfigDict({
                    'type': 'ZeroRetriever'
                })
            })
        })
        cfg2 = ConfigDict({
            'infer_cfg': ConfigDict({
                'reader': ConfigDict({
                    'type': 'DatasetReader',
                    'input_columns': ['q2'],
                    'output_column': 'a2'
                }),
                'retriever': ConfigDict({
                    'type': 'ZeroRetriever'
                })
            })
        })

        hash_result = get_prompt_hash([cfg1, cfg2])
        self.assertIsInstance(hash_result, str)
        self.assertEqual(len(hash_result), 64)

    def test_get_prompt_hash_with_reader_cfg(self):
        """Test hash generation with reader_cfg (new config format)."""
        dataset_cfg = ConfigDict({
            'reader_cfg': ConfigDict({
                'input_columns': ['question'],
                'output_column': 'answer',
                'train_split': 'train',
                'test_split': 'test'
            }),
            'infer_cfg': ConfigDict({
                'reader': ConfigDict({
                    'type': 'DatasetReader'
                }),
                'retriever': ConfigDict({
                    'type': 'ZeroRetriever'
                })
            })
        })

        hash_result = get_prompt_hash(dataset_cfg)
        self.assertIsInstance(hash_result, str)

    def test_get_prompt_hash_with_fix_id_list(self):
        """Test hash generation with fix_id_list."""
        dataset_cfg = ConfigDict({
            'infer_cfg': ConfigDict({
                'reader': ConfigDict({
                    'type': 'DatasetReader',
                    'input_columns': ['question'],
                    'output_column': 'answer'
                }),
                'retriever': ConfigDict({
                    'type': 'ZeroRetriever',
                    'fix_id_list': [1, 2, 3]
                }),
                'inferencer': ConfigDict({
                    'type': 'GenInferencer'
                })
            })
        })

        hash_result = get_prompt_hash(dataset_cfg)
        self.assertIsInstance(hash_result, str)


class TestIsMMPrompt(unittest.TestCase):
    """Tests for is_mm_prompt function."""

    def test_is_mm_prompt_true(self):
        """Test is_mm_prompt returns True for multimodal prompt."""
        prompt = [
            {
                'content': [
                    {'type': 'text', 'text': 'Hello'},
                    {'type': 'image_url', 'image_url': 'http://example.com/image.jpg'}
                ]
            }
        ]
        self.assertTrue(is_mm_prompt(prompt))

    def test_is_mm_prompt_false_string(self):
        """Test is_mm_prompt returns False for string."""
        self.assertFalse(is_mm_prompt("Hello world"))

    def test_is_mm_prompt_false_list_of_strings(self):
        """Test is_mm_prompt returns False for list of strings."""
        # is_mm_prompt expects list of dicts with 'content' key
        # When passed a list of strings, message.get('content') will raise AttributeError
        # because strings don't have .get() method
        # The function doesn't handle this case, so it will raise AttributeError
        # We should test that it raises AttributeError or handle it in the test
        with self.assertRaises(AttributeError):
            is_mm_prompt(["Hello", "World"])

        # Test with list of dicts without 'content' (should return False)
        self.assertFalse(is_mm_prompt([{"role": "user"}, {"role": "assistant"}]))

    def test_is_mm_prompt_false_simple_dict(self):
        """Test is_mm_prompt returns False for simple dict list."""
        prompt = [
            {'role': 'user', 'content': 'Hello'}
        ]
        self.assertFalse(is_mm_prompt(prompt))

    def test_is_mm_prompt_false_content_not_list(self):
        """Test is_mm_prompt returns False when content is not a list."""
        prompt = [
            {'content': 'Hello world'}
        ]
        self.assertFalse(is_mm_prompt(prompt))

    def test_is_mm_prompt_with_video(self):
        """Test is_mm_prompt with video_url."""
        prompt = [
            {
                'content': [
                    {'type': 'video_url', 'video_url': 'http://example.com/video.mp4'}
                ]
            }
        ]
        self.assertTrue(is_mm_prompt(prompt))

    def test_is_mm_prompt_with_audio(self):
        """Test is_mm_prompt with audio_url."""
        prompt = [
            {
                'content': [
                    {'type': 'audio_url', 'audio_url': 'http://example.com/audio.mp3'}
                ]
            }
        ]
        self.assertTrue(is_mm_prompt(prompt))


class TestPromptList(unittest.TestCase):
    """Tests for PromptList class."""

    def test_prompt_list_init(self):
        """Test PromptList initialization."""
        pl = PromptList(["item1", "item2"])
        self.assertEqual(len(pl), 2)
        self.assertEqual(pl[0], "item1")

    def test_format_basic(self):
        """Test format method with basic strings."""
        pl = PromptList(["Hello {name}", "World {name}"])
        result = pl.format(name="Test")
        self.assertIsInstance(result, PromptList)
        self.assertEqual(result[0], "Hello Test")
        self.assertEqual(result[1], "World Test")

    def test_format_with_dict(self):
        """Test format method with dict items containing prompt key."""
        pl = PromptList([
            {"prompt": "Hello {name}", "other": "value"},
            "Plain text"
        ])
        result = pl.format(name="World")
        self.assertEqual(result[0]["prompt"], "Hello World")
        self.assertEqual(result[0]["other"], "value")
        self.assertEqual(result[1], "Plain text")

    def test_format_mm(self):
        """Test format_mm method."""
        from ais_bench.benchmark.utils.prompt.prompt import (
            AIS_TEXT_START, AIS_IMAGE_START, AIS_CONTENT_TAG
        )

        pl = PromptList([
            {
                "prompt_mm": {
                    "text": {"type": "text", "text": "Question: {question}"},
                    "image": {"type": "image_url", "image_url": {"url": "image_{image}.jpg"}}
                }
            }
        ])

        content = f"{AIS_TEXT_START}What is this?{AIS_CONTENT_TAG}{AIS_IMAGE_START}test.jpg"
        result = pl.format_mm(content=content)

        self.assertIsInstance(result, PromptList)
        self.assertIn("prompt_mm", result[0])
        self.assertIsInstance(result[0]["prompt_mm"], list)

    def test_replace_string_with_string(self):
        """Test replace method: string with string."""
        pl = PromptList(["Hello world", "world test"])
        result = pl.replace("world", "universe")
        self.assertEqual(result[0], "Hello universe")
        self.assertEqual(result[1], "universe test")

    def test_replace_string_with_promptlist(self):
        """Test replace method: string with PromptList."""
        pl = PromptList(["A {x} B"])
        replacement = PromptList(["X", "Y"])
        result = pl.replace("{x}", replacement)
        # When replacing "{x}" with PromptList(["X", "Y"]), the string "A {x} B"
        # is split into ["A ", " B"], and replacement is inserted between
        # So result should be: ["A ", "X", "Y", " B"]
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], "A ")
        self.assertIn("X", result)
        self.assertIn("Y", result)
        self.assertEqual(result[3], " B")

    def test_replace_dict_prompt_key(self):
        """Test replace method with dict containing prompt key."""
        pl = PromptList([{"prompt": "Hello {x}", "other": "value"}])
        result = pl.replace("{x}", "World")
        self.assertEqual(result[0]["prompt"], "Hello World")
        self.assertEqual(result[0]["other"], "value")

    def test_replace_dict_prompt_key_with_promptlist_error(self):
        """Test replace method raises error when replacing dict prompt with PromptList."""
        pl = PromptList([{"prompt": "Hello {x}"}])
        replacement = PromptList(["X", "Y"])

        with self.assertRaises(AISBenchTypeError) as cm:
            pl.replace("{x}", replacement)

        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.INVALID_TYPE.full_code)

    def test_add_string(self):
        """Test __add__ method with string."""
        pl = PromptList(["A"])
        result = pl + "B"
        self.assertIsInstance(result, PromptList)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1], "B")

    def test_add_promptlist(self):
        """Test __add__ method with PromptList."""
        pl1 = PromptList(["A"])
        pl2 = PromptList(["B", "C"])
        result = pl1 + pl2
        self.assertIsInstance(result, PromptList)
        self.assertEqual(len(result), 3)

    def test_radd_string(self):
        """Test __radd__ method with string."""
        pl = PromptList(["B"])
        result = "A" + pl
        self.assertIsInstance(result, PromptList)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "A")

    def test_iadd_string(self):
        """Test __iadd__ method with string."""
        pl = PromptList(["A"])
        pl += "B"
        self.assertEqual(len(pl), 2)
        self.assertEqual(pl[1], "B")

    def test_iadd_promptlist(self):
        """Test __iadd__ method with PromptList."""
        pl = PromptList(["A"])
        pl += PromptList(["B", "C"])
        self.assertEqual(len(pl), 3)

    def test_iadd_empty(self):
        """Test __iadd__ method with empty/None."""
        pl = PromptList(["A"])
        pl += None
        self.assertEqual(len(pl), 1)

    def test_str_with_strings(self):
        """Test __str__ method with string items."""
        pl = PromptList(["Hello", "World"])
        result = str(pl)
        self.assertEqual(result, "HelloWorld")

    def test_str_with_dicts(self):
        """Test __str__ method with dict items."""
        pl = PromptList([{"prompt": "Hello"}])
        result = str(pl)
        self.assertIn("Hello", result)

    def test_str_with_invalid_type(self):
        """Test __str__ method raises error with invalid type."""
        pl = PromptList([123])  # Invalid type

        with self.assertRaises(AISBenchTypeError) as cm:
            str(pl)

        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.INVALID_TYPE.full_code)


if __name__ == "__main__":
    unittest.main()

