
import unittest
from ais_bench.benchmark.utils.prompt.prompt import PromptList
from ais_bench.benchmark.utils.logging.exceptions import AISBenchTypeError

class TestPromptList(unittest.TestCase):
    def test_replace_raises_aisbench_type_error(self):
        pl = PromptList([{'prompt': 'hello world'}])
        dst = PromptList(['foo', 'bar'])
        with self.assertRaises(AISBenchTypeError):
            pl.replace('world', dst)

    def test_str_raises_aisbench_type_error(self):
        pl = PromptList([123]) # Invalid type
        with self.assertRaises(AISBenchTypeError):
            str(pl)

if __name__ == '__main__':
    unittest.main()
