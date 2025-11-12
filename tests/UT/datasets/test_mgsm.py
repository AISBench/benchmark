import unittest
from unittest.mock import patch, mock_open

from ais_bench.benchmark.datasets.mgsm import (
    MGSMSDataset,
    mgsm_postprocess,
    MGSM_Evaluator,
)


class TestMGSM(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.mgsm.get_data_path", return_value="/fake/path.tsv")
    @patch("builtins.open")
    def test_dataset(self, mock_open_file, mock_get_path):
        content = "1+1\t2\n"
        m = mock_open(read_data=content)
        mock_open_file.return_value = m.return_value
        ds = MGSMSDataset.load("/any")
        self.assertEqual(ds[0]['answer'], '2')

    def test_postprocess(self):
        self.assertEqual(mgsm_postprocess("Answer: 1,234", 'en'), '1234')
        self.assertEqual(mgsm_postprocess("no prefix here", 'en'), '')

    def test_evaluator(self):
        eva = MGSM_Evaluator()
        out = eva.score(["5"], ["5"])
        self.assertIn('accuracy', out)


if __name__ == '__main__':
    unittest.main()
