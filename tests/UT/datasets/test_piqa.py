import unittest
from unittest.mock import patch, mock_open

from datasets import Dataset, DatasetDict

from ais_bench.benchmark.datasets.piqa import PIQADataset, PIQADatasetV2, PIQADatasetV3


def _mock_pair(data_content, label_content):
    m_data = mock_open(read_data=data_content)
    m_label = mock_open(read_data=label_content)
    return m_data, m_label


class TestPIQADatasets(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.piqa.get_data_path", return_value="/fake/path")
    @patch("builtins.open")
    def test_piqa(self, mock_open_file, mock_get_path):
        data_line = '{"goal": "do", "sol1": "a", "sol2": "b", "id": 1}\n'
        label_line = '1\n'
        m_data, m_label = _mock_pair(data_line, label_line)
        mock_open_file.side_effect = [
            m_data.return_value,
            m_label.return_value,
            m_data.return_value,
            m_label.return_value,
        ]
        ds = PIQADataset.load("/any")
        self.assertIsInstance(ds, DatasetDict)
        self.assertIn("train", ds)
        self.assertEqual(ds["train"][0]["label"], 1)

    @patch("ais_bench.benchmark.datasets.piqa.get_data_path", return_value="/fake/path")
    @patch("builtins.open")
    def test_piqa_v2(self, mock_open_file, mock_get_path):
        data_line = '{"goal": "Do", "sol1": "A", "sol2": "B", "id": 1}\n'
        label_line = '-1\n'
        m_data, m_label = _mock_pair(data_line, label_line)
        mock_open_file.side_effect = [
            m_data.return_value,
            m_label.return_value,
            m_data.return_value,
            m_label.return_value,
        ]
        ds = PIQADatasetV2.load("/any")
        self.assertEqual(ds["train"][0]["answer"], "NULL")

    @patch("ais_bench.benchmark.datasets.piqa.get_data_path", return_value="/fake/path")
    @patch("builtins.open")
    def test_piqa_v3(self, mock_open_file, mock_get_path):
        data_line = '{"goal": "do?", "sol1": "aaa", "sol2": "bbb", "id": 1}\n'
        label_line = '0\n'
        m_data, m_label = _mock_pair(data_line, label_line)
        mock_open_file.side_effect = [
            m_data.return_value,
            m_label.return_value,
            m_data.return_value,
            m_label.return_value,
        ]
        ds = PIQADatasetV3.load("/any")
        row = ds["train"][0]
        self.assertTrue(row["sol1"].istitle())


if __name__ == '__main__':
    unittest.main()
