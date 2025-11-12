import unittest
from unittest.mock import patch, MagicMock

from ais_bench.benchmark.datasets.vocalsound import VocalSoundDataset


class TestVocalSoundErrors(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.vocalsound.Path")
    @patch("ais_bench.benchmark.datasets.vocalsound.get_data_path", return_value="/fake/dir")
    def test_load_invalid_filename_raises(self, mock_get_path, mock_path_cls):
        # 文件名没有下划线，split('_')[-1] 仍然可用，但可能不符合预期
        # 通过模拟异常来进入except，从而覆盖ValueError路径
        mock_path = MagicMock()
        class CrashPath(str):
            def __fspath__(self):
                return str(self)
        # 通过让打开文件失败不可取，这里让 os.path.splitext 抛异常较复杂
        # 直接让 Path.glob 返回对象，其 __str__ 抛异常更简单
        bad_obj = MagicMock()
        bad_obj.__str__.side_effect = Exception("bad path")
        mock_path.glob.return_value = [bad_obj]
        mock_path_cls.return_value = mock_path
        with self.assertRaises(ValueError):
            VocalSoundDataset.load("/any", audio_type="audio_path")


if __name__ == "__main__":
    unittest.main()
