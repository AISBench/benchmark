import unittest

from mmengine import ConfigDict

from ais_bench.benchmark.cli.workers import AccViz


class TestAccVizDatasetAbbrMapping(unittest.TestCase):
    """judge 数据集改名与 storage 映射的行为"""

    @staticmethod
    def make_cfg():
        return ConfigDict(datasets=[
            ConfigDict(
                abbr='qa',
                judge_infer_cfg=ConfigDict(
                    judge_model=ConfigDict(abbr='judge-x'),
                ),
            ),
            ConfigDict(abbr='plain'),
        ])

    def run_pre_process(self, cfg):
        worker = AccViz.__new__(AccViz)
        worker._cfg_pre_process(cfg)
        return cfg

    def test_judge_dataset_is_renamed_and_mapped(self):
        cfg = self.run_pre_process(self.make_cfg())
        self.assertEqual([d['abbr'] for d in cfg.datasets], ['qa-judge-x', 'plain'])
        self.assertEqual(
            dict(cfg['response_anomaly_dataset_storage_abbrs']),
            {'qa-judge-x': 'qa'},
        )
        self.assertNotIn('judge_infer_cfg', cfg.datasets[0])

    def test_mapping_survives_repeated_pre_processing(self):
        cfg = self.make_cfg()
        self.run_pre_process(cfg)
        first = dict(cfg['response_anomaly_dataset_storage_abbrs'])
        self.run_pre_process(cfg)
        self.assertEqual(dict(cfg['response_anomaly_dataset_storage_abbrs']), first)
        self.assertEqual(cfg.datasets[0]['abbr'], 'qa-judge-x')

    def test_no_judge_dataset_adds_no_key(self):
        cfg = ConfigDict(datasets=[ConfigDict(abbr='plain')])
        self.run_pre_process(cfg)
        self.assertNotIn('response_anomaly_dataset_storage_abbrs', cfg)


if __name__ == '__main__':
    unittest.main()
