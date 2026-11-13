import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'ais_bench/benchmark/summarizers'
spec = importlib.util.spec_from_file_location('anomaly_summary', SOURCE / 'response_anomaly.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AnomalySummaryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.logger = SimpleNamespace(
            warning=lambda *args, **kwargs: None,
            info=lambda *args, **kwargs: None,
            debug=lambda *args, **kwargs: None,
        )

    def write(self, relative, records):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            ''.join(json.dumps(record) + '\n' for record in records),
            encoding='utf-8',
        )
        return path

    def write_predictions(self, records, model='model', dataset='data'):
        return self.write(
            f'predictions/{model}/{dataset}.jsonl',
            [{'id': item['id'], 'uuid': item['uuid']} for item in records],
        )

    def write_results(self, records, model='model', dataset='data'):
        return self.write(f'response_anomaly/{model}/{dataset}.jsonl', records)

    @staticmethod
    def table(total=False):
        return [['dataset', 'version', 'metric', 'mode'] +
                (['total_count'] if total else []) + ['model', 'missing']]

    def append(self, table, datasets=(('data', 'data'),)):
        return module.append_anomaly_rows(
            table, self.root, [('model', 'model')], list(datasets), self.logger)

    def test_missing_predictions_are_skipped(self):
        self.write_results([{'id': 1, 'uuid': 'a', 'detection_status': 'failed'}])
        table = self.table()
        self.assertEqual(self.append(table), [table[0]])

    def test_status_counts_rate_and_columns(self):
        records = [
            {'id': 1, 'uuid': 'a', 'detection_status': 'failed'},
            {'id': 1, 'uuid': 'a', 'detection_status': 'completed',
             'anomaly_type_name': 'garbled', 'is_anomaly': True},
            {'id': 2, 'uuid': 'b', 'detection_status': 'completed',
             'anomaly_type_name': 'normal', 'is_anomaly': False},
            {'id': 3, 'uuid': 'c', 'detection_status': 'skipped'},
        ]
        self.write_predictions(records)
        self.write_results(records)
        for total in (False, True):
            table = self.append(self.table(total))
            values = {row[2]: row[-2] for row in table[1:]}
            self.assertEqual(values['response_anomaly_total'], '3')
            self.assertEqual(values['response_anomaly_failed'], '0')
            self.assertEqual(values['response_anomaly_skipped'], '1')
            self.assertEqual(values['response_anomaly_garbled'], '1')
            self.assertEqual(values['response_anomaly_rate'], '50.00')
            self.assertTrue(all(len(row) == len(table[0]) for row in table))
            self.assertTrue(all(row[-1] == '-' for row in table[1:]))

    def test_stale_records_from_earlier_runs_are_ignored(self):
        current = {'id': 1, 'uuid': 'new'}
        self.write_predictions([current])
        self.write_results([
            {'id': 1, 'uuid': 'old', 'detection_status': 'completed',
             'anomaly_type_name': 'garbled', 'is_anomaly': True},
            dict(current, detection_status='completed',
                 anomaly_type_name='normal', is_anomaly=False),
        ])
        table = self.append(self.table())
        values = {row[2]: row[-2] for row in table[1:]}
        self.assertEqual(values['response_anomaly_total'], '1')
        self.assertEqual(values['response_anomaly_garbled'], '0')
        self.assertEqual(values['response_anomaly_rate'], '0.00')

    def test_judge_suffix_uses_inference_artifact(self):
        record = {'id': 1, 'uuid': 'a', 'detection_status': 'completed',
                  'anomaly_type_name': 'repetition', 'is_anomaly': True}
        self.write_predictions([record], dataset='data')
        self.write_results([record], dataset='data')
        table = self.append(self.table(), datasets=(('data', 'data-judge'),))
        self.assertEqual(table[1][0], 'data-judge')
        values = {row[2]: row[-2] for row in table[1:]}
        self.assertEqual(values['response_anomaly_total'], '1')
        self.assertEqual(values['response_anomaly_repetition'], '1')

    def test_invalid_result_file_is_not_reported_as_zero(self):
        self.write_predictions([{'id': 1, 'uuid': 'a'}])
        path = self.write_results([])
        path.write_text('{invalid', encoding='utf-8')
        self.assertIsNone(module.read_anomaly_counts(path, self.logger))
        table = self.table()
        self.assertEqual(self.append(table), [table[0]])

    def test_disabled_does_not_read_existing_results(self):
        self.write_predictions([{'id': 1, 'uuid': 'a'}])
        self.write_results([{'id': 1, 'uuid': 'a', 'detection_status': 'failed'}])
        tree = ast.parse((SOURCE / 'default.py').read_text(encoding='utf-8'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        summarize = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                         and node.name == 'summarize')
        guard = next(node for node in summarize.body if isinstance(node, ast.If)
                     and 'response_anomaly' in ast.unparse(node.test))
        compiled = compile(ast.fix_missing_locations(ast.Module(body=[guard], type_ignores=[])),
                           '<anomaly-guard>', 'exec')
        for cfg in ({}, {'response_anomaly': None},
                    {'response_anomaly': {}}, {'response_anomaly': {'enabled': False}}):
            table = self.table()
            exec(compiled, {'self': SimpleNamespace(cfg=cfg), 'table': table})
            self.assertEqual(table, self.table())

    def test_table_without_mode_column_is_left_untouched(self):
        """自定义 summarizer 的表头没有 mode 列时不应打断汇总"""
        record = {'id': 1, 'uuid': 'a', 'detection_status': 'completed',
                  'anomaly_type_name': 'normal', 'is_anomaly': False}
        self.write_predictions([record])
        self.write_results([record])
        table = [['dataset', 'metric', 'model']]
        self.assertEqual(
            module.append_anomaly_rows(table, self.root, [('model', 'model')],
                                       [('data', 'data')], self.logger),
            [['dataset', 'metric', 'model']],
        )

    def test_build_dataset_pairs_honours_display_filter(self):
        pairs = module.build_dataset_pairs(
            ['a', 'b', 'c'], ['b', 'c'], {'c': 'c-infer'})
        self.assertEqual(pairs, [('b', 'b'), ('c-infer', 'c')])
        self.assertEqual(
            module.build_dataset_pairs(['a'], None, {}),
            [('a', 'a')],
        )

    def test_invalid_utf8_predictions_are_skipped(self):
        """非法 UTF-8 字节不应中断汇总，只跳过该组"""
        record = {'id': 1, 'uuid': 'a', 'detection_status': 'completed',
                  'anomaly_type_name': 'normal', 'is_anomaly': False}
        self.write_results([record])
        path = self.write_predictions([])
        path.write_bytes(b'{"id":1,"uuid":"a"}\n\xff\xfe\x00broken\n')
        self.assertIsNone(module.load_prediction_keys(path, self.logger))
        table = self.table()
        self.assertEqual(self.append(table), [table[0]])

    def test_invalid_utf8_results_are_skipped(self):
        record = {'id': 1, 'uuid': 'a', 'detection_status': 'completed',
                  'anomaly_type_name': 'normal', 'is_anomaly': False}
        self.write_predictions([record])
        path = self.write_results([])
        path.write_bytes(b'{"id":1,"uuid":"a"}\n\xc3\x28\n')
        self.assertIsNone(module.read_anomaly_counts(path, self.logger))
        table = self.table()
        self.assertEqual(self.append(table), [table[0]])

    def test_all_three_outputs_include_metrics(self):
        record = {'id': 1, 'uuid': 'a', 'detection_status': 'unavailable'}
        self.write_predictions([record])
        self.write_results([record])
        table = self.append(self.table())
        tree = ast.parse((SOURCE / 'default.py').read_text(encoding='utf-8'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                   and node.name in ('_format_md_table', '_output_to_file')]
        namespace = {
            'osp': __import__('os.path', fromlist=['path']),
            'mmengine': SimpleNamespace(
                mkdir_or_exist=lambda path: Path(path).mkdir(parents=True, exist_ok=True)),
            'tabulate': SimpleNamespace(tabulate=lambda table, **kwargs: str(table)),
        }
        harness = ast.Module(
            body=[ast.ClassDef(name='Harness', bases=[], keywords=[],
                               body=methods, decorator_list=[])],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(harness), '<summary-output>', 'exec'), namespace)
        writer = namespace['Harness']()
        writer.work_dir = str(self.root)
        writer.logger = self.logger
        writer._output_to_file(None, 'test', table, '')
        for extension in ('txt', 'csv', 'md'):
            text = (self.root / 'summary' / f'summary_test.{extension}').read_text(encoding='utf-8')
            self.assertIn('response_anomaly_unavailable', text)
            self.assertIn('response_anomaly_total', text)


if __name__ == '__main__':
    unittest.main()
