import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


class FakeOutput:
    success = True
    uuid = 'case-1'
    input_tokens = 3
    output_tokens = 2
    reasoning_content = ''

    def __init__(self, logprobs, payload=None):
        self.origin_logprobs = logprobs
        self.extra_details_data = ({'response_anomaly_payload': payload}
                                   if payload is not None else {})

    def get_prediction(self):
        return 'answer'


def load_prediction_method():
    root = Path(__file__).resolve().parents[5]
    path = root / 'ais_bench/benchmark/openicl/icl_inferencer/output_handler/gen_inferencer_output_handler.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'get_prediction_result')
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    namespace = {'Output': FakeOutput}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 str(path), 'exec'), namespace)
    return namespace['get_prediction_result']


class PredictionLogprobsSwitchTest(unittest.TestCase):
    def test_disabled_preserves_customer_logprobs(self):
        for storage in (None, {}, {'work_dir': 'outputs'}):
            output = FakeOutput([{'token': 'answer', 'logprob': -0.1}])
            result = load_prediction_method()(SimpleNamespace(response_anomaly_enabled=False, response_anomaly_payload_storage=storage), output)
            self.assertEqual(result['origin_logprobs'], output.origin_logprobs)
            self.assertEqual(result['input_tokens'], 3)
            self.assertEqual(result['output_tokens'], 2)

    def test_enabled_omits_logprobs_even_without_payload(self):
        for payload in (None, {'tokens': [1], 'topk_logprobs': [{1: -0.1}]}):
            output = FakeOutput([{'token': 'answer', 'logprob': -0.1}], payload)
            handler = SimpleNamespace(response_anomaly_enabled=True, response_anomaly_payload_storage=None)
            result = load_prediction_method()(handler, output)
            self.assertNotIn('origin_logprobs', result)
            if payload is not None:
                self.assertEqual(result['response_anomaly_payload'], payload)

    def test_empty_logprobs_not_written(self):
        result = load_prediction_method()(SimpleNamespace(response_anomaly_enabled=False), FakeOutput([]))
        self.assertNotIn('origin_logprobs', result)


if __name__ == '__main__':
    unittest.main()
