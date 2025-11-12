import os
import json
import shutil
import sys
import logging
import pytest
import requests
from unittest.mock import patch
from ais_bench.benchmark.cli.main import main
import pandas as pd

AIME_DATA_COUNT = 30

class Response:
    def __init__(self):
        self.response = []
        for s in ["A","is","ben", "ch", "20"]:
            data = {
                "id":"a123","model_name":"qwen","model_version":None,"text_output":s,
            }
            self.response.append(f"data: {json.dumps(data)}\n")
    def stream(self,*args):
        for content in self.response:
            yield content.encode()

class TestClass:
    @classmethod
    def setup_class(cls):
        """
        class level setup_class
        """
        cls.init(TestClass)

    @classmethod
    def teardown_class(cls):

        print('\n ---class level teardown_class')

    def init(self):
        self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_data_path = os.path.abspath(os.path.join(self.cur_dir, "../testdatas"))
        if os.path.exists(self.test_data_path):
            shutil.rmtree(self.test_data_path)
        os.makedirs(self.test_data_path)
        self.perf_json_keys = ['Benchmark Duration', 'Total Requests', 'Failed Requests', 'Success Requests',
                               'Concurrency', 'Max Concurrency', 'Request Throughput', 'Total Input Tokens',
                               'Prefill Token Throughput', 'Total generated tokens', 'Input Token Throughput',
                               'Output Token Throughput', 'Total Token Throughput']
        self.perf_csv_headers = ['Performance Parameters', 'Average', 'Min', 'Max',
                                  'Median', 'P75', 'P90', 'P99', 'N']
        self.perf_csv_params = ['E2EL', 'TTFT', 'TPOT', 'InputTokens', 'OutputTokens',
                                 'OutputTokenThroughput']

    # mode infer
    def test_triton_stream_api_infer(self, monkeypatch):
        fake_prediction = "Aisbench20"
        fake_time_str = "triton_stream_aime2024_gen_0_shot_str"
        datasets_abbr_name = "aime2024"
        datasets_script_name = "aime2024_gen_0_shot_str"
        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "triton_stream_api_general", "--datasets", datasets_script_name,
            "--mode", "infer", "-w", self.test_data_path])
        monkeypatch.setattr("urllib3.PoolManager.request", lambda *args, **kwargs: Response())
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg, **kwargs: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/triton-stream-api-general/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        assert len(data) == AIME_DATA_COUNT
        assert data.get(f"{AIME_DATA_COUNT - 1}").get("prediction") == fake_prediction

    # mode perf
    def test_triton_stream_api_perf(self, monkeypatch):
        fake_perf_data = [{'id': 0, 'input_data': 'A A', 'input_token_id': [32, 362, 362],
                            'output': ' A A A', 'output_token_id': [362, 362, 362],
                            'prefill_latency': 56.9,
                            'decode_token_latencies': [26.4, 28.4], 'last_decode_latency': 28.4,
                            'decode_max_token_latency': 28.4, 'seq_latency': 2700.04,
                            'input_tokens_len': 2, 'generate_tokens_len': 3,
                            'generate_tokens_speed': 37.03, 'input_characters_len': 3,
                            'generate_characters_len': 6, 'characters_per_token': 2.0,
                            'prefill_batch_size': 0, 'decode_batch_size': [], 'queue_wait_time': [],
                            'request_id': '591c69416c694a6ab3194a06d6e1ed17',
                            'start_time': 1742952029.5993671, 'end_time': 1742952032.299417,
                            'is_success': True, 'is_empty': False, "chunk_time_point_list": [1, 2, 3, 4]}]
        fake_time_str = "triton_stream_aime2024_gen_0_shot_str_perf"
        datasets_abbr_name = "aime2024dataset"
        datasets_script_name = "aime2024_gen_0_shot_str"
        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "triton_stream_api_general", "--datasets", datasets_script_name,
            "--mode", "perf", "-w", self.test_data_path])
        monkeypatch.setattr("urllib3.PoolManager.request", lambda *args, **kwargs: Response())
        monkeypatch.setattr("ais_bench.benchmark.models.performance_api.PerformanceAPIModel.get_performance_data", lambda *arg: fake_perf_data)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg, **kwargs: fake_time_str)
        main()

        # check perf json
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/performances/triton-stream-api-general/{datasets_abbr_name}.json")
        perf_details_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/performances/triton-stream-api-general/{datasets_abbr_name}_details.json")
        assert os.path.exists(perf_details_json_path)
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r') as file:
            data = json.load(file)
        assert isinstance(data, dict)
        for key in self.perf_json_keys:
            assert key in data
        assert data['Total Requests']['total'] == len(fake_perf_data)

        #check perf csv
        infer_outputs_csv_path = os.path.join(self.test_data_path, f"{fake_time_str}/performances/triton-stream-api-general/{datasets_abbr_name}.csv")
        assert os.path.exists(infer_outputs_csv_path)

        data = pd.read_csv(infer_outputs_csv_path)
        for header in self.perf_csv_headers:
            assert header in data.columns
        first_column = data.iloc[:,0]
        for param in self.perf_csv_params:
            assert param in first_column.values

        assert data.loc[data['Performance Parameters'] == 'ITL', 'Max'].values[0] == str(fake_perf_data[0]['decode_max_token_latency']) + ' ms'
        assert data.loc[data['Performance Parameters'] == 'OutputTokenThroughput', 'Average'].values[0] == str(fake_perf_data[0]['generate_tokens_speed']) + ' token/s'