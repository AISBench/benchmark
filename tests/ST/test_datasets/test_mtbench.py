import os
import json
import shutil
import sys
import pytest
import pandas as pd
from ais_bench.benchmark.cli.main import main


class Response:
    def __init__(self):
        data = {'choices': [{"text": "11"}]}
        self.data = f"{json.dumps(data)}".encode()


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
        self.perf_csv_params = ['E2EL', 'TTFT', 'TPOT', 'InputTokens', 'OutputTokens','OutputTokenThroughput']
        
    def test_vllm_api_stream_chat_multiturn_perf_mtbench(self, monkeypatch):
        fake_prediction = [{'id': 0, 'input_data': 'A A', 'input_token_id': [32, 362, 362],
                            'output': ' A A A', 'output_token_id': [362, 362, 362],
                            'prefill_latency': 56.9, "chunk_time_point_list": [1, 1, 1],
                            'decode_token_latencies': [26.4, 28.4], 'last_decode_latency': 28.4,
                            'decode_max_token_latency': 28.4, 'seq_latency': 2700.04,
                            'input_tokens_len': 2, 'generate_tokens_len': 3,
                            'generate_tokens_speed': 37.03, 'input_characters_len': 3,
                            'generate_characters_len': 6, 'characters_per_token': 2.0,
                            'prefill_batch_size': 0, 'decode_batch_size': [], 'queue_wait_time': [],
                            'request_id': '591c69416c694a6ab3194a06d6e1ed17',
                            'start_time': 1742952029.5993671, 'end_time': 1742952032.299417,
                            'is_success': True, 'is_empty': False}]
        fake_time_str = "mtbench_fake_time"
        datasets_abbr_name = "mtbenchdataset"
        datasets_script_name = "mtbench_gen"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_stream_chat_multiturn", "--datasets", datasets_script_name,
            "--mode", "perf", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.openicl.icl_inferencer.icl_gen_perf_inferencer.GenPerfInferencer.inference_with_multi_process", lambda *arg,**xargs: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.tasks.openicl_perf.OpenICLPerfTask.set_performance_api", lambda *arg: True)
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat_multiturn.VllmMultiturnAPIChatStream._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("urllib3.PoolManager.request", lambda *args, **kwargs: Response())
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check perf json
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/performances/vllm-multiturn-api-chat-stream/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r') as file:
            data = json.load(file)
        assert isinstance(data, dict)
        for key in self.perf_json_keys:
            assert key in data
        assert data['Total Requests']['total'] == len(fake_prediction)
        
        #check perf csv
        infer_outputs_csv_path = os.path.join(self.test_data_path, f"{fake_time_str}/performances/vllm-multiturn-api-chat-stream/{datasets_abbr_name}.csv")
        assert os.path.exists(infer_outputs_csv_path)
        data = pd.read_csv(infer_outputs_csv_path)
        for header in self.perf_csv_headers:
            assert header in data.columns
        first_column = data.iloc[:,0]
        for param in self.perf_csv_params:
            assert param in first_column.values

        assert data.loc[data['Performance Parameters'] == 'ITL', 'Max'].values[0] == str(fake_prediction[0]['decode_max_token_latency']) + ' ms'
        assert data.loc[data['Performance Parameters'] == 'OutputTokenThroughput', 'Average'].values[0] == str(fake_prediction[0]['generate_tokens_speed']) + ' token/s'