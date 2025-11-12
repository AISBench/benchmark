import os
import json
import shutil
import sys
import logging
import pytest
import pandas as pd
from ais_bench.benchmark.cli.main import main

DATASETS_CONFIGS_LIST = [
    "mmlu",
    "gsm8k",
    "boolq",
    "ceval",
    "aime2024",
    "gpqa",
    "math",
    "synthetic",
]


class Response:
    def __init__(self, response:str = "Answer is :C"):
        data = {'choices': [{"text": response}]}
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
        self._set_datasets_config_path(self)
        self.perf_json_keys = ['Benchmark Duration', 'Total Requests', 'Failed Requests', 'Success Requests',
                               'Concurrency', 'Max Concurrency', 'Request Throughput', 'Total Input Tokens',
                               'Prefill Token Throughput', 'Total generated tokens', 'Input Token Throughput',
                               'Output Token Throughput', 'Total Token Throughput']
        self.perf_csv_headers = ['Performance Parameters', 'Average', 'Min', 'Max',
                                  'Median', 'P75', 'P90', 'P99', 'N']
        self.perf_csv_params = ['E2EL', 'TTFT', 'TPOT', 'InputTokens', 'OutputTokens',
                                'PrefillTokenThroughput', 'OutputTokenThroughput']

    def _set_datasets_config_path(self):
        dataset_configs_base_dir = os.path.abspath(os.path.join(self.cur_dir, "../../ais_bench/benchmark/configs/datasets"))
        for dataset in DATASETS_CONFIGS_LIST:
            sys.path.append(os.path.join(dataset_configs_base_dir, dataset))

    def test_vllm_api_all_qwen2_7b_synthetic_0_shot(self, monkeypatch):
        fake_prediction = "123321"
        fake_time_str = "synthetic_0_shot"
        datasets_abbr_name = "vllm-api-gpu-synthetic"
        datasets_script_name = "synthetic_gen"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("urllib3.PoolManager.request", lambda *args, **kwargs: Response(fake_prediction))
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general/synthetic.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general/synthetic.json")
        with open(results_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get("accuracy") is not None

        # check vis
        vis_csv_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.csv")
        assert os.path.exists(vis_csv_path)
        vis_txt_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.txt")
        assert os.path.exists(vis_txt_path)
        vis_md_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.md")
        assert os.path.exists(vis_md_path)

    def test_perf_mindie_stream_api_qwen2_7b_synthetic_save_result(self, monkeypatch):
        fake_prediction = [{'id': 0, 'input_data': 'A A', 'input_token_id': [32, 362, 362],
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
                            'is_success': True, 'is_empty': False}]
        fake_time_str = "perf_synthetic_save_result"
        datasets_script_name = "synthetic_gen"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "mindie_stream_api_general", "--datasets", datasets_script_name,
            "--mode", "perf", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.openicl.icl_inferencer.icl_gen_perf_inferencer.GenPerfInferencer.inference_with_multi_process", lambda *arg,**xargs: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()
        # check perf json
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/performances/mindie-stream-api/syntheticdataset.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r') as file:
            data = json.load(file)
        assert isinstance(data, dict)
        for key in self.perf_json_keys:
            assert key in data
        assert data['Total Requests'] == len(fake_prediction)

        #check perf csv
        infer_outputs_csv_path = os.path.join(self.test_data_path, f"{fake_time_str}/performances/mindie-stream-api/syntheticdataset.csv")
        assert os.path.exists(infer_outputs_csv_path)

        data = pd.read_csv(infer_outputs_csv_path)
        for header in self.perf_csv_headers:
            assert header in data.columns
        first_column = data.iloc[:,0]
        for param in self.perf_csv_params:
            assert param in first_column.values

        assert data.loc[data['Performance Parameters'] == 'ITL', 'Max'].values[0] == str(fake_prediction[0]['decode_max_token_latency']) + ' ms'
        assert data.loc[data['Performance Parameters'] == 'OutputTokenThroughput', 'Average'].values[0] == str(fake_prediction[0]['generate_tokens_speed']) + ' token/s'

    def test_perf_mindie_stream_api_qwen2_7b_synthetic_in80_out110_req3(self, monkeypatch):
        fake_time_str = "perf_synthetic_config1"
        datasets_script_name = "synthetic_gen"
        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "mindie_stream_api_general", "--datasets", datasets_script_name,
            "--mode", "perf", "-w", self.test_data_path])
        synthetic_config = {
            "Type":"string",
            "RequestCount": 3,
            "StringConfig" : {
                "Input" : {
                    "Method": "uniform",
                    "Params": {"MinValue": 80, "MaxValue": 80}
                },
                "Output" : {
                    "Method": "gaussian",
                    "Params": {"Mean": 100, "Var": 0, "MinValue": 100, "MaxValue": 100}
                }
            },
            "TokenIdConfig" : {
                "ModelPath": "",
                "RequestSize": 10
            }
        }
        fake_prediction = [{'id': 0, 'input_data': 'A A', 'input_token_id': [32, 362, 362],
                    'output': ' A A A', 'output_token_id': [362, 362, 362],
                    'prefill_latency': 56.9, 'prefill_throughput': 333.6,
                    'decode_token_latencies': [26.4, 28.4], 'last_decode_latency': 28.4,
                    'decode_max_token_latency': 28.4, 'seq_latency': 2700.04,
                    'input_tokens_len': 80, 'generate_tokens_len': 100,
                    'generate_tokens_speed': 37.03, 'input_characters_len': 3,
                    'generate_characters_len': 6, 'characters_per_token': 2.0,
                    'prefill_batch_size': 0, 'decode_batch_size': [], 'queue_wait_time': [],
                    'request_id': '591c69416c694a6ab3194a06d6e1ed17',
                    'start_time': 1742952029.5993671, 'end_time': 1742952032.299417,
                    'is_success': True, 'is_empty': False}] * synthetic_config['RequestCount']
        monkeypatch.setattr("ais_bench.benchmark.openicl.icl_inferencer.icl_gen_perf_inferencer.GenPerfInferencer.inference_with_multi_process", lambda *arg,**xargs: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.datasets.synthetic.get_synthetic_dataset_config", lambda *arg: synthetic_config)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check perf json
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/performances/mindie-stream-api/syntheticdataset.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert isinstance(data, dict)
        for key in self.perf_json_keys:
            assert key in data
        assert data['Total Requests'] == synthetic_config['RequestCount']

        #check perf csv
        infer_outputs_csv_path = os.path.join(self.test_data_path, f"{fake_time_str}/performances/mindie-stream-api/syntheticdataset.csv")
        assert os.path.exists(infer_outputs_csv_path)
        data = pd.read_csv(infer_outputs_csv_path)
        for header in self.perf_csv_headers:
            assert header in data.columns
        first_column = data.iloc[:,0]
        for param in self.perf_csv_params:
            assert param in first_column.values

        assert float(data.loc[data['Performance Parameters'] == 'InputTokens', 'Max'].values[0])==float(synthetic_config['StringConfig']['Input']['Params']['MinValue'])
        assert float(data.loc[data['Performance Parameters'] == 'OutputTokens', 'Max'].values[0])==float(synthetic_config['StringConfig']['Output']['Params']['MinValue'])

    def test_perf_mindie_stream_api_qwen2_7b_synthetic_in130_out200_req5(self, monkeypatch):
        fake_time_str = "perf_synthetic_config2"
        datasets_script_name = "synthetic_gen"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "mindie_stream_api_general", "--datasets", datasets_script_name,
            "--mode", "perf", "-w", self.test_data_path])
        synthetic_config = {
            "Type":"string",
            "RequestCount": 5,
            "StringConfig" : {
                "Input" : {
                    "Method": "uniform",
                    "Params": {"MinValue": 130, "MaxValue": 130}
                },
                "Output" : {
                    "Method": "gaussian",
                    "Params": {"Mean": 100, "Var": 200, "MinValue": 200, "MaxValue": 200}
                }
            },
            "TokenIdConfig" : {
                "ModelPath": "",
                "RequestSize": 10
            }
        }
        monkeypatch.setattr("ais_bench.benchmark.datasets.synthetic.get_synthetic_dataset_config", lambda *arg: synthetic_config)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check perf json
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/performances/mindie-stream-api/syntheticdataset.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert isinstance(data, dict)
        for key in self.perf_json_keys:
            assert key in data
        assert data['Total Requests'] == synthetic_config['RequestCount']

        #check perf csv
        infer_outputs_csv_path = os.path.join(self.test_data_path, f"{fake_time_str}/performances/mindie-stream-api/syntheticdataset.csv")
        assert os.path.exists(infer_outputs_csv_path)
        data = pd.read_csv(infer_outputs_csv_path)
        for header in self.perf_csv_headers:
            assert header in data.columns
        first_column = data.iloc[:,0]
        for param in self.perf_csv_params:
            assert param in first_column.values

        assert float(data.loc[data['Performance Parameters'] == 'InputTokens', 'Max'].values[0])==float(synthetic_config['StringConfig']['Input']['Params']['MinValue'])
        assert float(data.loc[data['Performance Parameters'] == 'OutputTokens', 'Max'].values[0])==float(synthetic_config['StringConfig']['Output']['Params']['MinValue'])

    def test_hf_base_model_perf_gsm8k(self, monkeypatch):
        os.environ['ASCEND_RT_VISIBLE_DEVICES']= '0'
        fake_prediction = "12333"
        fake_perf_result = {"Benchmark Duration":{"total":"23.6 ms"}}
        fake_time_str = "gsm8k_gen_4_shot_str"
        datasets_abbr_name = "gsm8k"
        datasets_script_name = "gsm8k_gen_4_shot_cot_str"
        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "hf_base_model", "--datasets", datasets_script_name,
            "--mode", "perf", "-w", self.test_data_path, "--summarizer", "example"])
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        monkeypatch.setattr(
            "ais_bench.benchmark.models.huggingface_above_v4_33.HuggingFaceBaseModel.generate",
            (lambda self, inputs, *arg, **kwargs: [fake_prediction for _ in range(len(inputs))]))
        monkeypatch.setattr(
            "ais_bench.benchmark.models.huggingface_above_v4_33.HuggingFaceBaseModel.handle_perf_result",
            (lambda *arg, **kwargs: fake_perf_result))
        monkeypatch.setattr(
            "ais_bench.benchmark.models.huggingface_above_v4_33._get_possible_max_seq_len",
            (lambda *arg, **kwargs: None))
        monkeypatch.setattr(
            "ais_bench.benchmark.models.huggingface_above_v4_33.HuggingFaceBaseModel._get_potential_stop_words",
            (lambda *arg, **kwargs: ['\n']))
        monkeypatch.setattr(
            "ais_bench.benchmark.models.huggingface_above_v4_33.HuggingFaceBaseModel._load_tokenizer",
            (lambda *arg, **kwargs: None))
        monkeypatch.setattr(
            "ais_bench.benchmark.models.huggingface_above_v4_33.HuggingFaceBaseModel._load_model",
            (lambda *arg, **kwargs: None))
        monkeypatch.setattr(
            "ais_bench.benchmark.models.huggingface_above_v4_33.HuggingFaceBaseModel.get_token_len",
            (lambda *arg, **kwargs: 512))
        main()

        # check perf result
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/performances/hf-base-model/{datasets_abbr_name}/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data == fake_perf_result

    def test_hf_chat_model_perf_synthetic(self, monkeypatch):
        fake_prediction = "12334"
        fake_perf_result = {"Benchmark Duration":{"total":"27.7 ms"}}
        fake_time_str = "synthetic_gen_4_shot_str"
        datasets_abbr_name = "synthetic"
        datasets_script_name = "synthetic_gen"
        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "hf_chat_model", "--datasets", datasets_script_name,
            "--mode", "perf", "-w", self.test_data_path, "--summarizer", "example"])
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        monkeypatch.setattr(
            "ais_bench.benchmark.models.huggingface_above_v4_33.HuggingFacewithChatTemplate.generate",
            (lambda self, inputs, *arg, **kwargs: [fake_prediction for _ in range(len(inputs))]))
        monkeypatch.setattr(
            "ais_bench.benchmark.models.huggingface_above_v4_33.HuggingFacewithChatTemplate.handle_perf_result",
            (lambda *arg, **kwargs: fake_perf_result))
        monkeypatch.setattr(
            "ais_bench.benchmark.models.huggingface_above_v4_33._get_possible_max_seq_len",
            (lambda *arg, **kwargs: None))
        monkeypatch.setattr(
            "ais_bench.benchmark.models.huggingface_above_v4_33.HuggingFacewithChatTemplate._get_potential_stop_words",
            (lambda *arg, **kwargs: ['\n']))
        monkeypatch.setattr(
            "ais_bench.benchmark.models.huggingface_above_v4_33.HuggingFacewithChatTemplate._load_tokenizer",
            (lambda *arg, **kwargs: None))
        monkeypatch.setattr(
            "ais_bench.benchmark.models.huggingface_above_v4_33.HuggingFacewithChatTemplate._load_model",
            (lambda *arg, **kwargs: None))
        monkeypatch.setattr(
            "ais_bench.benchmark.models.huggingface_above_v4_33.HuggingFacewithChatTemplate.get_token_len",
            (lambda *arg, **kwargs: 512))
        main()

        # check perf result
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/performances/hf-chat-model/{datasets_abbr_name}/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data == fake_perf_result


