import os
import json
import shutil
import sys
import logging
import pytest
from ais_bench.benchmark.cli.main import main

GSK8K_DATA_COUNT = 1319


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

    def test_mindie_llm_base_model_all_gsm8k_str(self, monkeypatch):
        fake_prediction = "123"
        fake_time_str = "aime2024_gen_0_shot_str"
        datasets_abbr_name = "aime2024"
        datasets_script_name = "aime2024_gen_0_shot_str"
        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "mindie_llm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path, "--summarizer", "example"])
        monkeypatch.setattr("ais_bench.benchmark.utils.run.get_models_attr", lambda *arg: "service")
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        monkeypatch.setattr(
            "ais_bench.benchmark.models.mindie_llm_api.MindieLLMModel.check_pa_runner",
            (lambda *arg, **kwargs: None))
        monkeypatch.setattr(
            "ais_bench.benchmark.models.mindie_llm_api.MindieLLMModel.warm_up",
            (lambda *arg, **kwargs: None))
        monkeypatch.setattr(
            "ais_bench.benchmark.models.mindie_llm_api.MindieLLMModel.get_model_or_runner",
            (lambda *arg, **kwargs: None))
        monkeypatch.setattr(
            "ais_bench.benchmark.models.mindie_llm_api.MindieLLMModel.generate",
            (lambda self, inputs, *arg, **kwargs: [fake_prediction for _ in range(len(inputs))]))
        monkeypatch.setattr(
            "ais_bench.benchmark.models.mindie_llm_api.MindieLLMModel.get_token_len",
            (lambda *arg, **kwargs: 512))
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/mindie-llm-api/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/mindie-llm-api/{datasets_abbr_name}.json")
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

    # this case monk to model is not effective, need to use ASCEND_RT_VISIBLE_DEVICES
    def test_mindie_llm_base_model_all_aime_str_perf(self, monkeypatch):
        fake_prediction = "123"
        fake_time_str = "aime2024_gen_0_shot_str_perf"
        datasets_abbr_name = "aime2024dataset"
        datasets_script_name = "aime2024_gen_0_shot_str"
        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "mindie_llm_api_general", "--datasets", datasets_script_name,
            "--mode", "perf", "-w", self.test_data_path, "--summarizer", "example"])
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        monkeypatch.setattr(
            "ais_bench.benchmark.models.mindie_llm_api.MindieLLMModel.check_pa_runner",
            (lambda *arg, **kwargs: None))
        monkeypatch.setattr(
            "ais_bench.benchmark.models.mindie_llm_api.MindieLLMModel.warm_up",
            (lambda *arg, **kwargs: None))
        monkeypatch.setattr(
            "ais_bench.benchmark.models.mindie_llm_api.MindieLLMModel.get_model_or_runner",
            (lambda *arg, **kwargs: None))
        monkeypatch.setattr(
            "ais_bench.benchmark.models.mindie_llm_api.MindieLLMModel.generate",
            (lambda self, inputs, *arg, **kwargs: [fake_prediction for _ in range(len(inputs))]))
        monkeypatch.setattr(
            "ais_bench.benchmark.models.mindie_llm_api.MindieLLMModel.get_token_len",
            (lambda *arg, **kwargs: 512))
        main()

        # check perf standard out
        perf_data_path = os.path.join(self.test_data_path, f"{fake_time_str}/performances/mindie-llm-api/{datasets_abbr_name}.json")
        assert os.path.exists(perf_data_path)
        with open(perf_data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get("e2e_latency") is not None

        # check perf special out
        perf_sp_data_path = os.path.join(self.test_data_path, f"{fake_time_str}/performances/mindie-llm-api/pa_runner_special_perf_data_{datasets_abbr_name}.json")
        with open(perf_sp_data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        for round in data:
            for key in ["batch_size", "seq_len_in", "seq_len_out", "total_time", "first_token_time", "non_first_token_time", "e2e_time", "e2e_throughput"]:
                assert round.get(key) is not None