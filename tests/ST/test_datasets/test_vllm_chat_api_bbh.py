import os
import json
import shutil
import sys
import logging
import pytest
from ais_bench.benchmark.cli.main import main


settings = [
    ('temporal_sequences', 'mcq'),
    ('disambiguation_qa', 'mcq'),
    ('date_understanding', 'mcq'),
    ('tracking_shuffled_objects_three_objects', 'mcq'),
    ('penguins_in_a_table', 'mcq'),
    ('geometric_shapes', 'mcq'),
    ('snarks', 'mcq'),
    ('ruin_names', 'mcq'),
    ('tracking_shuffled_objects_seven_objects', 'mcq'),
    ('tracking_shuffled_objects_five_objects', 'mcq'),
    ('logical_deduction_three_objects', 'mcq'),
    ('hyperbaton', 'mcq'),
    ('logical_deduction_five_objects', 'mcq'),
    ('logical_deduction_seven_objects', 'mcq'),
    ('movie_recommendation', 'mcq'),
    ('salient_translation_error_detection', 'mcq'),
    ('reasoning_about_colored_objects', 'mcq'),
    ('multistep_arithmetic_two', 'free_form'),
    ('navigate', 'free_form'),
    ('dyck_languages', 'free_form'),
    ('word_sorting', 'free_form'),
    ('sports_understanding', 'free_form'),
    ('boolean_expressions', 'free_form'),
    ('object_counting', 'free_form'),
    ('formal_fallacies', 'free_form'),
    ('causal_judgement', 'free_form'),
    ('web_of_lies', 'free_form'),
]

class Response:
    def __init__(self):
        data = {'choices': [{"message": {"content": "11"}}]}
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

    def test_vllm_api_chat_all_bbh_3_shot_cot_chat(self, monkeypatch):
        fake_prediction = "11"
        fake_time_str = "vllm_chat_bbh_gen_3_shot_cot_chat"
        datasets_abbr_name = "bbh-"
        datasets_script_name = "bbh_gen_3_shot_cot_chat"
        monkeypatch.setattr('sys.argv',
                            ["ais_bench", "--models", "vllm_api_general_chat", "--datasets", datasets_script_name,
                             "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("urllib3.PoolManager.request", lambda *args, **kwargs: Response())
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        for name, _ in settings:
            curr_datasets_abbr_name = datasets_abbr_name + name.replace(" ", "_")

            # check infer out
            infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general-chat/{curr_datasets_abbr_name}.json")
            assert os.path.exists(infer_outputs_json_path)
            with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            assert data.get(f"0").get("prediction") == fake_prediction

            # check eval out
            results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general-chat/{curr_datasets_abbr_name}.json")
            with open(results_json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            assert data.get("score") is not None

        # check vis
        vis_csv_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.csv")
        assert os.path.exists(vis_csv_path)
        vis_txt_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.txt")
        assert os.path.exists(vis_txt_path)
        vis_md_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.md")
        assert os.path.exists(vis_md_path)