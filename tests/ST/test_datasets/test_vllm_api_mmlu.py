import os
import json
import shutil
import sys
import logging
import pytest
from ais_bench.benchmark.cli.main import main


mmlu_all_sets = [
    'college_biology',
    'college_chemistry',
    'college_computer_science',
    'college_mathematics',
    'college_physics',
    'electrical_engineering',
    'astronomy',
    'anatomy',
    'abstract_algebra',
    'machine_learning',
    'clinical_knowledge',
    'global_facts',
    'management',
    'nutrition',
    'marketing',
    'professional_accounting',
    'high_school_geography',
    'international_law',
    'moral_scenarios',
    'computer_security',
    'high_school_microeconomics',
    'professional_law',
    'medical_genetics',
    'professional_psychology',
    'jurisprudence',
    'world_religions',
    'philosophy',
    'virology',
    'high_school_chemistry',
    'public_relations',
    'high_school_macroeconomics',
    'human_sexuality',
    'elementary_mathematics',
    'high_school_physics',
    'high_school_computer_science',
    'high_school_european_history',
    'business_ethics',
    'moral_disputes',
    'high_school_statistics',
    'miscellaneous',
    'formal_logic',
    'high_school_government_and_politics',
    'prehistory',
    'security_studies',
    'high_school_biology',
    'logical_fallacies',
    'high_school_world_history',
    'professional_medicine',
    'high_school_mathematics',
    'college_medicine',
    'high_school_us_history',
    'sociology',
    'econometrics',
    'high_school_psychology',
    'human_aging',
    'us_foreign_policy',
    'conceptual_physics',
]

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

    # mode all
    def test_vllm_api_general_all_mmlu_5_shot_str(self, monkeypatch):
        fake_prediction = "11"
        fake_time_str = "vllm_mmlu_gen_5_shot_str"
        datasets_abbr_name = "lukaemon_mmlu_"
        datasets_script_name = "mmlu_gen_5_shot_str"

        monkeypatch.setattr('sys.argv',
                            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
                             "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("urllib3.PoolManager.request", lambda *args, **kwargs: Response())
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        for category in mmlu_all_sets:
            curr_datasets_abbr_name = datasets_abbr_name + category

            # check infer out
            infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general/{curr_datasets_abbr_name}.json")
            assert os.path.exists(infer_outputs_json_path)
            with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            assert data.get(f"0").get("prediction") == fake_prediction

            # check eval out
            results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general/{curr_datasets_abbr_name}.json")
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