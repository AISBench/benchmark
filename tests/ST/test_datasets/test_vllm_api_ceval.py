import os
import json
import shutil
import sys
import logging
import pytest
from ais_bench.benchmark.cli.main import main


ceval_all_sets = [
    'computer_network',
    'operating_system',
    'computer_architecture',
    'college_programming',
    'college_physics',
    'college_chemistry',
    'advanced_mathematics',
    'probability_and_statistics',
    'discrete_mathematics',
    'electrical_engineer',
    'metrology_engineer',
    'high_school_mathematics',
    'high_school_physics',
    'high_school_chemistry',
    'high_school_biology',
    'middle_school_mathematics',
    'middle_school_biology',
    'middle_school_physics',
    'middle_school_chemistry',
    'veterinary_medicine',
    'college_economics',
    'business_administration',
    'marxism',
    'mao_zedong_thought',
    'education_science',
    'teacher_qualification',
    'high_school_politics',
    'high_school_geography',
    'middle_school_politics',
    'middle_school_geography',
    'modern_chinese_history',
    'ideological_and_moral_cultivation',
    'logic',
    'law',
    'chinese_language_and_literature',
    'art_studies',
    'professional_tour_guide',
    'legal_professional',
    'high_school_chinese',
    'high_school_history',
    'middle_school_history',
    'civil_servant',
    'sports_science',
    'plant_protection',
    'basic_medicine',
    'clinical_medicine',
    'urban_and_rural_planner',
    'accountant',
    'fire_engineer',
    'environmental_impact_assessment_engineer',
    'tax_accountant',
    'physician',
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
    def test_vllm_api_general_all_ceval_0_shot_str(self, monkeypatch):
        fake_prediction = "11"
        fake_time_str = "vllm_ceval_gen_0_shot_str"
        datasets_abbr_name = "ceval-"
        datasets_script_name = "ceval_gen_0_shot_str"

        monkeypatch.setattr('sys.argv',
                            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
                             "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("urllib3.PoolManager.request", lambda *args, **kwargs: Response())
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        for category in ceval_all_sets:
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

    def test_vllm_api_general_all_ceval_5_shot_str(self, monkeypatch):
        fake_prediction = "11"
        fake_time_str = "vllm_ceval_gen_5_shot_str"
        datasets_abbr_name = "ceval-"
        datasets_script_name = "ceval_gen_5_shot_str"

        monkeypatch.setattr('sys.argv',
                            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
                             "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("urllib3.PoolManager.request", lambda *args, **kwargs: Response())
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        for category in ceval_all_sets:
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