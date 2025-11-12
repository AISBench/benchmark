import os
import json
import shutil
import sys
import logging
import pytest
from ais_bench.benchmark.cli.main import main

DATASETS_CONFIGS_LIST = [
    "mmlu",
    "gsm8k",
    "boolq",
    "bbh",
    "race",
    "ceval",
    "aime2024",
    "gpqa",
    "math",
    "mmlu_pro",
    "mgsm",
    "agieval",
    "cmmlu",
    "humanevalx",
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

    def _set_datasets_config_path(self):
        dataset_configs_base_dir = os.path.abspath(os.path.join(self.cur_dir, "../../ais_bench/benchmark/configs/datasets"))
        for dataset in DATASETS_CONFIGS_LIST:
            sys.path.append(os.path.join(dataset_configs_base_dir, dataset))

    # mode all
    def test_vllm_api_all_mmlu_5_shot_str(self, monkeypatch):
        from mmlu_gen_5_shot_str import mmlu_all_sets
        fake_prediction = "Answer: A"
        fake_time_str = "mmlu_gen_5_shot_str"
        datasets_abbr_name = "lukaemon_mmlu_"
        datasets_script_name = "mmlu_gen_5_shot_str"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("urllib3.PoolManager.request", lambda *args, **kwargs: Response(fake_prediction))
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

    def test_vllm_api_all_ceval_5_shot_str(self, monkeypatch):
        from ceval_gen_5_shot_str import ceval_all_sets
        fake_prediction = "A"
        fake_time_str = "ceval_gen_5_shot_str"
        datasets_abbr_name = "ceval-"
        datasets_script_name = "ceval_gen_5_shot_str"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("urllib3.PoolManager.request", lambda *args, **kwargs: Response(fake_prediction))
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

    def test_vllm_api_all_ceval_0_shot_str(self, monkeypatch):
        from ceval_gen_0_shot_str import ceval_all_sets
        fake_prediction = "A"
        fake_time_str = "ceval_gen_0_shot_str"
        datasets_abbr_name = "ceval-"
        datasets_script_name = "ceval_gen_0_shot_str"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("urllib3.PoolManager.request", lambda *args, **kwargs: Response(fake_prediction))
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

    def test_vllm_api_all_boolq_0_shot_str(self, monkeypatch):
        fake_prediction = "Yes"
        fake_time_str = "SuperGLUE_BoolQ_gen_0_shot_str"
        datasets_abbr_name = "BoolQ"
        datasets_script_name = "SuperGLUE_BoolQ_gen_0_shot_str"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("urllib3.PoolManager.request", lambda *args, **kwargs: Response(fake_prediction))
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general/{datasets_abbr_name}.json")
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

    def test_vllm_api_all_aime2024_0_shot_str(self, monkeypatch):
        fake_prediction = "11"
        fake_time_str = "aime2024_gen_0_shot_str"
        datasets_abbr_name = "aime2024"
        datasets_script_name = "aime2024_gen_0_shot_str"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("urllib3.PoolManager.request", lambda *args, **kwargs: Response(fake_prediction))
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general/{datasets_abbr_name}.json")
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

    def test_vllm_api_all_qwen2_7b_gpqa_0_shot_str(self, monkeypatch):
        fake_prediction = "A"
        fake_time_str = "gpqa_gen_0_shot_str"
        datasets_abbr_name = "GPQA_diamond"
        datasets_script_name = "gpqa_gen_0_shot_str"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("urllib3.PoolManager.request", lambda *args, **kwargs: Response(fake_prediction))
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general/{datasets_abbr_name}.json")
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


    def test_vllm_api_all_qwen2_7b_humanevalx_0_shot(self, monkeypatch): # 唯一的测试函数名
        fake_prediction = "112" # 模拟的推理输出，随便写吧
        fake_time_str = "humanevalx_0_shot" # 模拟的时间戳，需要确保和其他用例不重复
        datasets_abbr_name = "humanevalx-" # 被测数据集配置文件中abbr的名称 humanevalx-
        datasets_script_name = "humanevalx_gen_0_shot" # 被测数据集配置文件名称
        languages = ['python', 'cpp', 'go', 'java', 'js']

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--summarizer", "example","--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("urllib3.PoolManager.request", lambda *args, **kwargs: Response(fake_prediction))
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        for lang in languages:
            curr_datasets_abbr_name = datasets_abbr_name + lang
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


    def test_vllm_api_all_qwen2_7b_math500_0_shot(self, monkeypatch):
        fake_prediction = "11"
        fake_time_str = "math500_0_shot"
        datasets_abbr_name = "math_prm800k_500"
        datasets_script_name = "math_prm800k_500_0shot_cot_gen"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("urllib3.PoolManager.request", lambda *args, **kwargs: Response(fake_prediction))
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general/{datasets_abbr_name}.json")
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

    def test_vllm_api_all_qwen2_7b_math500_5_shot(self, monkeypatch):
        fake_prediction = "11"
        fake_time_str = "math500_5_shot"
        datasets_abbr_name = "math_prm800k_500"
        datasets_script_name = "math_prm800k_500_5shot_cot_gen"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("urllib3.PoolManager.request", lambda *args, **kwargs: Response(fake_prediction))
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general/{datasets_abbr_name}.json")
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

    def test_vllm_api_all_qwen2_7b_drop_0_shot(self, monkeypatch):
        fake_prediction = "11"
        fake_time_str = "drop_gen_0_shot_str"
        datasets_abbr_name = "drop"
        datasets_script_name = "drop_gen_0_shot_str"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("urllib3.PoolManager.request", lambda *args, **kwargs: Response(fake_prediction))
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general/{datasets_abbr_name}.json")
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

    def test_vllm_api_all_qwen2_7b_drop_3_shot(self, monkeypatch):
        fake_prediction = "11"
        fake_time_str = "drop_gen_3_shot_str"
        datasets_abbr_name = "drop"
        datasets_script_name = "drop_gen_3_shot_str"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("urllib3.PoolManager.request", lambda *args, **kwargs: Response(fake_prediction))
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general/{datasets_abbr_name}.json")
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

    def test_vllm_api_all_qwen2_7b_humaneval_0_shot(self, monkeypatch):
        fake_prediction = "xxxx"
        fake_time_str = "humaneval_0_shot"
        datasets_abbr_name = "openai_humaneval"
        datasets_script_name = "humaneval_gen_0_shot"
        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("urllib3.PoolManager.request", lambda *args, **kwargs: Response(fake_prediction))
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general/{datasets_abbr_name}.json")
        with open(results_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get("humaneval_pass@1") is not None

        # check vis
        vis_csv_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.csv")
        assert os.path.exists(vis_csv_path)
        vis_txt_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.txt")
        assert os.path.exists(vis_txt_path)
        vis_md_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.md")
        assert os.path.exists(vis_md_path)

    def test_vllm_api_all_qwen2_7b_mmlu_pro_0_shot(self, monkeypatch):
        from mmlu_pro_categories import categories
        fake_prediction = "Answer: A"
        fake_time_str = "mmlu_pro_gen_0_shot"
        datasets_abbr_name = "mmlu_pro_"
        datasets_script_name = "mmlu_pro_gen_0_shot_str"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("urllib3.PoolManager.request", lambda *args, **kwargs: Response(fake_prediction))
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        for category in categories:
            curr_datasets_abbr_name = datasets_abbr_name + category.replace(" ", "_")

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

    def test_vllm_api_all_qwen2_7b_mmlu_pro_5_shot(self, monkeypatch):
        from mmlu_pro_categories import categories
        fake_prediction = "Answer: A"
        fake_time_str = "mmlu_pro_gen_5_shot_str"
        datasets_abbr_name = "mmlu_pro_"
        datasets_script_name = "mmlu_pro_gen_5_shot_str"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("urllib3.PoolManager.request", lambda *args, **kwargs: Response(fake_prediction))
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        for category in categories:
            curr_datasets_abbr_name = datasets_abbr_name + category.replace(" ", "_")

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

    def test_vllm_api_general_chat_lcb_code_gen_lite_0_shot(self, monkeypatch):
        fake_prediction = "xxxxxxxxxxx"
        fake_time_str = "lcb_code_gen_lite_0_shot"
        datasets_abbr_name = "lcb_code_generation"
        datasets_script_name = "livecodebench_code_generate_lite_gen_0_shot_chat"
        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general_chat", "--datasets", datasets_script_name, "--summarizer", "example",
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general-chat/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general-chat/{datasets_abbr_name}.json")
        with open(results_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get("pass@1") is not None

        # check vis
        vis_csv_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.csv")
        assert os.path.exists(vis_csv_path)
        vis_txt_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.txt")
        assert os.path.exists(vis_txt_path)
        vis_md_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.md")
        assert os.path.exists(vis_md_path)

    def test_vllm_api_general_chat_piqa_0_shot(self, monkeypatch):
        fake_prediction = "A"
        fake_time_str = "piqa_str_0_shot"
        datasets_abbr_name = "piqa"
        datasets_script_name = "piqa_gen_0_shot_chat_prompt"
        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general_chat", "--datasets", datasets_script_name, "--summarizer", "example",
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general-chat/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general-chat/{datasets_abbr_name}.json")
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

    def test_vllm_api_general_piqa_0_shot(self, monkeypatch):
        fake_prediction = "A"
        fake_time_str = "piqa_chat_0_shot"
        datasets_abbr_name = "piqa"
        datasets_script_name = "piqa_gen_0_shot_str"
        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("urllib3.PoolManager.request", lambda *args, **kwargs: Response(fake_prediction))
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general/{datasets_abbr_name}.json")
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

    def test_vllm_api_general_chat_mgsm_0_shot(self, monkeypatch):
        from mgsm_gen_0_shot_cot_chat_prompt import ALL_LANGUAGES
        fake_prediction = "Answer: 123"
        fake_time_str = "mgsm_gen_0_shot_cot_chat_prompt"
        datasets_abbr_name = "mgsm_"
        datasets_script_name = "mgsm_gen_0_shot_cot_chat_prompt"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general_chat", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        for category in ALL_LANGUAGES:
            curr_datasets_abbr_name = datasets_abbr_name + category.replace(" ", "_")

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
            assert data.get("accuracy") is not None

        # check vis
        vis_csv_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.csv")
        assert os.path.exists(vis_csv_path)
        vis_txt_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.txt")
        assert os.path.exists(vis_txt_path)
        vis_md_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.md")
        assert os.path.exists(vis_md_path)

    def test_vllm_api_general_chat_mgsm_8_shot(self, monkeypatch):
        from mgsm_gen_8_shot_cot_chat_prompt import ALL_LANGUAGES
        fake_prediction = "Answer: 123"
        fake_time_str = "mgsm_gen_8_shot_cot_chat_prompt"
        datasets_abbr_name = "mgsm_"
        datasets_script_name = "mgsm_gen_8_shot_cot_chat_prompt"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general_chat", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        for category in ALL_LANGUAGES:
            curr_datasets_abbr_name = datasets_abbr_name + category.replace(" ", "_")

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
            assert data.get("accuracy") is not None

        # check vis
        vis_csv_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.csv")
        assert os.path.exists(vis_csv_path)
        vis_txt_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.txt")
        assert os.path.exists(vis_txt_path)
        vis_md_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.md")
        assert os.path.exists(vis_md_path)

    def test_vllm_api_general_chat_agieval_gen_0_shot(self, monkeypatch):
        from agieval_gen_0_shot_chat_prompt import agieval_all_sets
        fake_prediction = "Answer: A"
        fake_time_str = "agieval_gen_0_shot_chat_prompt"
        datasets_abbr_name = "agieval-"
        datasets_script_name = "agieval_gen_0_shot_chat_prompt"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general_chat", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        for category in agieval_all_sets:
            curr_datasets_abbr_name = datasets_abbr_name + category.replace(" ", "_")

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
            assert data.get("accuracy") is not None or data.get("score") is not None

        # check vis
        vis_csv_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.csv")
        assert os.path.exists(vis_csv_path)

    def test_vllm_api_general_chat_winogrande_0_shot(self, monkeypatch):
        fake_prediction = "Answer: A"
        fake_time_str = "winogrande_gen_0_shot_chat_prompt"
        datasets_abbr_name = "winogrande"
        datasets_script_name = "winogrande_gen_0_shot_chat_prompt"
        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general_chat", "--datasets", datasets_script_name, "--summarizer", "example",
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general-chat/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general-chat/{datasets_abbr_name}.json")
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

    def test_vllm_api_general_chat_winogrande_5_shot(self, monkeypatch):
        fake_prediction = "Answer: A"
        fake_time_str = "winogrande_gen_5_shot_chat_prompt"
        datasets_abbr_name = "winogrande"
        datasets_script_name = "winogrande_gen_5_shot_chat_prompt"
        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general_chat", "--datasets", datasets_script_name, "--summarizer", "example",
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general-chat/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general-chat/{datasets_abbr_name}.json")
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

    def test_vllm_api_general_chat_mbpp_3_shot(self, monkeypatch):
        fake_prediction = "Answer: XXXXX"
        fake_time_str = "mbpp_passk_gen_3_shot_chat_prompt"
        datasets_abbr_name = "mbpp_passk"
        datasets_script_name = "mbpp_passk_gen_3_shot_chat_prompt"
        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general_chat", "--datasets", datasets_script_name, "--summarizer", "example",
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general-chat/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general-chat/{datasets_abbr_name}.json")
        with open(results_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get("pass@1") is not None

        # check vis
        vis_csv_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.csv")
        assert os.path.exists(vis_csv_path)
        vis_txt_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.txt")
        assert os.path.exists(vis_txt_path)
        vis_md_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.md")
        assert os.path.exists(vis_md_path)

    def test_vllm_api_general_chat_sanitized_mbpp_3_shot(self, monkeypatch):
        fake_prediction = "Answer: XXXXX"
        fake_time_str = "sanitized_mbpp_passk_gen_3_shot_chat_prompt"
        datasets_abbr_name = "sanitized_mbpp_passk"
        datasets_script_name = "sanitized_mbpp_passk_gen_3_shot_chat_prompt"
        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general_chat", "--datasets", datasets_script_name, "--summarizer", "example",
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general-chat/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general-chat/{datasets_abbr_name}.json")
        with open(results_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get("pass@1") is not None

        # check vis
        vis_csv_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.csv")
        assert os.path.exists(vis_csv_path)
        vis_txt_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.txt")
        assert os.path.exists(vis_txt_path)
        vis_md_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.md")
        assert os.path.exists(vis_md_path)

    def test_vllm_api_general_chat_arc_c_0_shot(self, monkeypatch):
        fake_prediction = "Answer: A"
        fake_time_str = "ARC_c_gen_0_shot_chat_prompt"
        datasets_abbr_name = "ARC-c"
        datasets_script_name = "ARC_c_gen_0_shot_chat_prompt"
        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general_chat", "--datasets", datasets_script_name, "--summarizer", "example",
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general-chat/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general-chat/{datasets_abbr_name}.json")
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

    def test_vllm_api_general_chat_arc_c_25_shot(self, monkeypatch):
        fake_prediction = "Answer: A"
        fake_time_str = "ARC_c_gen_25_shot_chat_prompt"
        datasets_abbr_name = "ARC-c"
        datasets_script_name = "ARC_c_gen_25_shot_chat_prompt"
        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general_chat", "--datasets", datasets_script_name, "--summarizer", "example",
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general-chat/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general-chat/{datasets_abbr_name}.json")
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

    def test_vllm_api_general_chat_arc_e_0_shot(self, monkeypatch):
        fake_prediction = "Answer: A"
        fake_time_str = "ARC_e_gen_0_shot_chat_prompt"
        datasets_abbr_name = "ARC-e"
        datasets_script_name = "ARC_e_gen_0_shot_chat_prompt"
        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general_chat", "--datasets", datasets_script_name, "--summarizer", "example",
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general-chat/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general-chat/{datasets_abbr_name}.json")
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

    def test_vllm_api_general_chat_arc_e_25_shot(self, monkeypatch):
        fake_prediction = "Answer: A"
        fake_time_str = "ARC_e_gen_25_shot_chat_prompt"
        datasets_abbr_name = "ARC-e"
        datasets_script_name = "ARC_e_gen_25_shot_chat_prompt"
        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general_chat", "--datasets", datasets_script_name, "--summarizer", "example",
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general-chat/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general-chat/{datasets_abbr_name}.json")
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

    def test_vllm_api_general_chat_hellaswag_0_shot(self, monkeypatch):
        fake_prediction = "Answer: A"
        fake_time_str = "hellaswag_gen_0_shot_chat_prompt"
        datasets_abbr_name = "hellaswag"
        datasets_script_name = "hellaswag_gen_0_shot_chat_prompt"
        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general_chat", "--datasets", datasets_script_name, "--summarizer", "example",
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general-chat/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general-chat/{datasets_abbr_name}.json")
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

    def test_vllm_api_general_chat_hellaswag_10_shot(self, monkeypatch):
        fake_prediction = "Answer: A"
        fake_time_str = "hellaswag_gen_10_shot_chat_prompt"
        datasets_abbr_name = "hellaswag"
        datasets_script_name = "hellaswag_gen_10_shot_chat_prompt"
        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general_chat", "--datasets", datasets_script_name, "--summarizer", "example",
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general-chat/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general-chat/{datasets_abbr_name}.json")
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

    def test_vllm_api_general_chat_triviaqa_5_shot(self, monkeypatch):
        fake_prediction = "Answer: A"
        fake_time_str = "triviaqa_gen_5_shot_chat_prompt"
        datasets_abbr_name = "triviaqa_5shot"
        datasets_script_name = "triviaqa_gen_5_shot_chat_prompt"
        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general_chat", "--datasets", datasets_script_name, "--summarizer", "example",
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general-chat/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general-chat/{datasets_abbr_name}.json")
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

    def test_vllm_api_general_chat_cmmlu_0_shot(self, monkeypatch):
        from cmmlu_gen_0_shot_cot_chat_prompt import cmmlu_all_sets
        fake_prediction = "Answer: A"
        fake_time_str = "cmmlu_gen_0_shot_cot_chat_prompt"
        datasets_abbr_name = "cmmlu-"
        datasets_script_name = "cmmlu_gen_0_shot_cot_chat_prompt"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general_chat", "--datasets", datasets_script_name, "--summarizer", "example",
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        for category in cmmlu_all_sets:
            curr_datasets_abbr_name = datasets_abbr_name + category.replace(" ", "_")

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
            assert data.get("accuracy") is not None

        # check vis
        vis_csv_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.csv")
        assert os.path.exists(vis_csv_path)
        vis_txt_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.txt")
        assert os.path.exists(vis_txt_path)
        vis_md_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.md")
        assert os.path.exists(vis_md_path)

    def test_vllm_api_general_chat_cmmlu_5_shot(self, monkeypatch):
        from cmmlu_gen_5_shot_cot_chat_prompt import cmmlu_all_sets
        fake_prediction = "Answer: A"
        fake_time_str = "cmmlu_gen_5_shot_cot_chat_prompt"
        datasets_abbr_name = "cmmlu-"
        datasets_script_name = "cmmlu_gen_5_shot_cot_chat_prompt"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general_chat", "--datasets", datasets_script_name, "--summarizer", "example",
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        for category in cmmlu_all_sets:
            curr_datasets_abbr_name = datasets_abbr_name + category.replace(" ", "_")

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
            assert data.get("accuracy") is not None

        # check vis
        vis_csv_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.csv")
        assert os.path.exists(vis_csv_path)
        vis_txt_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.txt")
        assert os.path.exists(vis_txt_path)
        vis_md_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.md")
        assert os.path.exists(vis_md_path)

    def test_vllm_api_general_chat_bbh_3_shot_cot(self, monkeypatch):
        from bbh_subset_settings import settings
        fake_prediction = "So the answer is (A)."
        fake_time_str = "bbh_gen_3_shot_cot_chat"
        datasets_abbr_name = "bbh-"
        datasets_script_name = "bbh_gen_3_shot_cot_chat"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general_chat", "--datasets", datasets_script_name, "--summarizer", "example",
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

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

    def test_vllm_api_general_chat_race_middle_5_shot_cot(self, monkeypatch):
        fake_prediction = "ANSWER: C"
        fake_time_str = "race_middle_gen_5_shot_cot_chat"
        datasets_abbr_name = "race-middle"
        datasets_script_name = "race_middle_gen_5_shot_cot_chat"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general_chat", "--datasets", datasets_script_name, "--summarizer", "example",
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general-chat/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general-chat/{datasets_abbr_name}.json")
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

    def test_vllm_api_general_chat_race_high_5_shot_cot(self, monkeypatch):
        fake_prediction = "ANSWER: C"
        fake_time_str = "race_high_gen_5_shot_cot_chat"
        datasets_abbr_name = "race-high"
        datasets_script_name = "race_high_gen_5_shot_cot_chat"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general_chat", "--datasets", datasets_script_name, "--summarizer", "example",
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general-chat/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general-chat/{datasets_abbr_name}.json")
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

    def test_vllm_api_all_qwen2_7b_ifeval_0_shot(self, monkeypatch):
        fake_prediction = "11"
        fake_time_str = "ifeval_0_shot"
        datasets_abbr_name = "ifeval"
        datasets_script_name = "ifeval_0_shot_gen_str"
        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("urllib3.PoolManager.request", lambda *args, **kwargs: Response(fake_prediction))
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general/{datasets_abbr_name}.json")
        with open(results_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get("Prompt-level-strict-accuracy") is not None
        assert data.get("Inst-level-strict-accuracy") is not None
        assert data.get("Prompt-level-loose-accuracy") is not None
        assert data.get("Inst-level-loose-accuracy") is not None

        # check vis
        vis_csv_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.csv")
        assert os.path.exists(vis_csv_path)
        vis_txt_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.txt")
        assert os.path.exists(vis_txt_path)
        vis_md_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.md")
        assert os.path.exists(vis_md_path)
