# C-Eval
[中文](README.md) | English
## Dataset Introduction
C-Eval is a comprehensive Chinese evaluation suite for foundation models. It contains 13,948 multiple-choice questions, covering 52 different disciplines and four difficulty levels.

> 🔗 Dataset Homepage Link: [https://github.com/SJTU-LIT/ceval#data](https://github.com/SJTU-LIT/ceval#data)

## Dataset Deployment
- The dataset compressed package can be downloaded from the link provided by ModelScope Community 🔗: [https://www.modelscope.cn/datasets/opencompass/ceval-exam/resolve/master/ceval-exam.zip](https://www.modelscope.cn/datasets/opencompass/ceval-exam/resolve/master/ceval-exam.zip).
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set in dataset tasks). Taking deployment on Linux as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
mkdir ceval/
mkdir ceval/formal_ceval
cd ceval/formal_ceval
wget https://www.modelscope.cn/datasets/opencompass/ceval-exam/resolve/master/ceval-exam.zip
unzip ceval-exam.zip
rm ceval-exam.zip
```
- Execute `tree ceval/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    ceval
    └── formal_ceval
        ├── dev
        │   ├── accountant_dev.csv
        │   ├── advanced_mathematics_dev.csv
        │   ├── art_studies_dev.csv
        │   ├── basic_medicine_dev.csv
        │   ├── business_administration_dev.csv
        │   ├── chinese_language_and_literature_dev.csv
        │   ├── civil_servant_dev.csv
        │   ├── clinical_medicine_dev.csv
        │   ├── college_chemistry_dev.csv
        │   ├── college_economics_dev.csv
        │   ├── college_physics_dev.csv
        │   ├── college_programming_dev.csv
        │   ├── computer_architecture_dev.csv
        │   ├── computer_network_dev.csv
        │   ├── discrete_mathematics_dev.csv
        │   ├── education_science_dev.csv
        │   ├── electrical_engineer_dev.csv
        │   ├── environmental_impact_assessment_engineer_dev.csv
        │   ├── fire_engineer_dev.csv
        │   ├── high_school_biology_dev.csv
        │   ├── high_school_chemistry_dev.csv
        │   ├── high_school_chinese_dev.csv
        │   ├── high_school_geography_dev.csv
        │   ├── high_school_history_dev.csv
        │   ├── high_school_mathematics_dev.csv
        │   ├── high_school_physics_dev.csv
        │   ├── high_school_politics_dev.csv
        │   ├── ideological_and_moral_cultivation_dev.csv
        │   ├── law_dev.csv
        │   ├── legal_professional_dev.csv
        │   ├── logic_dev.csv
        │   ├── mao_zedong_thought_dev.csv
        │   ├── marxism_dev.csv
        │   ├── metrology_engineer_dev.csv
        │   ├── middle_school_biology_dev.csv
        │   ├── middle_school_chemistry_dev.csv
        │   ├── middle_school_geography_dev.csv
        │   ├── middle_school_history_dev.csv
        │   ├── middle_school_mathematics_dev.csv
        │   ├── middle_school_physics_dev.csv
        │   ├── middle_school_politics_dev.csv
        │   ├── modern_chinese_history_dev.csv
        │   ├── operating_system_dev.csv
        │   ├── physician_dev.csv
        │   ├── plant_protection_dev.csv
        │   ├── probability_and_statistics_dev.csv
        │   ├── professional_tour_guide_dev.csv
        │   ├── sports_science_dev.csv
        │   ├── tax_accountant_dev.csv
        │   ├── teacher_qualification_dev.csv
        │   ├── urban_and_rural_planner_dev.csv
        │   └── veterinary_medicine_dev.csv
        ├── test
        │   ├── accountant_test.csv
        │   ├── advanced_mathematics_test.csv
        │   ├── art_studies_test.csv
        │   ├── basic_medicine_test.csv
        │   ├── business_administration_test.csv
        │   ├── chinese_language_and_literature_test.csv
        │   ├── civil_servant_test.csv
        │   ├── clinical_medicine_test.csv
        │   ├── college_chemistry_test.csv
        │   ├── college_economics_test.csv
        │   ├── college_physics_test.csv
        │   ├── college_programming_test.csv
        │   ├── computer_architecture_test.csv
        │   ├── computer_network_test.csv
        │   ├── discrete_mathematics_test.csv
        │   ├── education_science_test.csv
        │   ├── electrical_engineer_test.csv
        │   ├── environmental_impact_assessment_engineer_test.csv
        │   ├── fire_engineer_test.csv
        │   ├── high_school_biology_test.csv
        │   ├── high_school_chemistry_test.csv
        │   ├── high_school_chinese_test.csv
        │   ├── high_school_geography_test.csv
        │   ├── high_school_history_test.csv
        │   ├── high_school_mathematics_test.csv
        │   ├── high_school_physics_test.csv
        │   ├── high_school_politics_test.csv
        │   ├── ideological_and_moral_cultivation_test.csv
        │   ├── law_test.csv
        │   ├── legal_professional_test.csv
        │   ├── logic_test.csv
        │   ├── mao_zedong_thought_test.csv
        │   ├── marxism_test.csv
        │   ├── metrology_engineer_test.csv
        │   ├── middle_school_biology_test.csv
        │   ├── middle_school_chemistry_test.csv
        │   ├── middle_school_geography_test.csv
        │   ├── middle_school_history_test.csv
        │   ├── middle_school_mathematics_test.csv
        │   ├── middle_school_physics_test.csv
        │   ├── middle_school_politics_test.csv
        │   ├── modern_chinese_history_test.csv
        │   ├── operating_system_test.csv
        │   ├── physician_test.csv
        │   ├── plant_protection_test.csv
        │   ├── probability_and_statistics_test.csv
        │   ├── professional_tour_guide_test.csv
        │   ├── sports_science_test.csv
        │   ├── tax_accountant_test.csv
        │   ├── teacher_qualification_test.csv
        │   ├── urban_and_rural_planner_test.csv
        │   └── veterinary_medicine_test.csv
        └── val
            ├── accountant_val.csv
            ├── advanced_mathematics_val.csv
            ├── art_studies_val.csv
            ├── basic_medicine_val.csv
            ├── business_administration_val.csv
            ├── chinese_language_and_literature_val.csv
            ├── civil_servant_val.csv
            ├── clinical_medicine_val.csv
            ├── college_chemistry_val.csv
            ├── college_economics_val.csv
            ├── college_physics_val.csv
            ├── college_programming_val.csv
            ├── computer_architecture_val.csv
            ├── computer_network_val.csv
            ├── discrete_mathematics_val.csv
            ├── education_science_val.csv
            ├── electrical_engineer_val.csv
            ├── environmental_impact_assessment_engineer_val.csv
            ├── fire_engineer_val.csv
            ├── high_school_biology_val.csv
            ├── high_school_chemistry_val.csv
            ├── high_school_chinese_val.csv
            ├── high_school_geography_val.csv
            ├── high_school_history_val.csv
            ├── high_school_mathematics_val.csv
            ├── high_school_physics_val.csv
            ├── high_school_politics_val.csv
            ├── ideological_and_moral_cultivation_val.csv
            ├── law_val.csv
            ├── legal_professional_val.csv
            ├── logic_val.csv
            ├── mao_zedong_thought_val.csv
            ├── marxism_val.csv
            ├── metrology_engineer_val.csv
            ├── middle_school_biology_val.csv
            ├── middle_school_chemistry_val.csv
            ├── middle_school_geography_val.csv
            ├── middle_school_history_val.csv
            ├── middle_school_mathematics_val.csv
            ├── middle_school_physics_val.csv
            ├── middle_school_politics_val.csv
            ├── modern_chinese_history_val.csv
            ├── operating_system_val.csv
            ├── physician_val.csv
            ├── plant_protection_val.csv
            ├── probability_and_statistics_val.csv
            ├── professional_tour_guide_val.csv
            ├── sports_science_val.csv
            ├── tax_accountant_val.csv
            ├── teacher_qualification_val.csv
            ├── urban_and_rural_planner_val.csv
            └── veterinary_medicine_val.csv
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Import Statement | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- | --- |
| ceval_gen_0_shot_str | Generative task for the C-Eval dataset | Accuracy | 0-shot | String format |`from ais_bench.benchmark.configs.datasets.ceval.ceval_gen_0_shot_str import ceval_datasets as datasets`| [ceval_gen_0_shot_str.py](ceval_gen_0_shot_str.py) |
| ceval_gen_5_shot_str | Generative task for the C-Eval dataset | Accuracy | 5-shot | String format |`from ais_bench.benchmark.configs.datasets.ceval.ceval_gen_5_shot_str import ceval_datasets as datasets`| [ceval_gen_5_shot_str.py](ceval_gen_5_shot_str.py) |
| ceval_gen_0_shot_cot_chat_prompt | Generative task for the C-Eval dataset with logical chain in prompt (aligned with DeepSeek R1 accuracy test) | Accuracy | 0-shot | Chat format |`from ais_bench.benchmark.configs.datasets.ceval.ceval_gen_0_shot_cot_chat_prompt import ceval_datasets as datasets`| [ceval_gen_0_shot_cot_chat_prompt.py](ceval_gen_0_shot_cot_chat_prompt.py) |
| ceval_ppl_0_shot_str | PPL task for the C-Eval dataset | Accuracy | 0-shot | String format |`from ais_bench.benchmark.configs.datasets.ceval.ceval_ppl_0_shot_str import ceval_datasets as datasets`| [ceval_ppl_0_shot_str.py](ceval_ppl_0_shot_str.py) |