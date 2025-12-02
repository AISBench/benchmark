# MMLU
[中文](README.md) | English
## Dataset Introduction
MMLU (Massive Multitask Language Understanding) is a new benchmark designed to measure the world knowledge that large models acquire during pre-training under zero-shot and few-shot scenarios. This makes the benchmark more challenging and more similar to how we evaluate humans. It covers 57 subjects across fields such as STEM, humanities, and social sciences. The difficulty level ranges from elementary to advanced, testing both world knowledge and problem-solving abilities. The subjects span traditional areas like mathematics and history to more specialized fields such as law and ethics. The granularity and breadth of the subjects make this benchmark an ideal choice for identifying the blind spots of models.

> 🔗 Dataset Homepage: [https://github.com/hendrycks/test](https://github.com/hendrycks/test)

## Dataset Deployment
- The dataset compressed package can be downloaded from the link provided by OpenCompass 🔗: [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mmlu.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mmlu.zip).
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set for dataset tasks). Taking deployment on a Linux server as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mmlu.zip
unzip mmlu.zip
rm mmlu.zip
```
- Execute `tree mmlu/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure matches the one shown below, the dataset has been deployed successfully:
    ```
    mmlu/
    ├── dev
    │   ├── abstract_algebra_dev.csv
    │   ├── anatomy_dev.csv
    │   ├── astronomy_dev.csv
    │   ├── business_ethics_dev.csv
    │   ├── clinical_knowledge_dev.csv
    │   ├── college_biology_dev.csv
    │   ├── college_chemistry_dev.csv
    │   ├── college_computer_science_dev.csv
    │   ├── college_mathematics_dev.csv
    │   ├── college_medicine_dev.csv
    │   ├── college_physics_dev.csv
    │   ├── computer_security_dev.csv
    │   ├── conceptual_physics_dev.csv
    │   ├── econometrics_dev.csv
    │   ├── electrical_engineering_dev.csv
    │   ├── elementary_mathematics_dev.csv
    │   ├── formal_logic_dev.csv
    │   ├── global_facts_dev.csv
    │   ├── high_school_biology_dev.csv
    │   ├── high_school_chemistry_dev.csv
    │   ├── high_school_computer_science_dev.csv
    │   ├── high_school_european_history_dev.csv
    │   ├── high_school_geography_dev.csv
    │   ├── high_school_government_and_politics_dev.csv
    │   ├── high_school_macroeconomics_dev.csv
    │   ├── high_school_mathematics_dev.csv
    │   ├── high_school_microeconomics_dev.csv
    │   ├── high_school_physics_dev.csv
    │   ├── high_school_psychology_dev.csv
    │   ├── high_school_statistics_dev.csv
    │   ├── high_school_us_history_dev.csv
    │   ├── high_school_world_history_dev.csv
    │   ├── human_aging_dev.csv
    │   ├── human_sexuality_dev.csv
    │   ├── international_law_dev.csv
    │   ├── jurisprudence_dev.csv
    │   ├── logical_fallacies_dev.csv
    │   ├── machine_learning_dev.csv
    │   ├── management_dev.csv
    │   ├── marketing_dev.csv
    │   ├── medical_genetics_dev.csv
    │   ├── miscellaneous_dev.csv
    │   ├── moral_disputes_dev.csv
    │   ├── moral_scenarios_dev.csv
    │   ├── nutrition_dev.csv
    │   ├── philosophy_dev.csv
    │   ├── prehistory_dev.csv
    │   ├── professional_accounting_dev.csv
    │   ├── professional_law_dev.csv
    │   ├── professional_medicine_dev.csv
    │   ├── professional_psychology_dev.csv
    │   ├── public_relations_dev.csv
    │   ├── security_studies_dev.csv
    │   ├── sociology_dev.csv
    │   ├── us_foreign_policy_dev.csv
    │   ├── virology_dev.csv
    │   └── world_religions_dev.csv
    ├── possibly_contaminated_urls.txt
    ├── README.txt
    ├── test
    │   ├── abstract_algebra_test.csv
    │   ├── anatomy_test.csv
    │   ├── astronomy_test.csv
    │   ├── business_ethics_test.csv
    │   ├── clinical_knowledge_test.csv
    │   ├── college_biology_test.csv
    │   ├── college_chemistry_test.csv
    │   ├── college_computer_science_test.csv
    │   ├── college_mathematics_test.csv
    │   ├── college_medicine_test.csv
    │   ├── college_physics_test.csv
    │   ├── computer_security_test.csv
    │   ├── conceptual_physics_test.csv
    │   ├── econometrics_test.csv
    │   ├── electrical_engineering_test.csv
    │   ├── elementary_mathematics_test.csv
    │   ├── formal_logic_test.csv
    │   ├── global_facts_test.csv
    │   ├── high_school_biology_test.csv
    │   ├── high_school_chemistry_test.csv
    │   ├── high_school_computer_science_test.csv
    │   ├── high_school_european_history_test.csv
    │   ├── high_school_geography_test.csv
    │   ├── high_school_government_and_politics_test.csv
    │   ├── high_school_macroeconomics_test.csv
    │   ├── high_school_mathematics_test.csv
    │   ├── high_school_microeconomics_test.csv
    │   ├── high_school_physics_test.csv
    │   ├── high_school_psychology_test.csv
    │   ├── high_school_statistics_test.csv
    │   ├── high_school_us_history_test.csv
    │   ├── high_school_world_history_test.csv
    │   ├── human_aging_test.csv
    │   ├── human_sexuality_test.csv
    │   ├── international_law_test.csv
    │   ├── jurisprudence_test.csv
    │   ├── logical_fallacies_test.csv
    │   ├── machine_learning_test.csv
    │   ├── management_test.csv
    │   ├── marketing_test.csv
    │   ├── medical_genetics_test.csv
    │   ├── miscellaneous_test.csv
    │   ├── MMLU_test_contamination_annotations.json
    │   ├── moral_disputes_test.csv
    │   ├── moral_scenarios_test.csv
    │   ├── nutrition_test.csv
    │   ├── philosophy_test.csv
    │   ├── prehistory_test.csv
    │   ├── professional_accounting_test.csv
    │   ├── professional_law_test.csv
    │   ├── professional_medicine_test.csv
    │   ├── professional_psychology_test.csv
    │   ├── public_relations_test.csv
    │   ├── security_studies_test.csv
    │   ├── sociology_test.csv
    │   ├── us_foreign_policy_test.csv
    │   ├── virology_test.csv
    │   └── world_religions_test.csv
    └── val
        ├── abstract_algebra_val.csv
        ├── anatomy_val.csv
        ├── astronomy_val.csv
        ├── business_ethics_val.csv
        ├── clinical_knowledge_val.csv
        ├── college_biology_val.csv
        ├── college_chemistry_val.csv
        ├── college_computer_science_val.csv
        ├── college_mathematics_val.csv
        ├── college_medicine_val.csv
        ├── college_physics_val.csv
        ├── computer_security_val.csv
        ├── conceptual_physics_val.csv
        ├── econometrics_val.csv
        ├── electrical_engineering_val.csv
        ├── elementary_mathematics_val.csv
        ├── formal_logic_val.csv
        ├── global_facts_val.csv
        ├── high_school_biology_val.csv
        ├── high_school_chemistry_val.csv
        ├── high_school_computer_science_val.csv
        ├── high_school_european_history_val.csv
        ├── high_school_geography_val.csv
        ├── high_school_government_and_politics_val.csv
        ├── high_school_macroeconomics_val.csv
        ├── high_school_mathematics_val.csv
        ├── high_school_microeconomics_val.csv
        ├── high_school_physics_val.csv
        ├── high_school_psychology_val.csv
        ├── high_school_statistics_val.csv
        ├── high_school_us_history_val.csv
        ├── high_school_world_history_val.csv
        ├── human_aging_val.csv
        ├── human_sexuality_val.csv
        ├── international_law_val.csv
        ├── jurisprudence_val.csv
        ├── logical_fallacies_val.csv
        ├── machine_learning_val.csv
        ├── management_val.csv
        ├── marketing_val.csv
        ├── medical_genetics_val.csv
        ├── miscellaneous_val.csv
        ├── moral_disputes_val.csv
        ├── moral_scenarios_val.csv
        ├── nutrition_val.csv
        ├── philosophy_val.csv
        ├── prehistory_val.csv
        ├── professional_accounting_val.csv
        ├── professional_law_val.csv
        ├── professional_medicine_val.csv
        ├── professional_psychology_val.csv
        ├── public_relations_val.csv
        ├── security_studies_val.csv
        ├── sociology_val.csv
        ├── us_foreign_policy_val.csv
        ├── virology_val.csv
        └── world_religions_val.csv
    ```

## Available Dataset Tasks
### mmlu_gen_5_shot_str
#### Basic Information
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
| mmlu_gen_5_shot_str | Generative task for the MMLU dataset | Accuracy (naive_average) | 5-shot | String format | [mmlu_gen_5_shot_str.py](mmlu_gen_5_shot_str.py) |
| mmlu_gen_0_shot_cot_chat_prompt | Generative task for the MMLU dataset, with a logical chain in the prompt (aligned with DeepSeek R1 accuracy test) | Accuracy (naive_average) | 0-shot | String format | [mmlu_gen_0_shot_cot_chat_prompt.py](mmlu_gen_0_shot_cot_chat_prompt.py) |
| mmlu_ppl_0_shot_str | MMLU dataset PPL task | Accuracy (naive_average) | 0-shot | String format | [mmlu_ppl_0_shot_str.py](mmlu_ppl_0_shot_str.py) |