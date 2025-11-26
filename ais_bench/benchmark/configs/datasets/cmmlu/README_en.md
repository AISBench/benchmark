# CMMLU
[中文](README.md) | English
## Dataset Introduction
CMMLU (Chinese Massive Multitask Language Understanding) is a comprehensive capability evaluation system for large models specifically designed for the Chinese language and cultural context. It aims to systematically test the performance of language models in advanced knowledge reserves and reasoning abilities. This evaluation covers 67 subject themes and builds a complete knowledge system ranging from basic education to professional advancement. It includes not only science subjects requiring computational skills (such as physics and mathematics) but also fields in the humanities and social sciences. Due to the uniqueness of context and expression, many tasks are difficult to directly translate and implement in other languages. Additionally, the answers to a large number of questions in CMMLU have distinct Chinese local characteristics, and their correctness may not hold in other regions or language systems.

> 🔗 Dataset Homepage Link: [https://huggingface.co/datasets/haonan-li/cmmlu](https://huggingface.co/datasets/haonan-li/cmmlu)

## Dataset Deployment
- The dataset compressed package can be downloaded from the link provided by OpenCompass 🔗: [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/cmmlu.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/cmmlu.zip).
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set in dataset tasks). Taking deployment on Linux as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/cmmlu.zip
unzip cmmlu.zip
rm cmmlu.zip
```
- Execute `tree cmmlu/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    cmmlu
    ├── dev
    │   ├── agronomy.csv
    │   ├── anatomy.csv
    │   ├── ancient_chinese.csv
    │   ├── arts.csv
    │   ├── astronomy.csv
    │   ├── business_ethics.csv
    │   ├── chinese_civil_service_exam.csv
    │   ├── chinese_driving_rule.csv
    │   ├── chinese_food_culture.csv
    │   ├── chinese_foreign_policy.csv
    │   ├── chinese_history.csv
    │   ├── chinese_literature.csv
    │   ├── chinese_teacher_qualification.csv
    │   ├── clinical_knowledge.csv
    │   ├── college_actuarial_science.csv
    │   ├── college_education.csv
    │   ├── college_engineering_hydrology.csv
    │   ├── college_law.csv
    │   ├── college_mathematics.csv
    │   ├── college_medical_statistics.csv
    │   ├── college_medicine.csv
    │   ├── computer_science.csv
    │   ├── computer_security.csv
    │   ├── conceptual_physics.csv
    │   ├── construction_project_management.csv
    │   ├── economics.csv
    │   ├── education.csv
    │   ├── electrical_engineering.csv
    │   ├── elementary_chinese.csv
    │   ├── elementary_commonsense.csv
    │   ├── elementary_information_and_technology.csv
    │   ├── elementary_mathematics.csv
    │   ├── ethnology.csv
    │   ├── food_science.csv
    │   ├── genetics.csv
    │   ├── global_facts.csv
    │   ├── high_school_biology.csv
    │   ├── high_school_chemistry.csv
    │   ├── high_school_geography.csv
    │   ├── high_school_mathematics.csv
    │   ├── high_school_physics.csv
    │   ├── high_school_politics.csv
    │   ├── human_sexuality.csv
    │   ├── international_law.csv
    │   ├── journalism.csv
    │   ├── jurisprudence.csv
    │   ├── legal_and_moral_basis.csv
    │   ├── logical.csv
    │   ├── machine_learning.csv
    │   ├── management.csv
    │   ├── marketing.csv
    │   ├── marxist_theory.csv
    │   ├── modern_chinese.csv
    │   ├── nutrition.csv
    │   ├── philosophy.csv
    │   ├── professional_accounting.csv
    │   ├── professional_law.csv
    │   ├── professional_medicine.csv
    │   ├── professional_psychology.csv
    │   ├── public_relations.csv
    │   ├── security_study.csv
    │   ├── sociology.csv
    │   ├── sports_science.csv
    │   ├── traditional_chinese_medicine.csv
    │   ├── virology.csv
    │   ├── world_history.csv
    │   └── world_religions.csv
    └── test
        ├── agronomy.csv
        ├── anatomy.csv
        ├── ancient_chinese.csv
        ├── arts.csv
        ├── astronomy.csv
        ├── business_ethics.csv
        ├── chinese_civil_service_exam.csv
        ├── chinese_driving_rule.csv
        ├── chinese_food_culture.csv
        ├── chinese_foreign_policy.csv
        ├── chinese_history.csv
        ├── chinese_literature.csv
        ├── chinese_teacher_qualification.csv
        ├── clinical_knowledge.csv
        ├── college_actuarial_science.csv
        ├── college_education.csv
        ├── college_engineering_hydrology.csv
        ├── college_law.csv
        ├── college_mathematics.csv
        ├── college_medical_statistics.csv
        ├── college_medicine.csv
        ├── computer_science.csv
        ├── computer_security.csv
        ├── conceptual_physics.csv
        ├── construction_project_management.csv
        ├── economics.csv
        ├── education.csv
        ├── electrical_engineering.csv
        ├── elementary_chinese.csv
        ├── elementary_commonsense.csv
        ├── elementary_information_and_technology.csv
        ├── elementary_mathematics.csv
        ├── ethnology.csv
        ├── food_science.csv
        ├── genetics.csv
        ├── global_facts.csv
        ├── high_school_biology.csv
        ├── high_school_chemistry.csv
        ├── high_school_geography.csv
        ├── high_school_mathematics.csv
        ├── high_school_physics.csv
        ├── high_school_politics.csv
        ├── human_sexuality.csv
        ├── international_law.csv
        ├── journalism.csv
        ├── jurisprudence.csv
        ├── legal_and_moral_basis.csv
        ├── logical.csv
        ├── machine_learning.csv
        ├── management.csv
        ├── marketing.csv
        ├── marxist_theory.csv
        ├── modern_chinese.csv
        ├── nutrition.csv
        ├── philosophy.csv
        ├── professional_accounting.csv
        ├── professional_law.csv
        ├── professional_medicine.csv
        ├── professional_psychology.csv
        ├── public_relations.csv
        ├── security_study.csv
        ├── sociology.csv
        ├── sports_science.csv
        ├── traditional_chinese_medicine.csv
        ├── virology.csv
        ├── world_history.csv
        └── world_religions.csv
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
| cmmlu_gen_0_shot_cot_chat_prompt | Generative task for the CMMLU dataset with logical chain in prompt | Accuracy | 0-shot | Chat format | [cmmlu_gen_0_shot_cot_chat_prompt.py](cmmlu_gen_0_shot_cot_chat_prompt.py) |
| cmmlu_gen_5_shot_cot_chat_prompt | Generative task for the CMMLU dataset with logical chain in prompt | Accuracy | 5-shot | Chat format | [cmmlu_gen_5_shot_cot_chat_prompt.py](cmmlu_gen_5_shot_cot_chat_prompt.py) |
| cmmlu_ppl_0_shot_cot_chat_prompt | PPL task for the CMMLU dataset with logical chain in prompt | Accuracy | 0-shot | Chat format | [cmmlu_ppl_0_shot_cot_chat_prompt.py](cmmlu_ppl_0_shot_cot_chat_prompt.py) |