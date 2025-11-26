# CMMLU
中文 | [English](README_en.md)
## 数据集简介
CMMLU是一套专门针对中文语言与文化背景设计的大模型综合能力评估体系，旨在系统检验语言模型在高级知识储备与推理能力上的表现。该评测涵盖67个学科主题，构建了从基础教育到专业进阶的完整知识体系，既包含物理、数学等需要计算能力的理科项目，也涉及人文社科等学科领域。由于语境和表述的特殊性，许多任务难以通过其他语言直接转译实现。此外，CMMLU中大量题目的答案具有鲜明的中国本土特征，其正确性在其他地区或语言体系中可能并不成立。

> 🔗 数据集主页链接[https://huggingface.co/datasets/haonan-li/cmmlu](https://huggingface.co/datasets/haonan-li/cmmlu)

## 数据集部署
- 可以从opencompass提供的链接🔗 [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/cmmlu.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/cmmlu.zip)下载数据集压缩包。
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/cmmlu.zip
unzip cmmlu.zip
rm cmmlu.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree cmmlu/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    cmmlu
    ├── dev
    │   ├── agronomy.csv
    │   ├── anatomy.csv
    │   ├── ancient_chinese.csv
    │   ├── arts.csv
    │   ├── astronomy.csv
    │   ├── business_ethics.csv
    │   ├── chinese_civil_service_exam.csv
    │   ├── chinese_driving_rule.csv
    │   ├── chinese_food_culture.csv
    │   ├── chinese_foreign_policy.csv
    │   ├── chinese_history.csv
    │   ├── chinese_literature.csv
    │   ├── chinese_teacher_qualification.csv
    │   ├── clinical_knowledge.csv
    │   ├── college_actuarial_science.csv
    │   ├── college_education.csv
    │   ├── college_engineering_hydrology.csv
    │   ├── college_law.csv
    │   ├── college_mathematics.csv
    │   ├── college_medical_statistics.csv
    │   ├── college_medicine.csv
    │   ├── computer_science.csv
    │   ├── computer_security.csv
    │   ├── conceptual_physics.csv
    │   ├── construction_project_management.csv
    │   ├── economics.csv
    │   ├── education.csv
    │   ├── electrical_engineering.csv
    │   ├── elementary_chinese.csv
    │   ├── elementary_commonsense.csv
    │   ├── elementary_information_and_technology.csv
    │   ├── elementary_mathematics.csv
    │   ├── ethnology.csv
    │   ├── food_science.csv
    │   ├── genetics.csv
    │   ├── global_facts.csv
    │   ├── high_school_biology.csv
    │   ├── high_school_chemistry.csv
    │   ├── high_school_geography.csv
    │   ├── high_school_mathematics.csv
    │   ├── high_school_physics.csv
    │   ├── high_school_politics.csv
    │   ├── human_sexuality.csv
    │   ├── international_law.csv
    │   ├── journalism.csv
    │   ├── jurisprudence.csv
    │   ├── legal_and_moral_basis.csv
    │   ├── logical.csv
    │   ├── machine_learning.csv
    │   ├── management.csv
    │   ├── marketing.csv
    │   ├── marxist_theory.csv
    │   ├── modern_chinese.csv
    │   ├── nutrition.csv
    │   ├── philosophy.csv
    │   ├── professional_accounting.csv
    │   ├── professional_law.csv
    │   ├── professional_medicine.csv
    │   ├── professional_psychology.csv
    │   ├── public_relations.csv
    │   ├── security_study.csv
    │   ├── sociology.csv
    │   ├── sports_science.csv
    │   ├── traditional_chinese_medicine.csv
    │   ├── virology.csv
    │   ├── world_history.csv
    │   └── world_religions.csv
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

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|cmmlu_gen_0_shot_cot_chat_prompt|CMMLU数据集生成式任务, prompt带逻辑链|accuracy|0-shot|对话格式|[cmmlu_gen_0_shot_cot_chat_prompt.py](cmmlu_gen_0_shot_cot_chat_prompt.py)|
|cmmlu_gen_5_shot_cot_chat_prompt|CMMLU数据集生成式任务, prompt带逻辑链|accuracy|5-shot|对话格式|[cmmlu_gen_5_shot_cot_chat_prompt.py](cmmlu_gen_5_shot_cot_chat_prompt.py)|
|cmmlu_ppl_0_shot_cot_chat_prompt|CMMLU数据集PPL任务，prompt带逻辑链|accuracy|0-shot|对话格式|[cmmlu_ppl_0_shot_cot_chat_prompt.py](cmmlu_ppl_0_shot_cot_chat_prompt.py)|