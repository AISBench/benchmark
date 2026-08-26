# MMLU
中文 | [English](README_en.md)
## 数据集简介
MMLU（Massive Multitask Language Understanding）是一个新的基准，用于衡量在零样本（zero-shot）和少样本（few-shot）情形下，大模型在预训练期间获得的世界知识。这使得该基准测试更具挑战性，也更类似于我们评估人类的方式。该基准涵盖 STEM、人文（humanities）、社会科学（social sciences）等领域的 57 个学科（subject）。 它的难度从初级到高级，既考验世界知识，又考验解决问题的能力。 学科范围从数学和历史等传统领域到法律和伦理等更为专业的领域。学科的粒度和广度使该基准成为识别模型盲点的理想选择。

> 🔗 数据集主页 [https://github.com/hendrycks/test](https://github.com/hendrycks/test)

## 数据集部署
- 可以从opencompass提供的链接🔗 [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mmlu.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mmlu.zip)下载数据集压缩包。
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mmlu.zip
unzip mmlu.zip
rm mmlu.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree mmlu/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    mmlu/
    ├── dev
    │   ├── abstract_algebra_dev.csv
    │   ├── anatomy_dev.csv
    │   ├── astronomy_dev.csv
    │   ├── business_ethics_dev.csv
    │   ├── clinical_knowledge_dev.csv
    │   ├── college_biology_dev.csv
    │   ├── college_chemistry_dev.csv
    │   ├── college_computer_science_dev.csv
    │   ├── college_mathematics_dev.csv
    │   ├── college_medicine_dev.csv
    │   ├── college_physics_dev.csv
    │   ├── computer_security_dev.csv
    │   ├── conceptual_physics_dev.csv
    │   ├── econometrics_dev.csv
    │   ├── electrical_engineering_dev.csv
    │   ├── elementary_mathematics_dev.csv
    │   ├── formal_logic_dev.csv
    │   ├── global_facts_dev.csv
    │   ├── high_school_biology_dev.csv
    │   ├── high_school_chemistry_dev.csv
    │   ├── high_school_computer_science_dev.csv
    │   ├── high_school_european_history_dev.csv
    │   ├── high_school_geography_dev.csv
    │   ├── high_school_government_and_politics_dev.csv
    │   ├── high_school_macroeconomics_dev.csv
    │   ├── high_school_mathematics_dev.csv
    │   ├── high_school_microeconomics_dev.csv
    │   ├── high_school_physics_dev.csv
    │   ├── high_school_psychology_dev.csv
    │   ├── high_school_statistics_dev.csv
    │   ├── high_school_us_history_dev.csv
    │   ├── high_school_world_history_dev.csv
    │   ├── human_aging_dev.csv
    │   ├── human_sexuality_dev.csv
    │   ├── international_law_dev.csv
    │   ├── jurisprudence_dev.csv
    │   ├── logical_fallacies_dev.csv
    │   ├── machine_learning_dev.csv
    │   ├── management_dev.csv
    │   ├── marketing_dev.csv
    │   ├── medical_genetics_dev.csv
    │   ├── miscellaneous_dev.csv
    │   ├── moral_disputes_dev.csv
    │   ├── moral_scenarios_dev.csv
    │   ├── nutrition_dev.csv
    │   ├── philosophy_dev.csv
    │   ├── prehistory_dev.csv
    │   ├── professional_accounting_dev.csv
    │   ├── professional_law_dev.csv
    │   ├── professional_medicine_dev.csv
    │   ├── professional_psychology_dev.csv
    │   ├── public_relations_dev.csv
    │   ├── security_studies_dev.csv
    │   ├── sociology_dev.csv
    │   ├── us_foreign_policy_dev.csv
    │   ├── virology_dev.csv
    │   └── world_religions_dev.csv
    ├── possibly_contaminated_urls.txt
    ├── README.txt
    ├── test
    │   ├── abstract_algebra_test.csv
    │   ├── anatomy_test.csv
    │   ├── astronomy_test.csv
    │   ├── business_ethics_test.csv
    │   ├── clinical_knowledge_test.csv
    │   ├── college_biology_test.csv
    │   ├── college_chemistry_test.csv
    │   ├── college_computer_science_test.csv
    │   ├── college_mathematics_test.csv
    │   ├── college_medicine_test.csv
    │   ├── college_physics_test.csv
    │   ├── computer_security_test.csv
    │   ├── conceptual_physics_test.csv
    │   ├── econometrics_test.csv
    │   ├── electrical_engineering_test.csv
    │   ├── elementary_mathematics_test.csv
    │   ├── formal_logic_test.csv
    │   ├── global_facts_test.csv
    │   ├── high_school_biology_test.csv
    │   ├── high_school_chemistry_test.csv
    │   ├── high_school_computer_science_test.csv
    │   ├── high_school_european_history_test.csv
    │   ├── high_school_geography_test.csv
    │   ├── high_school_government_and_politics_test.csv
    │   ├── high_school_macroeconomics_test.csv
    │   ├── high_school_mathematics_test.csv
    │   ├── high_school_microeconomics_test.csv
    │   ├── high_school_physics_test.csv
    │   ├── high_school_psychology_test.csv
    │   ├── high_school_statistics_test.csv
    │   ├── high_school_us_history_test.csv
    │   ├── high_school_world_history_test.csv
    │   ├── human_aging_test.csv
    │   ├── human_sexuality_test.csv
    │   ├── international_law_test.csv
    │   ├── jurisprudence_test.csv
    │   ├── logical_fallacies_test.csv
    │   ├── machine_learning_test.csv
    │   ├── management_test.csv
    │   ├── marketing_test.csv
    │   ├── medical_genetics_test.csv
    │   ├── miscellaneous_test.csv
    │   ├── MMLU_test_contamination_annotations.json
    │   ├── moral_disputes_test.csv
    │   ├── moral_scenarios_test.csv
    │   ├── nutrition_test.csv
    │   ├── philosophy_test.csv
    │   ├── prehistory_test.csv
    │   ├── professional_accounting_test.csv
    │   ├── professional_law_test.csv
    │   ├── professional_medicine_test.csv
    │   ├── professional_psychology_test.csv
    │   ├── public_relations_test.csv
    │   ├── security_studies_test.csv
    │   ├── sociology_test.csv
    │   ├── us_foreign_policy_test.csv
    │   ├── virology_test.csv
    │   └── world_religions_test.csv
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

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|配套文件导入方式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- | --- |
|mmlu_gen_5_shot_str|MMLU数据集生成式任务|accuracy(naive_average)|5-shot|字符串格式|`from ais_bench.benchmark.configs.datasets.mmlu.mmlu_gen_5_shot_str import mmlu_datasets as datasets`|[mmlu_gen_5_shot_str.py](mmlu_gen_5_shot_str.py)|
|mmlu_gen_5_shot_chat_prompt|MMLU数据集生成式任务|accuracy(naive_average)|5-shot|对话格式|`from ais_bench.benchmark.configs.datasets.mmlu.mmlu_gen_5_shot_chat_prompt import mmlu_datasets as datasets`|[mmlu_gen_5_shot_chat_prompt.py](mmlu_gen_5_shot_chat_prompt.py)|
|mmlu_ppl_0_shot_str|MMLU数据集PPL任务|accuracy(naive_average)|0-shot|字符串格式|`from ais_bench.benchmark.configs.datasets.mmlu.mmlu_ppl_0_shot_str import mmlu_datasets as datasets`|[mmlu_ppl_0_shot_str.py](mmlu_ppl_0_shot_str.py)|
|mmlu_gen_0_shot_cot_chat_prompt|MMLU数据集生成式任务，prompt带逻辑链（对齐DeepSeek R1精度测试）|accuracy(naive_average)|0-shot|对话格式|`from ais_bench.benchmark.configs.datasets.mmlu.mmlu_gen_0_shot_cot_chat_prompt import mmlu_datasets as datasets`|[mmlu_gen_0_shot_cot_chat_prompt.py](mmlu_gen_0_shot_cot_chat_prompt.py)|

