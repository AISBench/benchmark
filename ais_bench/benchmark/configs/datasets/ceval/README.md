# C-Eval
中文 | [English](README_en.md)
## 数据集简介
C-Eval 是一个针对基础模型的综合性中文评估套件。它包含 13948 道多项选择题，涵盖 52 个不同学科以及四个难度等级。

> 🔗 数据集主页链接[https://github.com/SJTU-LIT/ceval#data](https://github.com/SJTU-LIT/ceval#data)

## 数据集部署
- 可以从魔塔社区提供的链接🔗 [https://www.modelscope.cn/datasets/opencompass/ceval-exam/resolve/master/ceval-exam.zip](https://www.modelscope.cn/datasets/opencompass/ceval-exam/resolve/master/ceval-exam.zip)下载数据集压缩包。
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
mkdir ceval/
mkdir ceval/formal_ceval
cd ceval/formal_ceval
wget https://www.modelscope.cn/datasets/opencompass/ceval-exam/resolve/master/ceval-exam.zip
unzip ceval-exam.zip
rm ceval-exam.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree ceval/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    ceval
    └── formal_ceval
        ├── dev
        │   ├── accountant_dev.csv
        │   ├── advanced_mathematics_dev.csv
        │   ├── art_studies_dev.csv
        │   ├── basic_medicine_dev.csv
        │   ├── business_administration_dev.csv
        │   ├── chinese_language_and_literature_dev.csv
        │   ├── civil_servant_dev.csv
        │   ├── clinical_medicine_dev.csv
        │   ├── college_chemistry_dev.csv
        │   ├── college_economics_dev.csv
        │   ├── college_physics_dev.csv
        │   ├── college_programming_dev.csv
        │   ├── computer_architecture_dev.csv
        │   ├── computer_network_dev.csv
        │   ├── discrete_mathematics_dev.csv
        │   ├── education_science_dev.csv
        │   ├── electrical_engineer_dev.csv
        │   ├── environmental_impact_assessment_engineer_dev.csv
        │   ├── fire_engineer_dev.csv
        │   ├── high_school_biology_dev.csv
        │   ├── high_school_chemistry_dev.csv
        │   ├── high_school_chinese_dev.csv
        │   ├── high_school_geography_dev.csv
        │   ├── high_school_history_dev.csv
        │   ├── high_school_mathematics_dev.csv
        │   ├── high_school_physics_dev.csv
        │   ├── high_school_politics_dev.csv
        │   ├── ideological_and_moral_cultivation_dev.csv
        │   ├── law_dev.csv
        │   ├── legal_professional_dev.csv
        │   ├── logic_dev.csv
        │   ├── mao_zedong_thought_dev.csv
        │   ├── marxism_dev.csv
        │   ├── metrology_engineer_dev.csv
        │   ├── middle_school_biology_dev.csv
        │   ├── middle_school_chemistry_dev.csv
        │   ├── middle_school_geography_dev.csv
        │   ├── middle_school_history_dev.csv
        │   ├── middle_school_mathematics_dev.csv
        │   ├── middle_school_physics_dev.csv
        │   ├── middle_school_politics_dev.csv
        │   ├── modern_chinese_history_dev.csv
        │   ├── operating_system_dev.csv
        │   ├── physician_dev.csv
        │   ├── plant_protection_dev.csv
        │   ├── probability_and_statistics_dev.csv
        │   ├── professional_tour_guide_dev.csv
        │   ├── sports_science_dev.csv
        │   ├── tax_accountant_dev.csv
        │   ├── teacher_qualification_dev.csv
        │   ├── urban_and_rural_planner_dev.csv
        │   └── veterinary_medicine_dev.csv
        ├── test
        │   ├── accountant_test.csv
        │   ├── advanced_mathematics_test.csv
        │   ├── art_studies_test.csv
        │   ├── basic_medicine_test.csv
        │   ├── business_administration_test.csv
        │   ├── chinese_language_and_literature_test.csv
        │   ├── civil_servant_test.csv
        │   ├── clinical_medicine_test.csv
        │   ├── college_chemistry_test.csv
        │   ├── college_economics_test.csv
        │   ├── college_physics_test.csv
        │   ├── college_programming_test.csv
        │   ├── computer_architecture_test.csv
        │   ├── computer_network_test.csv
        │   ├── discrete_mathematics_test.csv
        │   ├── education_science_test.csv
        │   ├── electrical_engineer_test.csv
        │   ├── environmental_impact_assessment_engineer_test.csv
        │   ├── fire_engineer_test.csv
        │   ├── high_school_biology_test.csv
        │   ├── high_school_chemistry_test.csv
        │   ├── high_school_chinese_test.csv
        │   ├── high_school_geography_test.csv
        │   ├── high_school_history_test.csv
        │   ├── high_school_mathematics_test.csv
        │   ├── high_school_physics_test.csv
        │   ├── high_school_politics_test.csv
        │   ├── ideological_and_moral_cultivation_test.csv
        │   ├── law_test.csv
        │   ├── legal_professional_test.csv
        │   ├── logic_test.csv
        │   ├── mao_zedong_thought_test.csv
        │   ├── marxism_test.csv
        │   ├── metrology_engineer_test.csv
        │   ├── middle_school_biology_test.csv
        │   ├── middle_school_chemistry_test.csv
        │   ├── middle_school_geography_test.csv
        │   ├── middle_school_history_test.csv
        │   ├── middle_school_mathematics_test.csv
        │   ├── middle_school_physics_test.csv
        │   ├── middle_school_politics_test.csv
        │   ├── modern_chinese_history_test.csv
        │   ├── operating_system_test.csv
        │   ├── physician_test.csv
        │   ├── plant_protection_test.csv
        │   ├── probability_and_statistics_test.csv
        │   ├── professional_tour_guide_test.csv
        │   ├── sports_science_test.csv
        │   ├── tax_accountant_test.csv
        │   ├── teacher_qualification_test.csv
        │   ├── urban_and_rural_planner_test.csv
        │   └── veterinary_medicine_test.csv
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

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|ceval_gen_0_shot_str|C-Eval数据集生成式任务|accuracy|0-shot|字符串格式|[ceval_gen_0_shot_str.py](ceval_gen_0_shot_str.py)|
|ceval_gen_5_shot_str|C-Eval数据集生成式任务|accuracy|5-shot|字符串格式|[ceval_gen_5_shot_str.py](ceval_gen_5_shot_str.py)|
|ceval_gen_0_shot_cot_chat_prompt|C-Eval数据集生成式任务，prompt带逻辑链（对齐DeepSeek R1精度测试）|accuracy|0-shot|对话格式|[ceval_gen_0_shot_cot_chat_prompt.py](ceval_gen_0_shot_cot_chat_prompt.py)|
|ceval_ppl_0_shot_str|C-Eval数据集PPL任务|accuracy|0-shot|字符串格式|[ceval_ppl_0_shot_str.py](ceval_ppl_0_shot_str.py)|