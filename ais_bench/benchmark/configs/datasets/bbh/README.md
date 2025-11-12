# BBH
中文 | [English](README_en.md)
## 数据集简介
BIG-Bench（Srivastava等人，2022年）是一个多样化的评估测试集，其重点关注当前语言模型被认为尚无法完成的任务。尽管语言模型已在该基准测试中取得显著进展——BIG-Bench论文中的最佳模型通过少量示例提示（few-shot prompting），在65%的任务上超越了人类评估者的平均成绩。但究竟在哪些任务上语言模型仍落后于人类平均水平？这些任务是否真的超出了当前语言模型的解决能力？

> 🔗 数据集主页链接[https://huggingface.co/datasets/lukaemon/bbh](https://huggingface.co/datasets/lukaemon/bbh)

## 数据集部署
- 可以从opencompass提供的链接🔗 [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/BBH.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/BBH.zip)下载数据集压缩包。
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/BBH.zip
unzip BBH.zip
rm BBH.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree BBH/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    BBH
    ├── data
    │   ├── boolean_expressions.json
    │   ├── causal_judgement.json
    │   ├── date_understanding.json
    │   ├── disambiguation_qa.json
    │   ├── dyck_languages.json
    │   ├── formal_fallacies.json
    │   ├── geometric_shapes.json
    │   ├── hyperbaton.json
    │   ├── logical_deduction_five_objects.json
    │   ├── logical_deduction_seven_objects.json
    │   ├── logical_deduction_three_objects.json
    │   ├── movie_recommendation.json
    │   ├── multistep_arithmetic_two.json
    │   ├── navigate.json
    │   ├── object_counting.json
    │   ├── penguins_in_a_table.json
    │   ├── README.md
    │   ├── reasoning_about_colored_objects.json
    │   ├── ruin_names.json
    │   ├── salient_translation_error_detection.json
    │   ├── snarks.json
    │   ├── sports_understanding.json
    │   ├── temporal_sequences.json
    │   ├── tracking_shuffled_objects_five_objects.json
    │   ├── tracking_shuffled_objects_seven_objects.json
    │   ├── tracking_shuffled_objects_three_objects.json
    │   ├── web_of_lies.json
    │   └── word_sorting.json
    └── lib_prompt
        ├── boolean_expressions.txt
        ├── causal_judgement.txt
        ├── date_understanding.txt
        ├── disambiguation_qa.txt
        ├── dyck_languages.txt
        ├── formal_fallacies.txt
        ├── geometric_shapes.txt
        ├── hyperbaton.txt
        ├── logical_deduction_five_objects.txt
        ├── logical_deduction_seven_objects.txt
        ├── logical_deduction_three_objects.txt
        ├── movie_recommendation.txt
        ├── multistep_arithmetic_two.txt
        ├── navigate.txt
        ├── object_counting.txt
        ├── penguins_in_a_table.txt
        ├── reasoning_about_colored_objects.txt
        ├── ruin_names.txt
        ├── salient_translation_error_detection.txt
        ├── snarks.txt
        ├── sports_understanding.txt
        ├── temporal_sequences.txt
        ├── tracking_shuffled_objects_five_objects.txt
        ├── tracking_shuffled_objects_seven_objects.txt
        ├── tracking_shuffled_objects_three_objects.txt
        ├── web_of_lies.txt
        └── word_sorting.txt
    ```

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|bbh_gen_3_shot_cot_chat|BBH数据集生成式任务|score(accuracy)|3-shot|对话格式|[bbh_gen_3_shot_cot_chat.py](bbh_gen_3_shot_cot_chat.py)|
