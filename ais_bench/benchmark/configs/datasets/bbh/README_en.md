# BBH
[中文](README.md) | English
## Dataset Introduction
BIG-Bench (Srivastava et al., 2022) is a diverse evaluation test suite that focuses on tasks deemed currently beyond the capabilities of language models. While language models have made significant progress on this benchmark—with the best model in the BIG-Bench paper surpassing the average performance of human evaluators on 65% of tasks through few-shot prompting—two key questions remain: On which specific tasks do language models still lag behind the human average? And are these tasks truly beyond the problem-solving capabilities of current language models?

> 🔗 Dataset Homepage Link: [https://huggingface.co/datasets/lukaemon/bbh](https://huggingface.co/datasets/lukaemon/bbh)

## Dataset Deployment
- The dataset compressed package can be downloaded from the link provided by OpenCompass 🔗: [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/BBH.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/BBH.zip).
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set in dataset tasks). Taking deployment on Linux as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/BBH.zip
unzip BBH.zip
rm BBH.zip
```
- Execute `tree BBH/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    BBH
    ├── data
    │   ├── boolean_expressions.json
    │   ├── causal_judgement.json
    │   ├── date_understanding.json
    │   ├── disambiguation_qa.json
    │   ├── dyck_languages.json
    │   ├── formal_fallacies.json
    │   ├── geometric_shapes.json
    │   ├── hyperbaton.json
    │   ├── logical_deduction_five_objects.json
    │   ├── logical_deduction_seven_objects.json
    │   ├── logical_deduction_three_objects.json
    │   ├── movie_recommendation.json
    │   ├── multistep_arithmetic_two.json
    │   ├── navigate.json
    │   ├── object_counting.json
    │   ├── penguins_in_a_table.json
    │   ├── README.md
    │   ├── reasoning_about_colored_objects.json
    │   ├── ruin_names.json
    │   ├── salient_translation_error_detection.json
    │   ├── snarks.json
    │   ├── sports_understanding.json
    │   ├── temporal_sequences.json
    │   ├── tracking_shuffled_objects_five_objects.json
    │   ├── tracking_shuffled_objects_seven_objects.json
    │   ├── tracking_shuffled_objects_three_objects.json
    │   ├── web_of_lies.json
    │   └── word_sorting.json
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

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Import Statement | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- | --- |
| bbh_gen_3_shot_cot_chat | Generative task for the BBH dataset | Score (Accuracy) | 3-shot | Chat format |`from ais_bench.benchmark.configs.datasets.bbh.bbh_gen_3_shot_cot_chat import bbh_datasets as datasets`| [bbh_gen_3_shot_cot_chat.py](bbh_gen_3_shot_cot_chat.py) |