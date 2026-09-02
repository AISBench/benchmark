# GPQA
[中文](README.md) | English
## Dataset Introduction
GPQA is a question-answering dataset consisting of multiple-choice questions. The high-difficulty questions within it are written and verified by experts in the fields of biology, physics, and chemistry. When these experts attempt to answer questions outside their own professional domains (for example, a physicist answering chemistry questions), their answer accuracy is only 34%—even if they can use Google Search without restrictions and spend more than 30 minutes on each question.

> 🔗 Dataset Homepage: [https://github.com/idavidrein/gpqa](https://github.com/idavidrein/gpqa)

## Dataset Deployment
- The dataset compressed package can be downloaded from the link provided by OpenCompass 🔗: [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gpqa.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gpqa.zip).
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set in dataset tasks). Taking deployment on Linux as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gpqa.zip
unzip gpqa.zip
rm gpqa.zip
```
- Execute `tree gpqa/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    gpqa
    ├── gpqa_diamond.csv
    ├── gpqa_experts.csv
    ├── gpqa_extended.csv
    ├── gpqa_main.csv
    └── license.txt
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Import Statement | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- | --- |
| gpqa_gen_0_shot_str | Generative task for the GPQA dataset | Accuracy (pass@1) | 0-shot | String format |`from ais_bench.benchmark.configs.datasets.gpqa.gpqa_gen_0_shot_str import gpqa_datasets as datasets`| [gpqa_gen_0_shot_str.py](gpqa_gen_0_shot_str.py) |
| gpqa_gen_0_shot_cot_chat_prompt | Generative task for the GPQA dataset (aligned with DeepSeek R1 accuracy test) | Accuracy (pass@1) | 0-shot | Chat format |`from ais_bench.benchmark.configs.datasets.gpqa.gpqa_gen_0_shot_cot_chat_prompt import gpqa_datasets as datasets`| [gpqa_gen_0_shot_cot_chat_prompt.py](gpqa_gen_0_shot_cot_chat_prompt.py) |
| gpqa_ppl_0_shot_str | PPL task for the GPQA dataset | Accuracy (pass@1) | 0-shot | String format |`from ais_bench.benchmark.configs.datasets.gpqa.gpqa_ppl_0_shot_str import gpqa_datasets as datasets`| [gpqa_ppl_0_shot_str.py](gpqa_ppl_0_shot_str.py) |

### Translation Notes
1. **Term Consistency**: Technical terms such as "生成式任务" (generative task), "评估指标" (evaluation metric), and "对齐DeepSeek R1精度测试" (aligned with DeepSeek R1 accuracy test) follow standard expressions in AI dataset documentation to ensure clarity for technical users.
2. **Proper Nouns & Acronyms**: "GPQA" (the dataset name) is retained as is; "OpenCompass" (platform name) and "DeepSeek R1" (model/test standard name) remain unchanged to maintain recognition in the technical community.
3. **Code & Path Preservation**: Linux commands (e.g., `cd`, `wget`), directory paths (e.g., `{tool_root_path}/ais_bench/datasets`), and filenames (e.g., `gpqa_diamond.csv`, `gpqa_gen_0_shot_str.py`) are copied exactly to avoid disrupting deployment workflows.
4. **Contextual Accuracy**: The description of experts’ limited cross-domain performance ("即便他们能无限制地使用谷歌搜索，且花费超过 30 分钟作答，答题准确率也仅有 34%") is translated to preserve logical relationships (using "even if" to convey the contrast) and key details (time limit, tool access, accuracy rate), ensuring the dataset’s difficulty characteristics are accurately communicated.