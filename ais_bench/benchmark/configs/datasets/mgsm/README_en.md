# mgsm
[中文](README.md) | English
## Dataset Introduction
The Multilingual Grade School Math benchmark (MGSM) is an evaluation benchmark focused on elementary school mathematics problems.

> 🔗 Dataset Homepage Link: [https://huggingface.co/datasets/juletxara/mgsm](https://huggingface.co/datasets/juletxara/mgsm)

## Dataset Deployment
- The dataset can be obtained from the Hugging Face dataset link 🔗: [https://huggingface.co/datasets/juletxara/mgsm](https://huggingface.co/datasets/juletxara/mgsm)
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set for dataset tasks). Taking deployment on a Linux server as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
git lfs install
git clone https://huggingface.co/datasets/juletxara/mgsm
```
- Execute `tree mgsm/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure matches the one shown below, the dataset has been deployed successfully:
    ```
    mgsm/
    ├── exemplars.py
    ├── mgsm_bn.tsv
    ├── mgsm_de.tsv
    ├── mgsm_en.tsv
    ├── mgsm_es.tsv
    ├── mgsm_fr.tsv
    ├── mgsm_ja.tsv
    ├── mgsm.py
    ├── mgsm_ru.tsv
    ├── mgsm_sw.tsv
    ├── mgsm_te.tsv
    ├── mgsm_th.tsv
    ├── mgsm_zh.tsv
    └── README.md
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
| mgsm_gen_0_shot_cot_chat_prompt | Generative task for the mgsm dataset, with a logical chain in the prompt | Accuracy | 0-shot | Chat format | [mgsm_gen_0_shot_cot_chat_prompt.py](mgsm_gen_0_shot_cot_chat_prompt.py) |
| mgsm_gen_8_shot_cot_chat_prompt | Generative task for the mgsm dataset, with a logical chain in the prompt | Accuracy | 8-shot | Chat format | [mgsm_gen_8_shot_cot_chat_prompt.py](mgsm_gen_8_shot_cot_chat_prompt.py) |