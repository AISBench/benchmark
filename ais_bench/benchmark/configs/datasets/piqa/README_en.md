# piqa
[中文](README.md) | English
## Dataset Introduction
The PIQA dataset proposes a physical commonsense reasoning task and constructs a corresponding benchmark dataset—Physical Interaction: Question Answering (PIQA for short, Physical Interaction Question Answering).

Physical commonsense is a major challenge on the path to achieving true AI completeness (including robots that can interact with the world and understand natural language).

> 🔗 Dataset Homepage: [https://huggingface.co/datasets/ybisk/piqa](https://huggingface.co/datasets/ybisk/piqa)

## Dataset Deployment
- The dataset compressed package can be downloaded from the link 🔗: [https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip](https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip)
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set for dataset tasks). Taking deployment on a Linux server as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip
unzip physicaliqa-train-dev.zip
rm physicaliqa-train-dev.zip
```
- Execute `tree physicaliqa-train-dev/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure matches the one shown below, the dataset has been deployed successfully:
    ```
    physicaliqa-train-dev
    ├── dev.jsonl
    ├── dev-labels.lst
    ├── train.jsonl
    └── train-labels.lst
    ```

## Available Dataset Tasks
### piqa_gen_0_shot_chat_prompt
#### Basic Information
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
| piqa_gen_0_shot_chat_prompt | Generative task for the piqa dataset | Accuracy | 0-shot | Chat Format | [piqa_gen_0_shot_chat_prompt.py](piqa_gen_0_shot_chat_prompt.py) |
| piqa_gen_0_shot_str | Generative task for the piqa dataset | Accuracy | 0-shot | String Format | [piqa_gen_0_shot_str.py](piqa_gen_0_shot_str.py) |
| piqa_ppl_0_shot_str | PPL task for the piqa dataset | Accuracy | 0-shot | String Format | [piqa_ppl_0_shot_str.py](piqa_ppl_0_shot_str.py) |