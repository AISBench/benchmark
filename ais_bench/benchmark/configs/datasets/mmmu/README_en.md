# MMMU
English | [中文](README.md)
## Dataset Introduction
MMMU is a cross-disciplinary graphic reasoning evaluation set for university level, obtained from teaching diagrams (charts, musical scores, chemical structures, etc.), covering six major fields including art, business, science and engineering, medicine, humanities, and engineering. It is used to measure the comprehensive understanding and reasoning ability of multimodal models in complex semantics and visual symbols.

> 🔗 Dataset Homepage [https://huggingface.co/datasets/MMMU/MMMU](https://huggingface.co/datasets/MMMU/MMMU)

## Dataset Deployment
- The accuracy evaluation of this dataset was aligned with OpenCompass's multimodal evaluation tool VLMEvalkit, and the dataset format was the tsv file provided by OpenCompass
- Dataset download：opencompass provided link🔗VAL [https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv](https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv)🔗 TEST[https://opencompass.openxlab.space/utils/VLMEval/MMMU_TEST.tsv](https://opencompass.openxlab.space/utils/VLMEval/MMMU_TEST.tsv)。
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set for dataset tasks). Taking deployment on a Linux server as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
mkdir mmmu
cd mmmu
wget https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv
```
- Execute `tree mmmu/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure matches the one shown below, the dataset has been deployed successfully:
    ```
    mmmu
    └── MMMU_DEV_VAL.tsv
    ```

## Available Dataset Tasks
### mmmu_gen
#### Basic Information
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
|mmmu_gen|Generative task for the mmmu dataset|acc|0-shot|String format|[mmmu_gen.py](mmmu_gen.py)|
|mmmu_gen_cot|COT Generative task for the mmmu dataset|acc|0-shot|String format|[mmmu_gen_cot.py](mmmu_gen_cot.py)|