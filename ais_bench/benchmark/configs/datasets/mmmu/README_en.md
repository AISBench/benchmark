# MMMU
English | [中文](README.md)
## Dataset Introduction
MMMU is a cross-disciplinary graphic reasoning evaluation set for university level, obtained from teaching diagrams (charts, musical scores, chemical structures, etc.), covering six major fields including art, business, science and engineering, medicine, humanities, and engineering. It is used to measure the comprehensive understanding and reasoning ability of multimodal models in complex semantics and visual symbols.

> 🔗 Dataset Homepage [https://modelscope.cn/datasets/AI-ModelScope/MMMU/summary](https://modelscope.cn/datasets/AI-ModelScope/MMMU/summary)

## Dataset Deployment
- This implementation is aligned with evalscope. The dataset source is the ModelScope dataset `AI-ModelScope/MMMU`, and the default evaluation split is `validation`.
- [mmmu_gen.py](mmmu_gen.py) reads local parquet data files from `{tool_root_path}/ais_bench/datasets/mmmu` by default.
- The default prompts are exposed as `MULT_CHOICE_PROMPT` and `OPEN_PROMPT` in [mmmu_gen.py](mmmu_gen.py), so users can customize the multiple-choice and open-question prompts directly in the config file.
- It is recommended to deploy the dataset under `{tool_root_path}/ais_bench/datasets` (the default path used by dataset tasks). Taking deployment on a Linux server as an example:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
git lfs install
git clone https://www.modelscope.cn/datasets/AI-ModelScope/MMMU.git mmmu
```
- Execute `tree mmmu/` under `{tool_root_path}/ais_bench/datasets` to check the directory structure. If subject directories or parquet data files are present, the dataset has been deployed successfully.
    ```
    mmmu
    ├── Accounting
    │   └── validation-*.parquet
    ├── Agriculture
    │   └── validation-*.parquet
    └── ...
    ```

## Available Dataset Tasks
### mmmu_gen
#### Basic Information
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
|mmmu_gen|Generative MMMU task: multiple-choice questions use the CoT single-answer template, while open questions use the `ANSWER: [ANSWER]` template|acc|0-shot|Multimodal chat format|[mmmu_gen.py](mmmu_gen.py)|
