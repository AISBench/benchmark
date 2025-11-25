# MMMU_Pro
English | [中文](README.md)
## Dataset Introduction
MMLU-Pro is an upgraded benchmark dataset jointly launched by researchers from the University of Waterloo and other institutions, as well as the MMMU team, in 2024. It combines multi-disciplinary text understanding and multi-modal reasoning capabilities. It is divided into a version that focuses on text reasoning (including 12K interdisciplinary complex problems) There are two types: the merged version (with 14 major subject categories) and the multimodal version (containing 3,460 multimodal questions, covering six core disciplines). The former integrates high-quality original MMLU questions, STEM website questions and other multi-source content and eliminates trivial questions, while the latter filters out questions that can be answered by the plain text model to ensure multimodal dependence. All are used for more rigorous assessment of the capabilities of AI models.
- options10 is an upgrade compared to the original MMLU's four options, expanding the number of candidate options for each question to ten. This change significantly reduces the probability of the model answering questions correctly through random guessing, forces the model to conduct deeper reasoning, and lowers the sensitivity of the model's score to changes in prompts to 2%. Significantly enhanced the robustness of benchmark tests.
- vision data is the core multimodal setup of this dataset. This type of data embeds questions into images such as screenshots or photos, forming test samples with only visual input. In the multimodal version, there are 1,730 such visual samples and 1,730 standard format samples. It requires the model to extract text and visual information from the images and fuse and process them to answer questions. This test model's ability to seamlessly integrate visual and text information is more in line with real-world application scenarios.

> 🔗 Dataset Homepage [https://huggingface.co/datasets/MMMU/MMMU_Pro](https://huggingface.co/datasets/MMMU/MMMU_Pro)

## Dataset Deployment
- The accuracy evaluation of this dataset was aligned with OpenCompass's multimodal evaluation tool VLMEvalkit, and the dataset format was the tsv file provided by OpenCompass
- Dataset download：opencompass provided link🔗options10 [https://opencompass.openxlab.space/utils/VLMEval/MMMU_Pro_10c.tsv](https://opencompass.openxlab.space/utils/VLMEval/MMMU_Pro_10c.tsv)🔗 vision [https://opencompass.openxlab.space/utils/VLMEval/MMMU_Pro_V.tsv](https://opencompass.openxlab.space/utils/VLMEval/MMMU_Pro_V.tsv)。
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set for dataset tasks). Taking deployment on a Linux server as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
mkdir mmmu_pro
cd mmmu_pro
wget https://opencompass.openxlab.space/utils/VLMEval/MMMU_Pro_10c.tsv
wget https://opencompass.openxlab.space/utils/VLMEval/MMMU_Pro_V.tsv
```
- Execute `tree mmmu_pro/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure matches the one shown below, the dataset has been deployed successfully:
    ```
    mmmu_pro
    ├── MMMU_Pro_10c.tsv
    └── MMMU_Pro_V.tsv
    ```

## Available Dataset Tasks
#### Basic Information
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
|mmmu_pro_options10_cot_gen|mmmu_pro options10 dataset thinking chain generative task|acc|0-shot|String format|[mmmu_pro_options10_cot_gen.py](mmmu_pro_options10_cot_gen.py)|
|mmmu_pro_options10_gen|mmmu_pro options10 dataset generative task|acc|0-shot|String format|[mmmu_pro_options10_gen.py](mmmu_pro_options10_gen.py)|
|mmmu_pro_vision_cot_gen|mmmu_pro vision dataset thinking chain generative task|acc|0-shot|String format|[mmmu_pro_vision_cot_gen.py](mmmu_pro_vision_cot_gen.py)|
|mmmu_pro_vision_gen|mmmu_pro vision dataset generative task|acc|0-shot|String format|[mmmu_pro_vision_gen.py](mmmu_pro_vision_gen.py)|
