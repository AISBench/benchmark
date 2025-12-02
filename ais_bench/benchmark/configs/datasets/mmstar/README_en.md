# MMStar
English | [中文](README.md)
## Dataset Introduction
MMStar is an "elite-level" visual-language assessment set, consisting of 1,500 English multiple-choice questions. All of them have undergone manual review to ensure that each question can only be answered by looking at the pictures, and the risk of training data leakage is extremely low. It also evenly covers six core capabilities (coarse-grained perception, fine-grained perception, case-based reasoning, logical reasoning, mathematics, and technology) and 18 fine-grained dimensions, which are used to strictly test the true multimodal comprehension ability of large models.

> 🔗 Dataset Homepage [https://huggingface.co/datasets/Lin-Chen/MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar)

## Dataset Deployment
- The accuracy evaluation of this dataset was aligned with OpenCompass's multimodal evaluation tool VLMEvalkit, and the dataset format was the tsv file provided by OpenCompass
- Dataset download：opencompass provided link🔗[https://opencompass.openxlab.space/utils/VLMEval/MMStar.tsv](https://opencompass.openxlab.space/utils/VLMEval/MMStar.tsv)。
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (this is the default path for dataset tasks). For deployment on a Linux server, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
mkdir mmstar
cd mmstar
wget https://opencompass.openxlab.space/utils/VLMEval/MMStar.tsv
```
- Execute `tree mmstar/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure matches the one shown below, the dataset has been deployed successfully:
    ```
    mmstar
    └── MMStar.tsv
    ```

## Available Dataset Tasks
### mmstar_gen
#### Basic Information
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
|mmstar_gen|Generative task for the mmstar dataset|acc|0-shot|String format|[mmstar_gen.py](mmstar_gen.py)|
