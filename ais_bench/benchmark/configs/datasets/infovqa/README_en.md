# InfoVQA
English | [中文](README.md)
## Dataset Introduction
The InfoVQA dataset was launched in 2021 as the dedicated dataset for the third task of the DocVQA Challenge, with its core function being to test the model's visual question-answering ability for infographic images. It contains 5,485 diverse infographic images collected from the Internet, along with 30,035 manually annotated Q&A pairs. The problem design is highly targeted. It not only requires the model to conduct comprehensive reasoning by integrating document layout, text content, graphic elements and data visualization content, but also focuses on problems that need basic reasoning and simple arithmetic skills, such as counting and comparing operations based on data in charts.

> 🔗 Dataset Homepage [https://huggingface.co/datasets/WinKawaks/InfoVQA](https://huggingface.co/datasets/WinKawaks/InfoVQA)

## Dataset Deployment
- The accuracy evaluation of this dataset was aligned with OpenCompass's multimodal evaluation tool VLMEvalkit, and the dataset format was the tsv file provided by OpenCompass
- Dataset download：opencompass provided link🔗 [https://opencompass.openxlab.space/utils/VLMEval/InfoVQA_VAL.tsv](https://opencompass.openxlab.space/utils/VLMEval/InfoVQA_VAL.tsv).
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set for dataset tasks). Taking deployment on a Linux server as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
mkdir InfoVQA
cd InfoVQA
wget https://opencompass.openxlab.space/utils/VLMEval/InfoVQA_VAL.tsv
```
- Execute `tree InfoVQA/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure matches the one shown below, the dataset has been deployed successfully:
    ```
    InfoVQA
    └── InfoVQA_VAL.tsv
    ```

## Available Dataset Tasks
#### Basic Information
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
|infovqa_gen|infovqa dataset generative task|anls|0-shot|String format|[infovqa_gen.py](infovqa_gen.py)|
