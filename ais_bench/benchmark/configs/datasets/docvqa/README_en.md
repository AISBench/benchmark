# DocVQA
English | [中文](README.md)
## Dataset Introduction
DocVQA is a classic dataset for visual question answering of document images, launched in 2021, aiming to drive the field of document analysis and recognition towards a "goal-driven" direction. This dataset contains over 12,000 document images, corresponding to 50,000 questions. The document images cover various types such as letters, integrating different styles of text like print and handwriting, as well as visual elements like tables, checkboxes, and separators. Unlike ordinary visual question answering tasks, it requires the model not only to extract the text content from the document but also to interpret visual cues such as the document layout and font style to answer questions. For instance, when answering questions like the company address mentioned in the letter, it is necessary to make a comprehensive judgment by combining the text and document structure.

> 🔗 Dataset Homepage [https://huggingface.co/datasets/lmms-lab/DocVQA](https://huggingface.co/datasets/lmms-lab/DocVQA)

## Dataset Deployment
- The accuracy evaluation of this dataset was aligned with OpenCompass's multimodal evaluation tool VLMEvalkit, and the dataset format was the tsv file provided by OpenCompass
- Dataset download：opencompass provided link🔗 [https://opencompass.openxlab.space/utils/VLMEval/DocVQA_VAL.tsv](https://opencompass.openxlab.space/utils/VLMEval/DocVQA_VAL.tsv).
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set for dataset tasks). Taking deployment on a Linux server as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
mkdir docvqa
cd docvqa
wget https://opencompass.openxlab.space/utils/VLMEval/DocVQA_VAL.tsv
```
- Execute `tree docvqa/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure matches the one shown below, the dataset has been deployed successfully:
    ```
    docvqa
    └── DocVQA_VAL.tsv
    ```

## Available Dataset Tasks
#### Basic Information
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
|docvqa_gen|docvqa dataset generative task|anls|0-shot|String format|[docvqa_gen.py](docvqa_gen.py)|
