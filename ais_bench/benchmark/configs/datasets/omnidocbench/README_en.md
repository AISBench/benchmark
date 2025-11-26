# OmniDocBench
[中文](README.md) | English
## Dataset Introduction
OmniDocBench is a benchmark for evaluating diverse document parsing in real-world scenarios, featuring the following characteristics:

- Diverse Document Types: This benchmark includes 1355 PDF pages, covering 9 document types, 4 layout types, and 3 language types. It encompasses a wide range of content, including academic papers, financial reports, newspapers, textbooks, and handwritten notes.
- Rich Annotation Information: It contains localization information for 15 block-level (such as text paragraphs, headings, tables, etc., totaling over 20k) and 4 span-level (such as text lines, inline formulas, subscripts, etc., totaling over 80k) document elements. Each element's region includes recognition results (text annotations, LaTeX annotations for formulas, and both LaTeX and HTML annotations for tables). OmniDocBench also provides annotations for the reading order of document components. Additionally, it includes various attribute tags at the page and block levels, with annotations for 5 page attribute tags, 3 text attribute tags, and 6 table attribute tags.
- High Annotation Quality: The data quality is high, achieved through manual screening, intelligent annotation, manual annotation, and comprehensive expert and large model quality checks.
- Supporting Evaluation Code: It includes end-to-end and single-module evaluation code to ensure fairness and accuracy in assessments.

> 🔗 Dataset Homepage Link: [https://huggingface.co/datasets/opendatalab/OmniDocBench](https://huggingface.co/datasets/opendatalab/OmniDocBench)

## Dataset Deployment
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set in dataset tasks). Taking deployment on Linux as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
git clone https://huggingface.co/datasets/opendatalab/OmniDocBench
```
- Execute `tree OmniDocBench/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    OmniDocBench
    ├── images
    │   ├── PPT_1001115_eng_page_003.png
    │   └── PPT_1001115_eng_page_005.png
    │   # ......
    |
    └── OmniDocBench.json
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
| omnidocbench_gen | Generative task for the OmniDocBench dataset | accuracy (pass@1) | 0-shot | String format | [omnidocbench_gen.py](omnidocbench_gen.py) |

## Usage Constraints:
- Currently, only the Edit_dist metric is supported (used to evaluate the DeepSeek-OCR model); other metrics are not supported yet. The "overall" score is the average of the Edit_dist scores across all dimensions.