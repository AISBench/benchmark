# OCRBench_v2
[中文](README.md) | English

## Dataset Introduction
OCRBench_v2 is a large-scale bilingual text-centric benchmark dataset designed to evaluate the performance of Large Multimodal Models (LMMs) on complex OCR tasks. The dataset contains 10,000 manually verified question-answer pairs, covering 31 different scenarios (such as street scenes, receipts, formulas, charts, etc.), and comprehensively evaluates model capabilities in text recognition, text localization, handwritten content extraction, and logical reasoning through 23 tasks.

The dataset was created by manually selecting data from 81 academic datasets and supplementing with private data to ensure scenario diversity, aiming to address the limitations of existing benchmarks in terms of task diversity, contextual complexity, and scale.

> 🔗 Dataset Homepage Link: [https://arxiv.org/abs/2501.00321](https://arxiv.org/abs/2501.00321)

## Dataset Deployment
- You can download the dataset file from the HuggingFace link 🔗: [https://huggingface.co/datasets/QYWH/ocrbench_v2/resolve/main/OCRBench_v2.tsv?download=true](https://huggingface.co/datasets/QYWH/ocrbench_v2/resolve/main/OCRBench_v2.tsv?download=true)
- The dataset file should be in TSV format (`.tsv` file), with the default file name being `OCRBench_v2.tsv`
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets/ocrbench_v2/` (the default path set for dataset tasks). Taking deployment on a Linux server as an example, the specific execution steps are as follows:

```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
mkdir -p ocrbench_v2
cd ocrbench_v2
wget https://huggingface.co/datasets/QYWH/ocrbench_v2/resolve/main/OCRBench_v2.tsv?download=true -O OCRBench_v2.tsv
```

- Execute `tree ocrbench_v2/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure matches the one shown below, the dataset has been deployed successfully:
    ```
    ocrbench_v2/
    └── OCRBench_v2.tsv
    ```

⏰ **Note**: Please install dependencies from [ocrbench_v2.txt](../../../../../requirements/datasets/ocrbench_v2.txt) before running the dataset.
```shell
# You need to be in the outermost "benchmark" folder and run the following command:
pip3 install -r requirements/datasets/ocrbench_v2.txt
```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
| ocrbench_v2_gen_0_shot_chat | Generative task for OCRBench_v2 dataset, supporting multimodal input (image + text) | Multiple metrics (depending on task type) | 0-shot | Chat format (multimodal) | [ocrbench_v2_gen_0_shot_chat.py](ocrbench_v2_gen_0_shot_chat.py) |

## Supported Task Types
The OCRBench_v2 dataset covers the following task types:

### English Tasks
- **Text Recognition**: Text recognition, fine-grained text recognition, full-page OCR
- **Text Detection**: Text grounding, VQA with position
- **Text Spotting**: Text spotting
- **Relationship Extraction**: Key information extraction, key information mapping
- **Element Parsing**: Document parsing, chart parsing, table parsing, formula recognition
- **Mathematical Calculation**: Math QA, text counting
- **Visual Text Understanding**: Document classification, cognition VQA, diagram QA
- **Knowledge Reasoning**: Reasoning VQA, science QA, APP agent, ASCII art classification

### Chinese Tasks
- **Text Recognition**: Full-page OCR
- **Relationship Extraction**: Key information extraction, handwritten answer extraction
- **Element Parsing**: Document parsing, chart parsing, table parsing, formula recognition
- **Visual Text Understanding**: Document classification, cognition VQA, diagram QA
- **Knowledge Reasoning**: Reasoning VQA, science QA, APP agent

