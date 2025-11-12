# NeedleBench V2: An Improved "Needle in a Haystack" Test Evaluation Benchmark
[中文](README.md) | English
## Dataset Introduction
NeedleBench V2 is an enhanced benchmark designed to rigorously evaluate the information retrieval and reasoning capabilities of Large Language Models (LLMs) in long-text scenarios. Building on the original NeedleBench, this version introduces key enhancements to provide a more accurate and unbiased assessment of LLMs’ ability to locate and reason about critical information within massive text volumes.

NeedleBench V2 offers tasks with different length configurations (4k, 8k, 32k, 128k, 200k, 256k, 1000k) to accommodate the evaluation needs of LLMs of varying scales. Each length configuration includes dedicated test scripts for the following tasks:


### Single Needle Retrieval
The Single Needle Retrieval task assesses LLMs’ ability to recall a single piece of critical information from a text filled with irrelevant content of a specific length. This task evaluates the model’s precision in identifying and recalling specific information within long texts.


### Multiple Needle Retrieval
The Multiple Needle Retrieval task challenges LLMs to identify and extract multiple key information points from extensive text. It simulates real-world scenarios where multiple data points, facts, or numbers need to be retrieved from documents or reports, evaluating the model’s efficiency in navigating and extracting relevant information from dense text.


### Multiple Needle Reasoning
In NeedleBench V2, the Multiple Needle Reasoning task has been significantly improved. The original "needles" (critical information points) based on the R4C/MultiHop datasets have been replaced with fictional information similar to that used in the Ancestry Tracing Challenge (ATC). This change addresses potential **intrinsic knowledge bias**—a limitation where the original datasets might have been included in the training data of some models. The task still evaluates LLMs’ ability to perform complex reasoning using retrieved information, requiring models to not only recall multiple information points but also conduct logical reasoning.


### Ancestry Tracing Challenge (ATC)
The Ancestry Tracing Challenge (ATC) has been optimized in NeedleBench V2. The distribution pattern of "needles" has shifted from a dense format (1, 2, 3, 4, 5 needles) to a sparse format based on powers of 2 (2¹, 2², 2³, etc.). Remaining the most complex task in NeedleBench, ATC requires models to recall and analyze every detail in long texts to solve problems that demand an understanding of complex relationships—such as genealogical queries or detailed case analyses.


NeedleBench V2 introduces a more balanced scoring system. The overall score is now calculated as a simple average of three core tasks (Single Needle Retrieval, Multiple Needle Retrieval, and Multiple Needle Reasoning), with each task weighted equally. This change departs from the previous weighted average method, providing a more straightforward and fair way to evaluate models’ performance across different retrieval and reasoning tasks.

> 🔗 Dataset Homepage Link: [https://huggingface.co/datasets/opencompass/NeedleBench](https://huggingface.co/datasets/opencompass/NeedleBench)


## Dataset Deployment
It is recommended to download the dataset from Hugging Face: [https://huggingface.co/datasets/opencompass/NeedleBench](https://huggingface.co/datasets/opencompass/NeedleBench)
- Deployment is recommended in the directory `{tool_root_path}/ais_bench/datasets` (the default path set for dataset tasks).
- After successful deployment, execute `tree NeedleBench/` in the directory `{tool_root_path}/ais_bench/datasets` to verify the directory structure. The deployment is successful if the structure matches the following:
    ```
    NeedleBench/
    ├── gitattributes
    ├── multi_needle_reasoning_en.json
    ├── multi_needle_reasoning_zh.json
    ├── names.json
    ├── needles.jsonl
    ├── PaulGrahamEssays.jsonl
    ├── README.md
    ├── zh_finance.jsonl
    ├── zh_game.jsonl
    ├── zh_general.jsonl
    ├── zh_government.jsonl
    ├── zh_movie.jsonl
    └── zh_tech.jsonl
    ```


## Available Dataset Tasks

| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code File Path |
| --- | --- | --- | --- | --- | --- |
| atc_0shot_nocot_2_power_en | atc_0shot_nocot_2_power_en | Accuracy | 0-shot | Chat Format | [atc/atc_0shot_nocot_2_power_en.py]() |
| needlebench_v2_4k | needlebench_v2_4k | Accuracy | 0-shot | Chat Format | [needlebench_v2_4k.py](needlebench_v2_4k/needlebench_v2_4k.py) |
| needlebench_v2_multi_reasoning_4k | needlebench_v2_multi_reasoning_4k | Accuracy | 0-shot | Chat Format | [needlebench_v2_multi_reasoning_4k.py](needlebench_v2_4k/needlebench_v2_multi_reasoning_4k.py) |
| needlebench_v2_multi_retrieval_4k | needlebench_v2_multi_retrieval_4k | Accuracy | 0-shot | Chat Format | [needlebench_v2_multi_retrieval_4k.py](needlebench_v2_4k/needlebench_v2_multi_retrieval_4k.py) |
| needlebench_v2_single_4k | needlebench_v2_single_4k | Accuracy | 0-shot | Chat Format | [needlebench_v2_single_4k.py](needlebench_v2_4k/needlebench_v2_single_4k.py) |
| needlebench_v2_8k | needlebench_v2_8k | Accuracy | 0-shot | Chat Format | [needlebench_v2_8k.py](needlebench_v2_8k/needlebench_v2_8k.py) |
| needlebench_v2_multi_reasoning_8k | needlebench_v2_multi_reasoning_8k | Accuracy | 0-shot | Chat Format | [needlebench_v2_multi_reasoning_8k.py](needlebench_v2_8k/needlebench_v2_multi_reasoning_8k.py) |
| needlebench_v2_multi_retrieval_8k | needlebench_v2_multi_retrieval_8k | Accuracy | 0-shot | Chat Format | [needlebench_v2_multi_retrieval_8k.py](needlebench_v2_8k/needlebench_v2_multi_retrieval_8k.py) |
| needlebench_v2_single_8k | needlebench_v2_single_8k | Accuracy | 0-shot | Chat Format | [needlebench_v2_single_8k.py](needlebench_v2_8k/needlebench_v2_single_8k.py) |
| needlebench_v2_multi_retrieval_compare_batch_8k | needlebench_v2_multi_retrieval_compare_batch_8k | Accuracy | 0-shot | Chat Format | [needlebench_v2_multi_retrieval_compare_batch_8k.py](needlebench_v2_8k/needlebench_v2_multi_retrieval_compare_batch_8k.py) |
| needlebench_v2_32k | needlebench_v2_32k | Accuracy | 0-shot | Chat Format | [needlebench_v2_32k.py](needlebench_v2_32k/needlebench_v2_32k.py) |
| needlebench_v2_multi_reasoning_32k | needlebench_v2_multi_reasoning_32k | Accuracy | 0-shot | Chat Format | [needlebench_v2_multi_reasoning_32k.py](needlebench_v2_32k/needlebench_v2_multi_reasoning_32k.py) |
| needlebench_v2_multi_retrieval_32k | needlebench_v2_multi_retrieval_32k | Accuracy | 0-shot | Chat Format | [needlebench_v2_multi_retrieval_32k.py](needlebench_v2_32k/needlebench_v2_multi_retrieval_32k.py) |
| needlebench_v2_single_32k | needlebench_v2_single_32k | Accuracy | 0-shot | Chat Format | [needlebench_v2_single_32k.py](needlebench_v2_32k/needlebench_v2_single_32k.py) |
| needlebench_v2_128k | needlebench_v2_128k | Accuracy | 0-shot | Chat Format | [needlebench_v2_128k.py](needlebench_v2_128k/needlebench_v2_128k.py) |
| needlebench_v2_multi_reasoning_128k | needlebench_v2_multi_reasoning_128k | Accuracy | 0-shot | Chat Format | [needlebench_v2_multi_reasoning_128k.py](needlebench_v2_128k/needlebench_v2_multi_reasoning_128k.py) |
| needlebench_v2_multi_retrieval_128k | needlebench_v2_multi_retrieval_128k | Accuracy | 0-shot | Chat Format | [needlebench_v2_multi_retrieval_128k.py](needlebench_v2_128k/needlebench_v2_multi_retrieval_128k.py) |
| needlebench_v2_single_128k | needlebench_v2_single_128k | Accuracy | 0-shot | Chat Format | [needlebench_v2_single_128k.py](needlebench_v2_128k/needlebench_v2_single_128k.py) |
| needlebench_v2_200k | needlebench_v2_200k | Accuracy | 0-shot | Chat Format | [needlebench_v2_200k.py](needlebench_v2_200k/needlebench_v2_200k.py) |
| needlebench_v2_multi_reasoning_200k | needlebench_v2_multi_reasoning_200k | Accuracy | 0-shot | Chat Format | [needlebench_v2_multi_reasoning_200k.py](needlebench_v2_200k/needlebench_v2_multi_reasoning_200k.py) |
| needlebench_v2_multi_retrieval_200k | needlebench_v2_multi_retrieval_200k | Accuracy | 0-shot | Chat Format | [needlebench_v2_multi_retrieval_200k.py](needlebench_v2_200k/needlebench_v2_multi_retrieval_200k.py) |
| needlebench_v2_single_200k | needlebench_v2_single_200k | Accuracy | 0-shot | Chat Format | [needlebench_v2_single_200k.py](needlebench_v2_200k/needlebench_v2_single_200k.py) |
| needlebench_v2_256k | needlebench_v2_256k | Accuracy | 0-shot | Chat Format | [needlebench_v2_256k.py](needlebench_v2_256k/needlebench_v2_256k.py) |
| needlebench_v2_multi_reasoning_256k | needlebench_v2_multi_reasoning_256k | Accuracy | 0-shot | Chat Format | [needlebench_v2_multi_reasoning_256k.py](needlebench_v2_256k/needlebench_v2_multi_reasoning_256k.py) |
| needlebench_v2_multi_retrieval_256k | needlebench_v2_multi_retrieval_256k | Accuracy | 0-shot | Chat Format | [needlebench_v2_multi_retrieval_256k.py](needlebench_v2_256k/needlebench_v2_multi_retrieval_256k.py) |
| needlebench_v2_single_256k | needlebench_v2_single_256k | Accuracy | 0-shot | Chat Format | [needlebench_v2_single_256k.py](needlebench_v2_256k/needlebench_v2_single_256k.py) |
| needlebench_v2_1000k | needlebench_v2_1000k | Accuracy | 0-shot | Chat Format | [needlebench_v2_1000k.py](needlebench_v2_1000k/needlebench_v2_1000k.py) |
| needlebench_v2_multi_reasoning_1000k | needlebench_v2_multi_reasoning_1000k | Accuracy | 0-shot | Chat Format | [needlebench_v2_multi_reasoning_1000k.py](needlebench_v2_1000k/needlebench_v2_multi_reasoning_1000k.py) |
| needlebench_v2_multi_retrieval_1000k | needlebench_v2_multi_retrieval_1000k | Accuracy | 0-shot | Chat Format | [needlebench_v2_multi_retrieval_1000k.py](needlebench_v2_1000k/needlebench_v2_multi_retrieval_1000k.py) |
| needlebench_v2_single_1000k | needlebench_v2_single_1000k | Accuracy | 0-shot | Chat Format | [needlebench_v2_single_1000k.py](needlebench_v2_1000k/needlebench_v2_single_1000k.py) |