# MTBench
[中文](README.md) | English
## Dataset Introduction
The MTBench dataset is a multi-turn conversation dataset covering 8 categories: Writing, Role-Playing, Reasoning, Mathematics, Coding, Information Extraction, STEM, and Humanities. Each category contains 10 questions with "expert-level" difficulty. In total, there are 80 multi-turn conversation samples, and each sample includes two turns of dialogue. This dataset is mainly used to evaluate the conversation capabilities of large language models (LLMs).

Below is a sample of the data, where `category` indicates the data category, `turns` (represented by the "prompt" field in the sample) denotes the two turns of questions, and `reference` represents the corresponding reference answers. Some data samples do not have the `reference` field:
> 🔗 Dataset Homepage Link: [https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts](https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts)
```
{"question_id": 111,
"category": "math",
"prompt": ["The vertices of a triangle are at points (0, 0), (-1, 1), and (3, 3). What is the area of the triangle?", "What's area of the circle circumscribing the triangle?"],
"reference": ["Area is 3", "5pi"]}
```

## Dataset Deployment
- The `question.jsonl` file contains 80 multi-turn conversation groups (160 turns in total). Download link 🔗: [https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts/blob/main/raw/question.jsonl](https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts/blob/main/raw/question.jsonl)
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set for dataset tasks). Taking deployment on a Linux server as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
mkdir mtbench/
cd mtbench/
wget https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts/blob/main/raw/question.jsonl
```
- Execute `tree mtbench/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure matches the one shown below, the dataset has been deployed successfully:
    ```
    mtbench
    └── question.jsonl
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Import Statement | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- | --- |
| mtbench_gen | Generative task for MTBench | Accuracy evaluation not supported temporarily | 0-shot | List format |`from ais_bench.benchmark.configs.datasets.mtbench.mtbench_gen import mtbench_datasets as datasets`| [mtbench_gen.py](mtbench_gen.py) |


*Note: The evaluation of this multi-turn conversation dataset supports service deployment frameworks such as vLLM, SGLang, and MindIE Service. When using it, you need to specify `--models` as `vllm_api_stream_chat_multiturn`.*


### Translation Notes
1. **Category & Term Accuracy**: The 8 dataset categories ("写作、角色扮演、推理、数学、编码、信息抽取、STEM、和人文学科") are translated into standard, industry-recognized terms (Writing, Role-Playing, Reasoning, Mathematics, Coding, Information Extraction, STEM, Humanities) to ensure consistency with global AI evaluation benchmarks. "专家级难度" is rendered as "expert-level difficulty" to accurately reflect the question complexity.
2. **Data Field Clarity**: The data field descriptions are supplemented for clarity: since the original text mentions "turns表示两轮问题" but the sample uses the "prompt" field, the translation adds a note ("represented by the 'prompt' field in the sample") to avoid confusion. "参考答案" is translated as "reference answers" (a common term in benchmark datasets for standard answers).
3. **Technical Consistency**: Terms like "多轮对话" (multi-turn conversation), "生成式任务" (generative task), and "服务化" (service deployment) align with consistent translations of previous AI dataset documents. Framework names (vLLM, SGLang, MindIE Service) and command parameters (`--models`, `vllm_api_stream_chat_multiturn`) are retained unchanged to ensure usability for technical users.
4. **Instruction Precision**: The note about accuracy evaluation ("暂不支持精度评测") is translated as "Accuracy evaluation not supported temporarily" to clearly convey the current limitation, while the deployment command steps and directory structure are kept exactly as original to maintain operational integrity.