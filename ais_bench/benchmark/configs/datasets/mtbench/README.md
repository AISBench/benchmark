# MTBench
中文 | [English](README_en.md)
## 数据集简介
MTBench数据集是一种多轮对话数据集，是覆盖写作、角色扮演、推理、数学、编码、信息抽取、STEM、和人文学科8个类别，每类10题，题目难度呈“专家级”；共有80个多轮对话数据，每条数据包含两轮对话，主要用于评估大模型的对话能力。
数据样例如下，`category`表示数据类别，`prompt`中包含两个问题，表示两轮对话，`reference`表示对应的参考答案，部分数据没有`reference`字段：
> 🔗 数据集主页链接[https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts](https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts)。
```
{"question_id": 111,
"category": "math",
"prompt": ["The vertices of a triangle are at points (0, 0), (-1, 1), and (3, 3). What is the area of the triangle?", "What's area of the circle circumscribing the triangle?"],
"reference": ["Area is 3", "5pi"]}
```

## 数据集部署

- question.jsonl数据中包含80组多轮对话，共计160轮，下载链接🔗 [https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts/blob/main/raw/question.jsonl](https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts/blob/main/raw/question.jsonl)
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
mkdir mtbench/
cd mtbench/
wget https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts/blob/main/raw/question.jsonl
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree mtbench/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    mtbench
    └── question.jsonl
    ```

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|配套文件导入方式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- | --- |
|mtbench_gen|mtbench生成式任务|暂不支持精度评测|0-shot|列表格式|`from ais_bench.benchmark.configs.datasets.mtbench.mtbench_gen import mtbench_datasets as datasets`|[mtbench_gen.py](mtbench_gen.py)|


*注意：该多轮对话数据集的测评支持vLLM、SGLang、MindIE Service等服务化，使用时需指定--models为vllm_api_stream_chat_multiturn*