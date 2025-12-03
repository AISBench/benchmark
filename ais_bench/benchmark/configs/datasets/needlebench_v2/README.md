# NeedleBench V2：改进版大海捞针测试评估基准
中文 | [English](README_en.md)
## 数据集简介

NeedleBench V2是一个改进版基准测试，旨在严格评估大型语言模型（LLMs）在长文本场景中的信息检索和推理能力。在原有NeedleBench的基础上，这个版本引入了重要的增强功能，为LLMs在海量文本中定位和推理关键信息的能力提供更准确、更公正的评估。

NeedleBench V2提供了不同长度配置的任务（4k、8k、32k、128k、200k、256k、1000k），以适应不同规模的语言模型评估需求。每种长度配置针对以下任务提供了专门的测试脚本：

### 单针信息检索

单针信息检索任务评估LLMs从特定长度的无关信息文本中回忆单个重要信息的能力。这个任务评估模型在长文本中识别和回忆特定信息的精确性。

### 多针信息检索

多针信息检索任务挑战LLMs识别和提取广泛文本中的多个关键信息点的能力。它模拟了现实世界中的场景，其中需要从文档或报告中检索多个数据点、事实或数字，评估模型在浏览和从密集文本中提取相关信息的效率。

### 多针信息推理

在NeedleBench V2中，多针信息推理任务得到了显著改进。原来基于R4C/MultiHop数据集的"针"已被替换为类似于祖源追溯挑战中的虚构信息。这一改变解决了潜在的内生知识偏差问题，因为原始数据集可能已被包含在一些模型的训练数据中。这个任务继续评估LLMs使用检索到的信息进行复杂推理的能力，要求模型不仅能回忆多个信息点，还能进行逻辑推理。

### 祖源追溯挑战 (ATC)

祖源追溯挑战在NeedleBench V2中进行了优化。针的分布模式从密集形式（1、2、3、4、5针）变为基于2的幂次的稀疏形式（2¹、2²、2³等）。这个任务仍然是NeedleBench中最复杂的任务，要求模型回忆和分析长文本中的每个细节，以解决需要理解复杂关系的问题，如家谱查询或详细案例分析。

NeedleBench V2引入了更平衡的评分系统。总体评分现在是通过三个主要任务（单针信息检索、多针信息检索和多针信息推理）的简单平均值计算得出，每个任务获得相等的权重。这一改变从先前的加权平均方法提供了一种更直接、更公平的方式，评估模型在不同检索和推理任务中的能力。

> 🔗 数据集主页链接[https://huggingface.co/datasets/opencompass/NeedleBench](https://huggingface.co/datasets/opencompass/NeedleBench)

## 数据集部署
建议从HuggingFace下载数据集：[https://huggingface.co/opencompass/NeedleBench](https://huggingface.co/datasets/opencompass/NeedleBench)
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径）
- 部署完成后，在`{工具根路径}/ais_bench/datasets`目录下执行`tree NeedleBench/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
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
## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|atc_0shot_nocot_2_power_en|atc_0shot_nocot_2_power_en|准确率(accuracy)|0-shot|对话格式|[atc_0shot_nocot_2_power_en.py](atc/atc_0shot_nocot_2_power_en.py)|
|needlebench_v2_4k|needlebench_v2_4k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_4k.py](needlebench_v2_4k/needlebench_v2_4k.py)|
|needlebench_v2_multi_reasoning_4k|needlebench_v2_multi_reasoning_4k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_multi_reasoning_4k.py](needlebench_v2_4k/needlebench_v2_multi_reasoning_4k.py)|
|needlebench_v2_multi_retrieval_4k|needlebench_v2_multi_retrieval_4k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_multi_retrieval_4k.py](needlebench_v2_4k/needlebench_v2_multi_retrieval_4k.py)|
|needlebench_v2_single_4k|needlebench_v2_single_4k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_single_4k.py](needlebench_v2_4k/needlebench_v2_single_4k.py)|
|needlebench_v2_8k|needlebench_v2_8k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_8k.py](needlebench_v2_8k/needlebench_v2_8k.py)|
|needlebench_v2_multi_reasoning_8k|needlebench_v2_multi_reasoning_8k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_multi_reasoning_8k.py](needlebench_v2_8k/needlebench_v2_multi_reasoning_8k.py)|
|needlebench_v2_multi_retrieval_8k|needlebench_v2_multi_retrieval_8k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_multi_retrieval_8k.py](needlebench_v2_8k/needlebench_v2_multi_retrieval_8k.py)|
|needlebench_v2_single_8k|needlebench_v2_single_8k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_single_8k.py](needlebench_v2_8k/needlebench_v2_single_8k.py)|
|needlebench_v2_multi_retrieval_compare_batch_8k|needlebench_v2_multi_retrieval_compare_batch_8k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_multi_retrieval_compare_batch_8k.py](needlebench_v2_8k/needlebench_v2_multi_retrieval_compare_batch_8k.py)|
|needlebench_v2_32k|needlebench_v2_32k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_32k.py](needlebench_v2_32k/needlebench_v2_32k.py)|
|needlebench_v2_multi_reasoning_32k|needlebench_v2_multi_reasoning_32k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_multi_reasoning_32k.py](needlebench_v2_32k/needlebench_v2_multi_reasoning_32k.py)|
|needlebench_v2_multi_retrieval_32k|needlebench_v2_multi_retrieval_32k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_multi_retrieval_32k.py](needlebench_v2_32k/needlebench_v2_multi_retrieval_32k.py)|
|needlebench_v2_single_32k|needlebench_v2_single_32k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_single_32k.py](needlebench_v2_32k/needlebench_v2_single_32k.py)|
|needlebench_v2_128k|needlebench_v2_128k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_128k.py](needlebench_v2_128k/needlebench_v2_128k.py)|
|needlebench_v2_multi_reasoning_128k|needlebench_v2_multi_reasoning_128k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_multi_reasoning_128k.py](needlebench_v2_128k/needlebench_v2_multi_reasoning_128k.py)|
|needlebench_v2_multi_retrieval_128k|needlebench_v2_multi_retrieval_128k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_multi_retrieval_128k.py](needlebench_v2_128k/needlebench_v2_multi_retrieval_128k.py)|
|needlebench_v2_single_128k|needlebench_v2_single_128k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_single_128k.py](needlebench_v2_128k/needlebench_v2_single_128k.py)|
|needlebench_v2_200k|needlebench_v2_200k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_200k.py](needlebench_v2_200k/needlebench_v2_200k.py)|
|needlebench_v2_multi_reasoning_200k|needlebench_v2_multi_reasoning_200k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_multi_reasoning_200k.py](needlebench_v2_200k/needlebench_v2_multi_reasoning_200k.py)|
|needlebench_v2_multi_retrieval_200k|needlebench_v2_multi_retrieval_200k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_multi_retrieval_200k.py](needlebench_v2_200k/needlebench_v2_multi_retrieval_200k.py)|
|needlebench_v2_single_200k|needlebench_v2_single_200k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_single_200k.py](needlebench_v2_200k/needlebench_v2_single_200k.py)|
|needlebench_v2_256k|needlebench_v2_256k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_256k.py](needlebench_v2_256k/needlebench_v2_256k.py)|
|needlebench_v2_multi_reasoning_256k|needlebench_v2_multi_reasoning_256k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_multi_reasoning_256k.py](needlebench_v2_256k/needlebench_v2_multi_reasoning_256k.py)|
|needlebench_v2_multi_retrieval_256k|needlebench_v2_multi_retrieval_256k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_multi_retrieval_256k.py](needlebench_v2_256k/needlebench_v2_multi_retrieval_256k.py)|
|needlebench_v2_single_256k|needlebench_v2_single_256k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_single_256k.py](needlebench_v2_256k/needlebench_v2_single_256k.py)|
|needlebench_v2_1000k|needlebench_v2_1000k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_1000k.py](needlebench_v2_1000k/needlebench_v2_1000k.py)|
|needlebench_v2_multi_reasoning_1000k|needlebench_v2_multi_reasoning_1000k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_multi_reasoning_1000k.py](needlebench_v2_1000k/needlebench_v2_multi_reasoning_1000k.py)|
|needlebench_v2_multi_retrieval_1000k|needlebench_v2_multi_retrieval_1000k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_multi_retrieval_1000k.py](needlebench_v2_1000k/needlebench_v2_multi_retrieval_1000k.py)|
|needlebench_v2_single_1000k|needlebench_v2_single_1000k|准确率(accuracy)|0-shot|对话格式|[needlebench_v2_single_1000k.py](needlebench_v2_1000k/needlebench_v2_single_1000k.py)|