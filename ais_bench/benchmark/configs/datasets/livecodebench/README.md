# LiveCodeBench
中文 | [English](README_en.md)
## 数据集简介
LiveCodeBench 是一个持续更新的"实时"基准测试平台，用于全面评估大语言模型（LLMs）的代码相关能力。该平台主要评估模型在代码生成、自我修复、测试输出预测和代码执行等多方面的表现。当前展示的是其代码生成场景，同时也被用于通过测试用例反馈来评估模型的自我修复能力。

该基准测试的问题采集自编程竞赛网站，特别注重保持题目质量、测试用例质量以及题目难度多样性。当前版本包含来自LeetCode、AtCoder和Codeforces的500余道题目。每个问题实例均包含题目描述、输入/输出示例及隐藏测试用例，且所有题目均标注难度等级和发布时间，便于测量模型在不同时间窗口下的表现。最终目标是针对每个问题实例生成正确且高效的解决方案。

初始版本的代码生成数据集因包含大量测试用例而导致数据集规模过大。当前（精简）版本在保持与原始数据集相似性能的前提下，对测试用例进行了筛选和抽样。未来LiveCodeBench将采用此精简版本进行代码生成评估。

> 🔗 数据集主页链接[https://livecodebench.github.io/](https://livecodebench.github.io/)

## 数据集部署
- 可以从huggingface的数据集链接🔗 [https://huggingface.co/datasets/livecodebench/code_generation_lite/tree/main](https://huggingface.co/datasets/livecodebench/code_generation_lite/tree/main)中获取
- 请将该数据集部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），若部署在自定义路径可能遇到数据集相关的报错（2025.11.25），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
git lfs install
git clone https://huggingface.co/datasets/livecodebench/code_generation_lite
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree code_generation_lite/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    code_generation_lite
    ├── code_generation_lite.py
    ├── test6.jsonl
    ├── test5.jsonl
    ├── test4.jsonl
    ├── test3.jsonl
    ├── test2.jsonl
    └── test.jsonl
    ```

> ⚠️ **重要**：`code_generation_lite.py` 是数据集仓库自带的加载脚本，包含了数据集的**版本信息**与**数据导入逻辑**（数据集按版本区分，版本信息即来自此文件）。**该文件缺失将导致数据集加载失败**。通过 `git lfs install && git clone https://huggingface.co/datasets/livecodebench/code_generation_lite` 完整克隆仓库即可自动获得该文件，请勿删除或遗漏。

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|配套文件导入方式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- | --- |
|livecodebench_0_shot_chat_v4_v5|code_generation_lite数据集的生成式任务，与DeepSeek-R1测评使用数据集一致：LiveCodeBench(2024-08 – 2025-01)|pass@1|0-shot|对话格式|`from ais_bench.benchmark.configs.datasets.livecodebench.livecodebench_0_shot_chat_v4_v5 import LCB_datasets as datasets`|[livecodebench_0_shot_chat_v4_v5.py](livecodebench_0_shot_chat_v4_v5.py)|
|livecodebench_0_shot_chat_v4_v5_v6|code_generation_lite数据集的生成式任务, 与DeepSeek-V3.1和DeepSeek-V3.2测评使用数据集一致：LiveCodeBench(2024-08 – 2025-05)|pass@1|0-shot|对话格式|`from ais_bench.benchmark.configs.datasets.livecodebench.livecodebench_0_shot_chat_v4_v5_v6 import LCB_datasets as datasets`|[livecodebench_0_shot_chat_v4_v5_v6.py](livecodebench_0_shot_chat_v4_v5_v6.py)|
|livecodebench_0_shot_chat_v6|code_generation_lite数据集的生成式任务, 与Qwen3测评使用数据集一致：LiveCodeBench(2025-05)|pass@1|0-shot|对话格式|`from ais_bench.benchmark.configs.datasets.livecodebench.livecodebench_0_shot_chat_v6 import LCB_datasets as datasets`|[livecodebench_0_shot_chat_v6.py](livecodebench_0_shot_chat_v6.py)|
