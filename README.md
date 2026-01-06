<div align="center">
  <br />
  <br />

  # **AISBench 评测工具**
  #### 面向人工智能领域的测试基准工具
  <!-- 用分隔线替代背景 -->
  ---

[![][github-release-shield]][github-release-link]
[![][github-releasedate-shield]][github-releasedate-link]
[![][github-contributors-shield]][github-contributors-link]<br>
[![][github-forks-shield]][github-forks-link]
[![][github-stars-shield]][github-stars-link]
[![][github-issues-shield]][github-issues-link]
[![License](https://img.shields.io/badge/license-Apache--2.0-red?logo=apache)](https://www.apache.org/licenses/LICENSE-2.0)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/AISBench/benchmark)
<br><br>
[🌐官方网站](https://www.aisbench.com) |
[📖工具文档](https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/) |
[👨‍💻开发者文档](https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/develop_guide/contributing.html) |
[🔥最新进展](#-最新进展)|
[🤔报告问题](https://github.com/AISBench/benchmark/issues/new/choose)
<br><br>简体中文 | [English](README_en.md)
</div>

> ❗<span style="color: red;"><b>重要</b></span>
>
> **⭐️收藏项目**，你将能第一时间获取 AISBench评测工具 的最新动态～

## 🔥 最新进展
- **\[2025.12.19\]** 🎉 **AISBench 架构全面重构完成！**
  - ✨ **架构升级**：对cli、models、inferencer、tasks组件进行了全面重构，支持快速接入新的测试基准，参考📚 [开发者文档](https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/develop_guide/contributing.html)了解详情！
  - 🖥️ **任务管理界面**：全新的任务UI管理界面，支持同时监控每个任务的详细执行状态，包括任务名称、进度、时间成本、状态、日志路径、扩展参数等，让任务执行状态一目了然！
  - ⚡ **并行执行增强**：扩展了多任务并行功能，支持多个性能或精度测评任务并行执行，大幅提升评测效率！
  - 📊 **新增15+测评基准**：新增docvqa、infovqa、ocrbench_v2、omnidocbench、mmmu、mmmu_pro、mmstar、videomme、FewCLUE系列、dapo_math、leval等多模态和文本测评基准！
  - 🤖 **新增模型支持**：新增vllm/vllm-ascend VL 离线推理模型支持！
  - 🔧 **功能增强**：新增流式推理开关、自定义URL路径、API key配置；支持API模型推理warmup；支持自定义多模态数据集性能测评；部分数据集支持服务化PPL（困惑度）测评等多项功能！
  - 🏗️ **基础设施优化**：重构local models和api models组件，统一流式和非流式实现；重构inferencer组件，采用多进程+协程调用方式，提高并发能力；测试结果数据格式优化为jsonl，降低IO压力；采用错误码统一管理错误信息等！
- **\[2025.11.25\]** 支持服务化模型PPL(Perplexity-based，困惑度)模式精度测评。🔥🔥🔥
- **\[2025.9.08\]** 支持📚[模拟真实业务流量](https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/advanced_tutorials/rps_distribution.html)：通过控制请求发送速率波动，感知在模拟真实场景下服务化的性能测评结果！🔥🔥🔥

- **\[2025.8.28\]** 支持📚[多次独立重复推理精度场景](https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/base_tutorials/scenes_intro/accuracy_benchmark.html#id12)，计算pass@k/cons@k/avg@n等不同维度的精度指标！🔬🔬🔬

- **\[2025.8.19\]**
  - 新增Function Call专用模型配置 [vllm_api_function_call_chat](ais_bench/benchmark/configs/models/vllm_api/vllm_api_function_call_chat.py)，支持 [BFCL 函数调用能力评估](ais_bench/benchmark/configs/datasets/BFCL/README.md) 🔥🔥🔥
  - 提供工具支持的[性能测试规格说明](https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/base_tutorials/scenes_intro/performance_benchmark.html#id25)，优化推理集群场景工具内存占用及性能计算。最大规格场景（250K条请求，输入/输出token 4K/4K）内存占用降低60%，内存占用小于64GB；性能结果计算效率提升20倍。🚀🚀🚀
- **\[2025.7.15\]**
  - 支持[sharegpt](ais_bench/benchmark/configs/datasets/sharegpt/README.md)和[mtbench](ais_bench/benchmark/configs/datasets/mtbench/README.md)多轮对话数据集服务化性能测评和可视化，测评方式见📚[多轮对话测评指南](https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/advanced_tutorials/multiturn_benchmark.html)！🔥🔥🔥
  - 性能评测场景使用[自定义数据集](https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/advanced_tutorials/custom_dataset.html)，支持按请求粒度指定最大输出长度！🔥🔥🔥

- **\[2025.6.19\]** 支持📚[性能评测结果可视化](https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/base_tutorials/results_intro/performance_visualization.html)，辅助定位推理服务性能瓶颈！🔥🔥🔥
- **\[2025.6.12\]** 支持[textvqa](ais_bench/benchmark/configs/datasets/textvqa/README.md)、[videobench](ais_bench/benchmark/configs/datasets/videobench/README.md)和[vocalsound](ais_bench/benchmark/configs/datasets/vocalsound/README.md)等多模态数据集的精度和性能评测！🔥🔥🔥

- **\[2025.6.6\]** AISBench支持稳态性能评测，获取系统真实最佳性能，参考📚 [服务化稳定状态性能测试](doc/users_guide/stable_stage.md)进行快速上手! 🔥🔥🔥

- **\[2025.5.16\]** 支持3W+高并发服务化性能评测，📚 [性能指标](doc/users_guide/performance_metric.md)对齐🔗 [vllm benchmark](https://github.com/vllm-project/vllm/tree/main/benchmarks)，参考📚 [服务化性能测评指南](https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/base_tutorials/scenes_intro/performance_benchmark.html)了解详情！🔥🔥🔥

- **\[2025.4.30\]** 精度评测支持断点续测和失败用例重测，大幅提高精度评测鲁棒性，参考📚 [中断续测 & 失败用例重测](https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/base_tutorials/scenes_intro/accuracy_benchmark.html#id10)进行快速上手! 🔥🔥🔥

- **\[2025.4.15\]** 优化固定batch发送请求的方式为continuous batch模式发送请求，大幅提高精度评测效率! 🔥🔥🔥

- **\[2025.4.12\]** 支持合并MMLU、Ceval等所有多文件数据集为单个数据集任务进行精度评测，参考📚 [合并多文件数据集](https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/base_tutorials/scenes_intro/accuracy_benchmark.html#id11)了解详情！ 🔥🔥🔥


## 🌏 简介
AISBench Benchmark 是基于 [OpenCompass](https://github.com/open-compass/opencompass) 构建的模型评测工具，兼容 OpenCompass 的配置体系、数据集结构与模型后端实现，并在此基础上扩展了对服务化模型的支持能力。

当前，AISBench 支持两大类推理任务的评测场景：

🔍 [精度测评](https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/base_tutorials/scenes_intro/home.html#id2)：支持对服务化模型和本地模型在各类问答、推理基准数据集上的精度验证，覆盖文本、多模态等多种场景。

🚀 [性能测评](https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/base_tutorials/scenes_intro/home.html#id5)：支持对服务化模型的延迟与吞吐率评估，并可进行压测场景下的极限性能测试，支持稳态性能评测和真实业务流量模拟。

## 🛠️ 工具安装
✅ 环境要求

**Python 版本**：仅支持 Python **3.10**、 **3.11** 或 **3.12**

不支持 Python 3.9 及以下版本，也不兼容 Python 3.13 及以上版本

**推荐使用 Conda 管理环境**，以避免依赖冲突
```shell
conda create --name ais_bench python=3.10 -y
conda activate ais_bench
```

📦 安装方式（源码安装）

AISBench 当前仅提供源码安装方式，请确保安装环境联网：
```shell
git clone https://github.com/AISBench/benchmark.git
cd benchmark/
pip3 install -e ./ --use-pep517
```
该命令会自动安装核心依赖。
执行`ais_bench -h`，如果打印出AISBench评测工具的所有命令行的帮助信息，说明安装成功

⚙️ 服务化框架支持（可选）

若需评估服务化模型（如 vLLM、Triton 等），需额外安装相关依赖：
```shell
pip3 install -r requirements/api.txt
pip3 install -r requirements/extra.txt
```
⚙️ Huggingface多模态模型/vllm多模态离线推理支持（可选）
```shell
pip3 install -r requirements/hf_vl_dependency.txt
```
🔗 Berkeley Function Calling Leaderboard (BFCL) 测评支持

```shell
pip3 install -r requirements/datasets/bfcl_dependencies.txt --no-deps
```

**重要提示**：由于 `bfcl_eval` 会自动安装 `pathlib` 库，而 Python 3.5+ 环境已内置该库，为避免版本冲突，请务必使用 `--no-deps` 参数跳过额外依赖的自动安装。

如需进一步配置、使用 CLI 或 Python 脚本发起评测任务，请参考[快速入门指南](#快速入门)。

## ❌ 工具卸载
如需卸载 AISBench Benchmark，可执行以下命令：
```shell
pip3 uninstall ais_bench_benchmark
```

## 🚀 快速入门
### 命令含义
AISBench命令执行的单个或多个评测任务是由模型任务（单个或多个）、数据集任务（单个或多个）和结果呈现任务（单个）的组合定义的，AISBench的其他命令行则规定了评测任务的场景（精度评测场景、性能评测场景等）。以如下AISBench命令为例：
```shell
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --summarizer example
```
此命令没有指定其他命令行，默认是一个精度评测场景的任务，其中：
- `--models`指定了模型任务，即`vllm_api_general_chat`模型任务。

- `--datasets`指定了数据集任务，即`demo_gsm8k_gen_4_shot_cot_chat_prompt`数据集任务。

- `--summarizer`指定了结果呈现任务，即`example`结果呈现任务(不指定`--summarizer`精度评测场景默认使用`example`任务)，一般使用默认，不需要在命令行中指定，后续命令不指定。

多任务测评请参考：📚 精度场景的[多任务测评](./docs/source_zh_cn/base_tutorials/scenes_intro/accuracy_benchmark.md#多任务测评) 和 性能场景的[多任务测评](./docs/source_zh_cn/base_tutorials/scenes_intro/performance_benchmark.md#多任务测评)。

如需自行组合测评任务，实现更灵活的测评方式，可参考：📚 [自定义配置文件运行AISBench](./docs/source_zh_cn/advanced_tutorials/run_custom_config.md#自定义配置文件运行AISBench)。

### 任务含义查询(可选)
所选模型任务`vllm_api_general_chat`、数据集任务`demo_gsm8k_gen_4_shot_cot_chat_prompt`和结果呈现任务`example`的具体信息(简介，使用约束等)可以分别从如下链接中查询含义：
- `--models`: 📚 [服务化推理后端](https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/base_tutorials/all_params/models.html#id2)

- `--datasets`: 📚 [开源数据集](https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/base_tutorials/all_params/datasets.html#id3) → 📚 [详细介绍](ais_bench/benchmark/configs/datasets/demo/README.md)

- `--summarizer`: 📚 [结果汇总任务](https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/base_tutorials/all_params/summarizer.html)

### 运行命令前置准备
- `--models`: 使用`vllm_api_general_chat`模型任务，需要准备支持`v1/chat/completions`子服务的推理服务，可以参考🔗 [VLLM启动OpenAI 兼容服务器](https://docs.vllm.com.cn/en/latest/getting_started/quickstart.html#openai-compatible-server)启动推理服务
- `--datasets`: 使用`demo_gsm8k_gen_4_shot_cot_chat_prompt`数据集任务，需要准备gsm8k数据集，可以从🔗 [opencompass
提供的gsm8k数据集压缩包](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip)下载。将解压后的`gsm8k/`文件夹部署到AISBench评测工具根路径下的`ais_bench/datasets`文件夹下。

### 任务对应配置文件修改
每个模型任务、数据集任务和结果呈现任务都对应一个配置文件，运行命令前需要修改这些配置文件的内容。这些配置文件路径可以通过在原有AISBench命令基础上加上`--search`来查询，例如：
```shell
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --search
```
> ⚠️ **注意**： 执行带search命令会打印出任务对应的配置文件的绝对路径。

执行查询命令可以得到如下查询结果：
```shell
╒══════════════╤═══════════════════════════════════════╤════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╕
│ Task Type    │ Task Name                             │ Config File Path                                                                                                               │
╞══════════════╪═══════════════════════════════════════╪════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╡
│ --models     │ vllm_api_general_chat                 │ /your_workspace/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py                                 │
├──────────────┼───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ --datasets   │ demo_gsm8k_gen_4_shot_cot_chat_prompt │ /your_workspace/benchmark/ais_bench/benchmark/configs/datasets/demo/demo_gsm8k_gen_4_shot_cot_chat_prompt.py                   │
╘══════════════╧═══════════════════════════════════════╧════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╛

```

- 快速入门中数据集任务配置文件`demo_gsm8k_gen_4_shot_cot_chat_prompt.py`不需要做额外修改，数据集任务配置文件内容介绍可参考📚 [配置开源数据集](https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/base_tutorials/all_params/datasets.html#id6)

模型配置文件`vllm_api_general_chat.py`中包含了模型运行相关的配置内容，是需要依据实际情况修改的。快速入门中需要修改的内容用注释标明。
```python
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="",                    # 指定模型序列化词表文件绝对路径（精度测试场景一般不需要配置）
        model="",        # 指定服务端已加载模型名称，依据实际VLLM推理服务拉取的模型名称配置（配置成空字符串会自动获取）
        stream=False,
        request_rate=0,           # 请求发送频率，每1/request_rate秒发送1个请求给服务端，小于0.1则一次性发送所有请求
        retry=2,                  # 每个请求最大重试次数
        api_key="",               # 自定义API key，默认是空字符串
        host_ip="localhost",      # 指定推理服务的IP
        host_port=8080,           # 指定推理服务的端口
        url="",                     # 自定义访问推理服务的URL路径(当base url不是http://host_ip:host_port的组合时需要配置, 配置后host_ip和host_port会被忽略)
        max_out_len=512,          # 推理服务输出的token的最大数量
        batch_size=1,               # 请求发送的最大并发数
        trust_remote_code=False,    # tokenizer是否信任远程代码，默认False;
        generation_kwargs=dict(   # 模型推理参数，参考VLLM文档配置，AISBench评测工具不做处理，在发送的请求中附带
            temperature=0.01,
            ignore_eos=False,
        )
    )
]
```
### 执行命令
修改好配置文件后，执行命令启动服务化精度评测：
```bash
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt
```
#### 查看任务执行细节
执行AISBench命令后，任务管理界面会在命令行实时刷新显示任务执行状态（键盘按"P"键可以暂停/恢复刷新，用于复制看板信息，再按"P"键可以继续刷新）。任务管理界面支持同时监控多个任务的详细执行状态，包括任务名称、进度、时间成本、状态、日志路径、扩展参数等信息，例如：
```
Base path of result&log : outputs/default/20250628_151326
Task Progress Table (Updated at: 2025-11-06 10:08:21)
Page: 1/1  Total 2 rows of data
Press Up/Down arrow to page,  'P' to PAUZE/RESUME screen refresh, 'Ctrl + C' to exit

+----------------------------------+-----------+-------------------------------------------------+-------------+-------------+-------------------------------------------------+------------------------------------------------+
| Task Name                        |   Process | Progress                                        | Time Cost   | Status      | Log Path                                        | Extend Parameters                              |
+==================================+===========+=================================================+=============+=============+=================================================+================================================+
| vllm-api-general-chat/demo_gsm8k |    547141 | [###############               ] 4/8 [0.5 it/s] | 0:00:11     | inferencing | logs/infer/vllm-api-general-chat/demo_gsm8k.out | {'POST': 5, 'RECV': 4, 'FINISH': 4, 'FAIL': 0} |
+----------------------------------+-----------+-------------------------------------------------+-------------+-------------+-------------------------------------------------+------------------------------------------------+

```

任务执行的细节日志会不断落盘在默认的输出路径，这个输出路径在实时刷新的看板上显示，即`Log Path`。`Log Path`（`logs/infer/vllm-api-general-chat/demo_gsm8k.out`）是在`Base path`（`outputs/default/20250628_151326`）下的路径，以上述的看板信息为例，任务执行的详细日志路径为：
```shell
# {Base path}/{Log Path}
outputs/default/20250628_151326/logs/infer/vllm-api-general-chat/demo_gsm8k.out
```

> 💡 如果希望执行过程中将详细日志直接打印，执行命令时可以加上 `--debug`:
`ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --debug`




`Base path`（`outputs/default/20250628_151326`）下包含了所有任务的执行细节，命令执行结束后所有的执行细节如下：
```shell
20250628_151326/
├── configs # 模型任务、数据集任务和结构呈现任务对应的配置文件合成的一个配置
│   └── 20250628_151326_29317.py
├── logs # 执行过程中日志，命令中如果加--debug，不会有过程日志落盘（都直接打印出来了）
│   ├── eval
│   │   └── vllm-api-general-chat
│   │       └── demo_gsm8k.out # 基于predictions/文件夹下的推理结果的精度评测过程的日志
│   └── infer
│       └── vllm-api-general-chat
│           └── demo_gsm8k.out # 推理过程日志
├── predictions
│   └── vllm-api-general-chat
│       └── demo_gsm8k.json # 推理结果（推理服务返回的所有输出）
├── results
│   └── vllm-api-general-chat
│       └── demo_gsm8k.json # 精度评测计算的原始分数
└── summary
    ├── summary_20250628_151326.csv # 最终精度分数呈现（表格格式）
    ├── summary_20250628_151326.md # 最终精度分数呈现（markdown格式）
    └── summary_20250628_151326.txt # # 最终精度分数呈现（文本格式）
```
> ⚠️ **注意**： 不同评测场景落盘任务执行细节内容不同，具体请参考具体评测场景的指南。

#### 输出结果
因为只有8条数据，会很快跑出结果，结果显示的示例如下
```bash
dataset                 version  metric   mode  vllm_api_general_chat
----------------------- -------- -------- ----- ----------------------
demo_gsm8k              401e4c   accuracy gen                   62.50
```

更多教程请查看我们的👉[文档](https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/)


## 🔜 即将推出
- [x] **\[已完成\]** ✅ AISBench完成全面重构，支持在AISBench框架下🔌插件化集成前沿测试基准，以应对业界愈发复杂多样化的测试任务；并且显著提高易用性。
- [ ] **\[规划中\]** 持续扩展业界前沿的多模态测评能力，支持更多多模态数据集和评测场景。
- [ ] **\[规划中\]** 提供业界主流Agent测评能力，支持Agent任务链和工具调用等复杂场景的评测。

## 🤝 致谢
- 本项目代码基于🔗 [OpenCompass](https://github.com/open-compass/opencompass)做拓展开发。
- 本项目部分数据集和提示词实现修改自[simple-evals](https://github.com/openai/simple-evals)。
- 本项目代码中打点的性能指标与[VLLM Benchmark](https://github.com/vllm-project/vllm/tree/main/benchmarks)对齐。
- 本项目的BFCL函数调用能力评估功能基于 [Berkeley Function Calling Leaderboard (BFCL)](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) 实现。

<p align="right"><a href="#top">🔝Back to top</a></p>

[github-contributors-link]: https://github.com/AISBench/benchmark/graphs/contributors
[github-contributors-shield]: https://img.shields.io/github/contributors/AISBench/benchmark?color=c4f042&labelColor=black&style=flat-square
[github-forks-link]: https://github.com/AISBench/benchmark/network/members
[github-forks-shield]: https://img.shields.io/github/forks/AISBench/benchmark?color=8ae8ff&labelColor=black&style=flat-square
[github-issues-link]: https://github.com/AISBench/benchmark/issues
[github-issues-shield]: https://img.shields.io/github/issues/AISBench/benchmark?color=ff80eb&labelColor=black&style=flat-square
[github-license-link]: https://github.com/AISBench/benchmark/blob/main/LICENSE
[github-license-shield]: https://img.shields.io/github/license/AISBench/benchmark?color=white&labelColor=black&style=flat-square
[github-release-link]: https://github.com/AISBench/benchmark/releases
[github-release-shield]:  https://img.shields.io/github/v/release/AISBench/benchmark?color=369eff&labelColor=black&logo=github&style=flat-square
[github-releasedate-link]: https://github.com/AISBench/benchmark/releases
[github-releasedate-shield]: https://img.shields.io/github/release-date/AISBench/benchmark?labelColor=black&style=flat-square
[github-stars-link]: https://github.com/AISBench/benchmark/stargazers
[github-stars-shield]: https://img.shields.io/github/stars/AISBench/benchmark?color=ffcb47&labelColor=black&style=flat-square
[github-trending-shield]: https://trendshift.io/api/badge/repositories/6630
[github-trending-url]: https://trendshift.io/repositories/6630