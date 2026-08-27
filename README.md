<div align="center">
  <br />
  <br />

# **AISBench 评测工具**

#### 面向人工智能领域的测试基准工具

  <!-- 用分隔线替代背景 -->

***

[![](https://img.shields.io/github/v/release/AISBench/benchmark?color=369eff\&labelColor=black\&logo=github\&style=flat-square)](https://github.com/AISBench/benchmark/releases)
[![](https://img.shields.io/github/release-date/AISBench/benchmark?labelColor=black\&style=flat-square)](https://github.com/AISBench/benchmark/releases)
[![](https://img.shields.io/github/contributors/AISBench/benchmark?color=c4f042\&labelColor=black\&style=flat-square)](https://github.com/AISBench/benchmark/graphs/contributors)
[![](https://img.shields.io/github/forks/AISBench/benchmark?color=8ae8ff\&labelColor=black\&style=flat-square)](https://github.com/AISBench/benchmark/network/members)
[![](https://img.shields.io/github/stars/AISBench/benchmark?color=ffcb47\&labelColor=black\&style=flat-square)](https://github.com/AISBench/benchmark/stargazers)
[![](https://img.shields.io/github/issues/AISBench/benchmark?color=ff80eb\&labelColor=black\&style=flat-square)](https://github.com/AISBench/benchmark/issues)
[![License](https://img.shields.io/badge/license-Apache--2.0-red?logo=apache)](https://www.apache.org/licenses/LICENSE-2.0)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/AISBench/benchmark)

[🌐官方网站](https://www.aisbench.com) |
[📖工具文档](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/) |
[👨‍💻开发者文档](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/develop_guide/contributing.html) |
[🔥最新进展](#-最新进展)|
[🤔报告问题](https://github.com/AISBench/benchmark/issues/new/choose)
[📦容器镜像](https://github.com/AISBench/benchmark/pkgs/container/aisbench_benchmark)

简体中文 | [English](README_en.md)

</div>

> ❗<span style="color: red;"><b>重要</b></span>
>
> **⭐️收藏项目**，你将能第一时间获取 AISBench评测工具 的最新动态～

## 🔥 最新进展

- **\[2026.6.29]** 接入文生图模型多维评测基准 **OneIG-Benchmark**：围绕**对齐性、文本渲染、推理能力、风格表现、多样性**五个维度对生成图片进行综合评估，支持 EN/ZH 双语言模式，采用 **LLM-as-Judge + 专用小模型** 的混合评测方式，与官方评测方法精度差异 < 1%。🔥🔥🔥 示例与说明见 [在AISBench中测评OneIG-Benchmark](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/extended_benchmark/lmm_generate/oneig.html)。
- **\[2026.6.15]** 新增 **Docker 镜像多架构支持**：AISBench Benchmark 镜像同步支持 **x86_64** 与 **aarch64 (ARM)** 架构，覆盖 Ubuntu 22.04/24.04、openEuler 22.03/24.03 与 Python 3.10/3.11/3.12 多种组合；同时 AISBench 已发布到 **PyPI**，可通过 `pip install ais_bench_benchmark` / `pip install ais_bench_benchmark[full]` 一键安装。镜像说明见 [Docker 镜像概览](docker/OVERVIEW.zh.md)。🚀🚀🚀
- **\[2026.6.11]** 接入长时域软件工程智能体评测基准 **SWE-Bench Pro**：支持在 x86 环境下对智能体模型在长时域软件工程任务上的表现进行评测，提供 `full` / `mini` 数据集以及 `mini-swe-agent` infer 与 SWE-Bench Pro harness eval 的端到端流程。示例与说明见 [在AISBench中测评SWE-Bench Pro](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/extended_benchmark/agent/swe_bench_pro.html)。🔥🔥🔥
- **\[2026.5.30]** 支持在AISBench中测评基于Harbor的Terminal-Bench 2.0，并支持mini数据集，大幅降低评测成本🔥🔥🔥。示例与说明见 [在AISBench中测评基于Harbor的Terminal-Bench 2.0](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/extended_benchmark/agent/harbor_bench.html)。
- **\[2026.5.18]** 支持在AISBench中测评SWE-Bench、TAU2-Bench、VBench 1.0的mini子集，大幅降低评测成本🔥🔥🔥。mini子集示例与说明在对应测评文档中搜索`mini`关键词：
  - [在AISBench中测评SWE-Bench](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/extended_benchmark/agent/swe_bench.html)
  - [在AISBench中测评TAU2-Bench](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/extended_benchmark/agent/tau2_bench.html)
  - [在AISBench中测评VBench](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/extended_benchmark/lmm_generate/vbench.html)
- **\[2026.5.07]** 接入视频生成质量评测基准 VBench 1.0：支持在 GPU / NPU 上对**生成视频**进行多维度质量/语义指标测评。示例与说明见 [在AISBench中测评VBench](docs/source_zh_cn/extended_benchmark/lmm_generate/vbench.md)。 🔥🔥🔥
- **\[2026.4.14]** 接入大模型智能体评测基准τ²-Bench，支持双控环境下的对话、工具调用与合规能力评估，详见[在AISBench中测评τ²-Bench](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/extended_benchmark/agent/tau2_bench.html)。 🔥🔥🔥
- **\[2026.4.10]** 接入首个智能体评测基准SWE-Bench， 支持对智能体模型进行评测，详见[在AISBench中测评SWE-Bench](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/extended_benchmark/agent/swe_bench.html)。 🔥🔥🔥
- **\[2026.3.10]** 接入首个图像生成类评测基准GEdit-Bench， 支持对图像生成模型进行评测，详见[在AISBench中测评GEdit-Bench](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/extended_benchmark/lmm_generate/gedit_bench.html)。 🔥🔥🔥
- **\[2026.3.1]** 支持接入裁判模型进行评估，详见[使用裁判模型进行测评](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/advanced_tutorials/judge_model_evaluate.html)。 🔥🔥🔥
- **\[2026.1.31]** 支持 [Mooncake Trace](ais_bench/benchmark/configs/datasets/mooncake_trace/README.md) trace 数据集性能测评，支持按时间戳调度请求、hash\_id 缓存与可复现 prompt 生成，详见数据集 README。🔥🔥🔥
- **\[2025.12.19]** 🎉 **AISBench 架构全面重构完成！**
  - ✨ **架构升级**：对cli、models、inferencer、tasks组件进行了全面重构，支持快速接入新的测试基准，参考📚 [开发者文档](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/develop_guide/contributing.html)了解详情！
  - 🖥️ **任务管理界面**：全新的任务UI管理界面，支持同时监控每个任务的详细执行状态，包括任务名称、进度、时间成本、状态、日志路径、扩展参数等，让任务执行状态一目了然！
  - ⚡ **并行执行增强**：扩展了多任务并行功能，支持多个性能或精度测评任务并行执行，大幅提升评测效率！
  - 📊 **新增15+测评基准**：新增docvqa、infovqa、ocrbench\_v2、omnidocbench、mmmu、mmmu\_pro、mmstar、videomme、FewCLUE系列、dapo\_math、leval等多模态和文本测评基准！
  - 🤖 **新增模型支持**：新增vllm/vllm-ascend VL 离线推理模型支持！
  - 🔧 **功能增强**：新增流式推理开关、自定义URL路径、API key配置；支持API模型推理warmup；支持自定义多模态数据集性能测评；部分数据集支持服务化PPL（困惑度）测评等多项功能！
  - 🏗️ **基础设施优化**：重构local models和api models组件，统一流式和非流式实现；重构inferencer组件，采用多进程+协程调用方式，提高并发能力；测试结果数据格式优化为jsonl，降低IO压力；采用错误码统一管理错误信息等！
- **\[2025.11.25]** 支持服务化模型PPL(Perplexity-based，困惑度)模式精度测评。🔥🔥🔥
- **\[2025.9.08]** 支持📚[模拟真实业务流量](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/advanced_tutorials/rps_distribution.html)：通过控制请求发送速率波动，感知在模拟真实场景下服务化的性能测评结果！🔥🔥🔥
- **\[2025.8.28]** 支持📚[多次独立重复推理精度场景](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/scenes_intro/accuracy_benchmark.html#id12)，计算pass\@k/cons\@k/avg\@n等不同维度的精度指标！🔬🔬🔬
- **\[2025.8.19]**
  - 新增Function Call专用模型配置 [vllm\_api\_function\_call\_chat](ais_bench/benchmark/configs/models/vllm_api/vllm_api_function_call_chat.py)，支持 [BFCL 函数调用能力评估](ais_bench/benchmark/configs/datasets/BFCL/README.md) 🔥🔥🔥
  - 提供工具支持的[性能测试规格说明](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/scenes_intro/performance_benchmark.html#id25)，优化推理集群场景工具内存占用及性能计算。最大规格场景（250K条请求，输入/输出token 4K/4K）内存占用降低60%，内存占用小于64GB；性能结果计算效率提升20倍。🚀🚀🚀
- **\[2025.7.15]**
  - 支持[sharegpt](ais_bench/benchmark/configs/datasets/sharegpt/README.md)和[mtbench](ais_bench/benchmark/configs/datasets/mtbench/README.md)多轮对话数据集服务化性能测评和可视化，测评方式见📚[多轮对话测评指南](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/advanced_tutorials/multiturn_benchmark.html)！🔥🔥🔥
  - 性能评测场景使用[自定义数据集](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/advanced_tutorials/custom_dataset.html)，支持按请求粒度指定最大输出长度！🔥🔥🔥
- **\[2025.6.19]** 支持📚[性能评测结果可视化](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/results_intro/performance_visualization.html)，辅助定位推理服务性能瓶颈！🔥🔥🔥
- **\[2025.6.12]** 支持[textvqa](ais_bench/benchmark/configs/datasets/textvqa/README.md)、[videobench](ais_bench/benchmark/configs/datasets/videobench/README.md)和[vocalsound](ais_bench/benchmark/configs/datasets/vocalsound/README.md)等多模态数据集的精度和性能评测！🔥🔥🔥
- **\[2025.6.6]** AISBench支持稳态性能评测，获取系统真实最佳性能，参考📚 [服务化稳定状态性能测试](docs/source_zh_cn/advanced_tutorials/stable_stage.md)进行快速上手! 🔥🔥🔥
- **\[2025.5.16]** 支持3W+高并发服务化性能评测，📚 [性能指标](docs/source_zh_cn/base_tutorials/results_intro/performance_metric.md)对齐🔗 [vllm benchmark](https://github.com/vllm-project/vllm/tree/main/benchmarks)，参考📚 [服务化性能测评指南](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/scenes_intro/performance_benchmark.html)了解详情！🔥🔥🔥
- **\[2025.4.30]** 精度评测支持断点续测和失败用例重测，大幅提高精度评测鲁棒性，参考📚 [中断续测 & 失败用例重测](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/scenes_intro/accuracy_benchmark.html#id10)进行快速上手! 🔥🔥🔥

## 🌏 简介

AISBench Benchmark 是基于 [OpenCompass](https://github.com/open-compass/opencompass) 构建的模型评测工具，兼容 OpenCompass 的配置体系、数据集结构与模型后端实现，并在此基础上扩展了对服务化模型的支持能力。

当前，AISBench 支持两大类推理任务的评测场景：

🔍 [精度测评](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/scenes_intro/home.html#id2)：支持对服务化模型和本地模型在各类问答、推理基准数据集上的精度验证，覆盖文本、多模态等多种场景。

🚀 [性能测评](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/scenes_intro/home.html#id5)：支持对服务化模型的延迟与吞吐率评估，并可进行压测场景下的极限性能测试，支持稳态性能评测和真实业务流量模拟。

## 🛠️ 工具安装

✅ 环境要求

**Python 版本**：仅支持 Python **3.10**、 **3.11** 或 **3.12**

不支持 Python 3.9 及以下版本，也不兼容 Python 3.13 及以上版本

**推荐使用 Conda 管理环境**，以避免依赖冲突

```shell
conda create --name ais_bench python=3.10 -y
conda activate ais_bench
```

### 📦 安装方式-源码安装（首选）

AISBench 当前推荐使用源码安装方式，以便获得更好的自定义配置文件使用体验，请确保安装环境联网：

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

🔗 OCRBench\_v2数据集测评支持（可选）

```shell
pip3 install -r requirements/datasets/ocrbench_v2.txt
```

### 📦 安装方式-一键安装（备选）

AISBench 也提供了一键安装方式，适用于基于预置配置文件的快速体验和评估场景，请确保安装环境联网。

- 基本功能的安装命令如下：

```shell
pip3 install ais_bench_benchmark
```

- 全量功能的安装命令如下：

```shell
pip3 install ais_bench_benchmark[full]
```

如需进一步配置、使用 CLI 或 Python 脚本发起评测任务，请参考[快速入门指南](#快速入门)。

## ❌ 工具卸载

如需卸载 AISBench Benchmark，可执行以下命令：

```shell
pip3 uninstall ais_bench_benchmark
```

## 🚀 快速入门

### 运行命令前置准备

- 需要准备支持`v1/chat/completions`子服务的推理服务，可以参考🔗 [VLLM启动OpenAI 兼容服务器](https://docs.vllm.com.cn/en/latest/getting_started/quickstart.html#openai-compatible-server)启动推理服务
- 需要准备gsm8k数据集，可以从🔗 [opencompass
  提供的gsm8k数据集压缩包](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip)下载。将解压后的`gsm8k/`文件夹部署到AISBench评测工具根路径下的`ais_bench/datasets`文件夹下。

### 启动测评（两种方式任选其一）

| ⭐ 推荐：使用自定义配置文件      | 备选：使用命令行参数(原快速入门方式)                      |
| :------------------ | :------------------------------ |
| 修改一个文件，集中管理所有配置，在任意路径写配置     | 通过 `--models` `--datasets` 参数指定 |
| 一次编写，多次复用           | 每次运行需输入完整命令                     |
| 支持 Python 全部语法，灵活扩展 | 仅支持笛卡尔积组合                       |

**⭐ 推荐：使用自定义配置文件**

AISBench 提供了预置的自定义配置文件 [model\_api\_test\_zh\_cn.py](ais_bench/configs/model_api_test_zh_cn.py)，将常见的推理服务化测试配置（模型选择、服务地址、端口、生成参数等）集中在一个文件中，无需分别查找和修改多个配置文件。该文件本质上是 Python 脚本，支持所有 Python 语法，你可以自由扩展。

打开 `ais_bench/configs/model_api_test_zh_cn.py`，根据实际情况修改以下配置（如果是`pip3 install ais_bench_benchmark`方式直接安装工具，可以在任意路径自行创建`model_api_test_zh_cn.py`，将以下配置内容写入该文件）：

```python
from mmengine.config import read_base

with read_base():
# 模型任务，选择其中一个，其他模型任务参考：https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/base_tutorials/all_params/models.html 获取更多模型任务
    # vllm_api_general 是基础模型，仅支持文本生成
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general import models as vllm_api_general
    # vllm_api_general_chat 是对话模型，支持对话
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_chat import models as vllm_api_general_chat
    # vllm_api_stream_chat 是流式对话模型，支持流式对话
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat
    # vllm_api_general_stream 是流式模型，支持流式生成
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_stream import models as vllm_api_general_stream

# 数据集任务，参考：https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/get_started/datasets.html 获取更多数据集任务
    from ais_bench.benchmark.configs.datasets.demo.demo_gsm8k_gen_4_shot_cot_chat_prompt import gsm8k_datasets as datasets

models = vllm_api_general_chat

models[0]["path"] = ""  # 指定模型序列化词表文件的绝对路径（精度测试场景一般不需要配置）
models[0]["model"] = "" # 指定服务端加载的模型名称，根据 VLLM 推理服务实际拉取的模型名称配置（配置为空字符串则自动获取）
models[0]["request_rate"] = 0 # 请求发送频率：每 1/request_rate 秒向服务端发送 1 条请求；小于 0.001 时一次性发送所有请求
models[0]["api_key"] = "" # 自定义 API key，默认为空字符串
models[0]["host_ip"] = "localhost" # 指定推理服务的 IP
models[0]["host_port"] = 8080 # 指定推理服务的端口
models[0]["url"] = "" # 自定义访问推理服务的 URL 路径（当基础 URL 不是 http://host_ip:host_port 的组合时需要配置；配置后 host_ip 和 host_port 将被忽略）
models[0]["max_out_len"] = 512 # 推理服务输出的最大 token 数
models[0]["batch_size"] = 1 # 发送请求的最大并发数
models[0]["trust_remote_code"] = False # tokenizer 是否信任远程代码，默认为 False
models[0]["generation_kwargs"] = dict( # 模型推理参数，参考 VLLM 文档配置；AISBench 评测工具不做处理，直接附加到发送的请求中
    temperature=0.01,
    ignore_eos=False,
)

# datasets[0]["path"] = ais_bench/datasets/gsm8k # 指定数据集目录的绝对路径（精度测试场景需要配置）

work_dir = 'outputs/default/'  # 指定任务结果和日志的保存工作目录（默认为 outputs/default/）

```
> 💡 配置文件中已预置了常用模型类型的导入（`vllm_api_general`、`vllm_api_general_chat`、`vllm_api_stream_chat`、`vllm_api_general_stream`），只需取消/修改注释即可切换。更多自定义配置文件的用法请参考 📚 [自定义配置文件运行AISBench](./docs/source_zh_cn/advanced_tutorials/run_custom_config.md)。

数据集任务的选取、准备和使用参考如下步骤：
1. 在📚 [开源数据集](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/get_started/datasets.html#id3)内选取数据集任务
2. 进入数据的 📚 [详细介绍/数据集部署](ais_bench/benchmark/configs/datasets/demo/README.md#数据集部署)准备数据集
3. 参考📚 [详细介绍/可用数据集任务](ais_bench/benchmark/configs/datasets/demo/README.md#可用数据集任务)选取可用数据集任务，并将对应的任务导入方式（例如`from ais_bench.benchmark.configs.datasets.demo.demo_gsm8k_gen_4_shot_cot_chat_prompt import gsm8k_datasets as datasets`）复制到自定义配置文件中

修改好配置文件后，执行如下命令启动服务化精度评测：

```bash
ais_bench ais_bench/configs/model_api_test_zh_cn.py
```

***

**备选：使用命令行参数**

如果你更习惯使用命令行参数方式，AISBench 同样支持通过 `--models`、`--datasets`、`--summarizer` 参数直接指定任务。以下是与上述自定义配置文件方式**执行效果完全相同**的命令行方式。

AISBench命令执行的单个或多个评测任务是由模型任务（单个或多个）、数据集任务（单个或多个）和结果呈现任务（单个）的组合定义的。以如下AISBench命令为例：

```shell
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --summarizer example
```

此命令没有指定其他命令行，默认是一个精度评测场景的任务，其中：

- `--models`指定了模型任务，即`vllm_api_general_chat`模型任务。
- `--datasets`指定了数据集任务，即`demo_gsm8k_gen_4_shot_cot_chat_prompt`数据集任务。
- `--summarizer`指定了结果呈现任务，即`example`结果呈现任务(不指定`--summarizer`精度评测场景默认使用`example`任务)，一般使用默认，不需要在命令行中指定。

多任务测评请参考：📚 精度场景的[多任务测评](./docs/source_zh_cn/base_tutorials/scenes_intro/accuracy_benchmark.md#多任务测评) 和 性能场景的[多任务测评](./docs/source_zh_cn/base_tutorials/scenes_intro/performance_benchmark.md#多任务测评)。

如需自行组合测评任务，实现更灵活的测评方式，可参考：📚 [自定义配置文件运行AISBench](./docs/source_zh_cn/advanced_tutorials/run_custom_config.md#自定义配置文件运行AISBench)。

所选模型任务`vllm_api_general_chat`、数据集任务`demo_gsm8k_gen_4_shot_cot_chat_prompt`和结果呈现任务`example`的具体信息(简介，使用约束等)可以分别从如下链接中查询含义：

- `--models`: 📚 [服务化推理后端](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/all_params/models.html#id2)
- `--datasets`: 📚 [开源数据集](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/get_started/datasets.html#id3) → 📚 [详细介绍](ais_bench/benchmark/configs/datasets/demo/README.md)
- `--summarizer`: 📚 [结果汇总任务](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/all_params/summarizer.html)

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

- 快速入门中数据集任务配置文件`demo_gsm8k_gen_4_shot_cot_chat_prompt.py`不需要做额外修改，数据集任务配置文件内容介绍可参考📚 [配置开源数据集](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/all_params/datasets.html#id6)

模型配置文件`vllm_api_general_chat.py`中包含了模型运行相关的配置内容，是需要依据实际情况修改的。快速入门中需要修改的内容用注释标明。

> 💡 **提示**：模型配置中的部分参数（如 `host_ip`、`host_port`、`model`、`url`、`max_out_len`、`generation_kwargs` 等）无需修改配置文件，可直接通过命令行覆盖，例如：
>
> ```bash
> ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --host-ip 127.0.0.1 --host-port 8000
> ```
>
> 命令行显式指定的参数会覆盖本次执行的所有模型配置中对应字段；仅覆盖配置中**已存在的字段**，未指定的参数保持配置文件原值。更多可覆盖参数及覆盖范围说明请参考 📚 [用户配置参数 - API 模型通用覆盖参数](./docs/source_zh_cn/base_tutorials/all_params/cli_args.md#api-模型通用覆盖参数)。

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
        use_timestamp=False,      # 是否按数据集中 timestamp 调度请求，适用于含 timestamp 的数据集（如 Mooncake Trace）
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

修改好配置文件后，执行命令启动服务化精度评测：

```bash
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt
```

### 查看任务执行细节

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
> `ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --debug`

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

### 输出结果

因为只有8条数据，会很快跑出结果，结果显示的示例如下

```bash
dataset                 version  metric   mode  vllm_api_general_chat
----------------------- -------- -------- ----- ----------------------
demo_gsm8k              401e4c   accuracy gen                   62.50
```

更多教程请查看我们的👉[文档](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/)

## 🔜 即将推出

- [x] **\[已完成]** ✅ AISBench完成全面重构，支持在AISBench框架下🔌插件化集成前沿测试基准，以应对业界愈发复杂多样化的测试任务；并且显著提高易用性。
- [x] **\[已完成]** 持续扩展业界前沿的多模态测评能力，支持更多多模态数据集和评测场景。
- [x] **\[已完成]** 提供业界主流Agent测评能力，支持Agent任务链和工具调用等复杂场景的评测。

## 🤝 致谢

- 本项目代码基于🔗 [OpenCompass](https://github.com/open-compass/opencompass)做拓展开发。
- 本项目部分数据集和提示词实现修改自[simple-evals](https://github.com/openai/simple-evals)。
- 本项目代码中打点的性能指标与[VLLM Benchmark](https://github.com/vllm-project/vllm/tree/main/benchmarks)对齐。
- 本项目的BFCL函数调用能力评估功能基于 [Berkeley Function Calling Leaderboard (BFCL)](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) 实现。

<p align="right"><a href="#top">🔝Back to top</a></p>
