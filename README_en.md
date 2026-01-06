<div align="center">
  <br />
  <br />

  # **AISBench Benchmark Tool**
  #### A Testing Benchmark Tool for the Artificial Intelligence Field
  <!-- Use a separator line instead of a background -->
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
[🌐 Official Website](https://www.aisbench.com) |
[📖 Tool Documentation](https://ais-bench-benchmark-rf.readthedocs.io/en/latest/) |
[👨‍💻 Developer Documentation](https://ais-bench-benchmark-rf.readthedocs.io/en/latest/develop_guide/contributing.html) |
[🔥 Latest Updates](#-latest-updates)|
[🤔 Report Issues](https://github.com/AISBench/benchmark/issues/new/choose)
<br><br>[简体中文](README.md) | English
</div>

> ❗<span style="color: red;"><b>Important</b></span>
>
> **⭐️Star this project** to get the latest updates of AISBench Benchmark Tool in real time!

## 🔥 Latest Updates
- **\[2025.12.19\]** 🎉 **AISBench Architecture Refactoring Completed!**
  - ✨ **Architecture Upgrade**: Comprehensive refactoring of cli, models, inferencer, and tasks components, supporting rapid integration of new test benchmarks. See 📚 [Developer Documentation](https://ais-bench-benchmark-rf.readthedocs.io/en/latest/develop_guide/contributing.html) for details!
  - 🖥️ **Task Management Interface**: Brand new task UI management interface that supports simultaneous monitoring of detailed execution status for each task, including task name, progress, time cost, status, log path, extended parameters, etc., making task execution status clear at a glance!
  - ⚡ **Enhanced Parallel Execution**: Extended multi-task parallel functionality, supporting parallel execution of multiple performance or accuracy evaluation tasks, significantly improving evaluation efficiency!
  - 📊 **15+ New Evaluation Benchmarks**: Added docvqa, infovqa, ocrbench_v2, omnidocbench, mmmu, mmmu_pro, mmstar, videomme, FewCLUE series, dapo_math, leval and other multimodal and text evaluation benchmarks!
  - 🤖 **New Model Support**: Added vllm/vllm-ascend VL offline inference model support!
  - 🔧 **Feature Enhancements**: Added streaming inference switch, custom URL path, API key configuration; supports API model inference warmup; supports custom multimodal dataset performance evaluation; some datasets support service-based PPL (perplexity) evaluation and many other features!
  - 🏗️ **Infrastructure Optimization**: Refactored local models and api models components, unified streaming and non-streaming implementations; refactored inferencer component, adopted multi-process + coroutine calling approach to improve concurrency; optimized test result data format to jsonl, reducing IO pressure; adopted error codes for unified error information management and more!
- **\[2025.11.25\]** Support for PPL (Perplexity-based) mode accuracy evaluation for service-deployed models.🔥🔥🔥
- **\[2025.9.08\]** Support for 📚[Simulating Real Business Traffic](https://ais-bench-benchmark-rf.readthedocs.io/en/latest/advanced_tutorials/rps_distribution.html): By controlling fluctuations in request sending rates, perceive the performance evaluation results of service deployment in simulated real-world scenarios! 🔥🔥🔥

- **\[2025.8.28\]** Support for 📚[Multiple Independent Repeated Inference Accuracy Scenarios](https://ais-bench-benchmark-rf.readthedocs.io/en/latest/base_tutorials/scenes_intro/accuracy_benchmark.html#id12), calculating accuracy metrics across different dimensions such as pass@k/cons@k/avg@n! 🔬🔬🔬

- **\[2025.8.19\]**
  - Added a dedicated model configuration for Function Call: [vllm_api_function_call_chat](ais_bench/benchmark/configs/models/vllm_api/vllm_api_function_call_chat.py), supporting [BFCL Function Calling Capability Evaluation](ais_bench/benchmark/configs/datasets/BFCL/README_en.md) 🔥🔥🔥
  - Provided [Performance Test Specification Documentation](https://ais-bench-benchmark-rf.readthedocs.io/en/latest/base_tutorials/scenes_intro/performance_benchmark.html#id25) supported by the tool, optimizing memory usage and performance calculation of the tool in inference cluster scenarios. For the maximum specification scenario (250K requests, input/output tokens: 4K/4K), memory usage is reduced by 60% (now less than 64GB), and performance result calculation efficiency is improved by 20x. 🚀🚀🚀

- **\[2025.7.15\]**
  - Supported service deployment performance evaluation and visualization for multi-turn dialogue datasets such as [sharegpt](ais_bench/benchmark/configs/datasets/sharegpt/README_en.md) and [mtbench](ais_bench/benchmark/configs/datasets/mtbench/README_en.md). See 📚[Multi-Turn Dialogue Evaluation Guide](https://ais-bench-benchmark-rf.readthedocs.io/en/latest/advanced_tutorials/multiturn_benchmark.html) for evaluation methods! 🔥🔥🔥
  - Enabled the use of [custom datasets](https://ais-bench-benchmark-rf.readthedocs.io/en/latest/advanced_tutorials/custom_dataset.html) in performance evaluation scenarios, supporting the specification of maximum output length at the request granularity! 🔥🔥🔥

- **\[2025.6.19\]** Support for 📚[Performance Evaluation Result Visualization](https://ais-bench-benchmark-rf.readthedocs.io/en/latest/base_tutorials/results_intro/performance_visualization.html) to help locate performance bottlenecks of inference services! 🔥🔥🔥

- **\[2025.6.12\]** Supported accuracy and performance evaluation for multimodal datasets including [textvqa](ais_bench/benchmark/configs/datasets/textvqa/README_en.md), [videobench](ais_bench/benchmark/configs/datasets/videobench/README_en.md), and [vocalsound](ais_bench/benchmark/configs/datasets/vocalsound/README_en.md)! 🔥🔥🔥

- **\[2025.6.6\]** AISBench supports steady-state performance evaluation to obtain the true optimal performance of the system. Refer to 📚 [Service Deployment Steady-State Performance Test](doc/users_guide/stable_stage.md) to get started quickly! 🔥🔥🔥

- **\[2025.5.16\]** Supported performance evaluation for high concurrency service deployment (up to 30,000+ concurrent requests). 📚 [Performance Metrics](doc/users_guide/performance_metric.md) are aligned with 🔗 [vllm benchmark](https://github.com/vllm-project/vllm/tree/main/benchmarks). See 📚 [Service Deployment Performance Evaluation Guide](https://ais-bench-benchmark-rf.readthedocs.io/en/latest/base_tutorials/scenes_intro/performance_benchmark.html) for details! 🔥🔥🔥

- **\[2025.4.30\]** Accuracy evaluation supports resuming from breakpoints and re-evaluating failed cases, significantly improving the robustness of accuracy evaluation. Refer to 📚 [Resume from Interruption & Re-evaluate Failed Cases](https://ais-bench-benchmark-rf.readthedocs.io/en/latest/base_tutorials/scenes_intro/accuracy_benchmark.html#id10) to get started quickly! 🔥🔥🔥

- **\[2025.4.15\]** Optimized the request sending method from fixed-batch to continuous batch mode, significantly improving accuracy evaluation efficiency! 🔥🔥🔥

- **\[2025.4.12\]** Supported merging all multi-file datasets (such as MMLU, Ceval) into a single dataset task for accuracy evaluation. See 📚 [Merge Multi-File Datasets](https://ais-bench-benchmark-rf.readthedocs.io/en/latest/base_tutorials/scenes_intro/accuracy_benchmark.html#id11) for details! 🔥🔥🔥


## 🌏 Introduction
AISBench Benchmark is a model evaluation tool built based on [OpenCompass](https://github.com/open-compass/opencompass). It is compatible with OpenCompass's configuration system, dataset structure, and model backend implementation, and on this basis, extends support for service-deployed models.

Currently, AISBench supports evaluation scenarios for two major types of inference tasks:

🔍 [Accuracy Evaluation](https://ais-bench-benchmark-rf.readthedocs.io/en/latest/base_tutorials/scenes_intro/home.html#id2): Supports accuracy verification of service-deployed models and local models on various question-answering and reasoning benchmark datasets, covering text, multimodal and other scenarios.

🚀 [Performance Evaluation](https://ais-bench-benchmark-rf.readthedocs.io/en/latest/base_tutorials/scenes_intro/home.html#id5): Supports latency and throughput evaluation of service-deployed models, as well as extreme performance testing under stress test scenarios, supporting steady-state performance evaluation and real business traffic simulation.


## 🛠️ Tool Installation
✅ Environment Requirements

**Python Version**: Only Python **3.10**, **3.11** or **3.12** is supported.

Python 3.9 and below are not supported, nor are Python 3.13 and above.

**It is recommended to use Conda for environment management** to avoid dependency conflicts:
```shell
conda create --name ais_bench python=3.10 -y
conda activate ais_bench
```

📦 Installation Method (Source Code Installation)

AISBench currently only provides source code installation. Ensure the installation environment has internet access:
```shell
git clone https://github.com/AISBench/benchmark.git
cd benchmark/
pip3 install -e ./ --use-pep517
```
This command will automatically install core dependencies.
Execute `ais_bench -h`. If the help information for all command-line options of the AISBench evaluation tool is printed, the installation is successful.

⚙️ Service Deployment Framework Support (Optional)

If you need to evaluate service-deployed models (such as vLLM, Triton, etc.), install additional dependencies:
```shell
pip3 install -r requirements/api.txt
pip3 install -r requirements/extra.txt
```
⚙️ Huggingface Multi-modal Model/ vLLM Offline Inference Support（Optional）
```shell
pip3 install -r requirements/hf_vl_dependency.txt
```
🔗 Berkeley Function Calling Leaderboard (BFCL) Evaluation Support
```shell
pip3 install -r requirements/datasets/bfcl_dependencies.txt --no-deps
```

**Important Note**: Since `bfcl_eval` automatically installs the `pathlib` library (which is already built into Python 3.5+ environments), use the `--no-deps` parameter to skip automatic installation of additional dependencies to avoid version conflicts.

For further configuration or to initiate evaluation tasks using CLI or Python scripts, refer to the [Quick Start Guide](#quick-start).


## ❌ Tool Uninstallation
To uninstall AISBench Benchmark, execute the following command:
```shell
pip3 uninstall ais_bench_benchmark
```


## 🚀 Quick Start
### Command Meaning
A single or multiple evaluation tasks executed by an AISBench command are defined by a combination of model tasks (single or multiple), dataset tasks (single or multiple), and result presentation tasks (single). Other command-line options of AISBench specify the scenario of the evaluation task (accuracy evaluation scenario, performance evaluation scenario, etc.). Take the following AISBench command as an example:
```shell
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --summarizer example
```
This command does not specify other command-line options, so it defaults to an accuracy evaluation task, where:
- `--models` specifies the model task: the `vllm_api_general_chat` model task.
- `--datasets` specifies the dataset task: the `demo_gsm8k_gen_4_shot_cot_chat_prompt` dataset task.
- `--summarizer` specifies the result presentation task: the `example` result presentation task (if `--summarizer` is not specified, the `example` task is used by default for accuracy evaluation scenarios). It is generally used as default and does not need to be specified in the command line (subsequent commands will omit this option).


### Task Meaning Query (Optional)
Detailed information (introduction, usage constraints, etc.) about the selected model task (`vllm_api_general_chat`), dataset task (`demo_gsm8k_gen_4_shot_cot_chat_prompt`), and result presentation task (`example`) can be queried from the following links:
- `--models`: 📚 [Service Deployment Inference Backend](https://ais-bench-benchmark-rf.readthedocs.io/en/latest/base_tutorials/all_params/models.html#id2)
- `--datasets`: 📚 [Open-Source Datasets](https://ais-bench-benchmark-rf.readthedocs.io/en/latest/base_tutorials/all_params/datasets.html#id3) → 📚 [Detailed Introduction](ais_bench/benchmark/configs/datasets/demo/README_en.md)
- `--summarizer`: 📚 [Result Summary Tasks](https://ais-bench-benchmark-rf.readthedocs.io/en/latest/base_tutorials/all_params/summarizer.html)


# Modification of Configuration Files Corresponding to Tasks
Each model task, dataset task, and result presentation task corresponds to a configuration file. The content of these configuration files needs to be modified before running the command. The paths of these configuration files can be queried by adding `--search` to the original AISBench command. For example:
```shell
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --search
```
> ⚠️ **Note**: Executing the command with the `search` option will print the absolute path of the configuration file corresponding to the task.

After executing the query command, you will get the following query results:
```shell
╒══════════════╤═══════════════════════════════════════╤════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╕
│ Task Type    │ Task Name                             │ Config File Path                                                                                                               │
╞══════════════╪═══════════════════════════════════════╪════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╡
│ --models     │ vllm_api_general_chat                 │ /your_workspace/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py                                 │
├──────────────┼───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ --datasets   │ demo_gsm8k_gen_4_shot_cot_chat_prompt │ /your_workspace/benchmark/ais_bench/benchmark/configs/datasets/demo/demo_gsm8k_gen_4_shot_cot_chat_prompt.py                   │
╘══════════════╧═══════════════════════════════════════╧════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╛

```

- The dataset task configuration file `demo_gsm8k_gen_4_shot_cot_chat_prompt.py` in the quick start does not require additional modifications. For an introduction to the content of the dataset task configuration file, please refer to 📚 [Configure Open-Source Datasets](https://ais-bench-benchmark-rf.readthedocs.io/en/latest/base_tutorials/all_params/datasets.html#id6)

The model configuration file `vllm_api_general_chat.py` contains configuration content related to model operation and needs to be modified according to actual conditions. The content that needs to be modified in the quick start is marked with comments.
```python
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="",                    # Specify the absolute path of the model serialized vocabulary file (generally not required for accuracy testing scenarios)
        model="",        # Specify the name of the model loaded on the server, configured according to the actual model name pulled by the VLLM inference service (configure as an empty string to get it automatically)
        stream=False,
        request_rate=0,           # Request sending frequency: send 1 request to the server every 1/request_rate seconds; if less than 0.1, all requests are sent at once
        retry=2,                  # Maximum number of retries for each request
        api_key="",               # Custom API key, default is an empty string
        host_ip="localhost",      # Specify the IP of the inference service
        host_port=8080,           # Specify the port of the inference service
        url="",                     # Custom URL path for accessing the inference service (needs to be configured when the base URL is not a combination of http://host_ip:host_port; after configuration, host_ip and host_port will be ignored)
        max_out_len=512,          # Maximum number of tokens output by the inference service
        batch_size=1,               # Maximum concurrency for sending requests
        trust_remote_code=False,    # Whether the tokenizer trusts remote code, default is False;
        generation_kwargs=dict(   # Model inference parameters, configured with reference to the VLLM documentation; the AISBench evaluation tool does not process them and attaches them to the sent request
            temperature=0.01,
            ignore_eos=False,
        )
    )
]
```
# Execution Command
After modifying the configuration files, execute the command to start the service-based accuracy evaluation:
```bash
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt
```
## View Task Execution Details
After executing the AISBench command, the task management interface will display task execution status in real-time on the command line (press the "P" key to pause/resume refreshing for copying dashboard information, press "P" again to continue refreshing). The task management interface supports simultaneous monitoring of detailed execution status for multiple tasks, including task name, progress, time cost, status, log path, extended parameters and other information. For example:
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

The detailed logs of task execution will be continuously saved to the default output path, which is displayed on the real-time refreshed dashboard as `Log Path`. The `Log Path` (`logs/infer/vllm-api-general-chat/demo_gsm8k.out`) is under the `Base path` (`outputs/default/20250628_151326`). Taking the above dashboard information as an example, the path of the detailed log of task execution is:
```shell
# {Base path}/{Log Path}
outputs/default/20250628_151326/logs/infer/vllm-api-general-chat/demo_gsm8k.out
```

> 💡 If you want the detailed logs to be printed directly during execution, you can add `--debug` to the execution command:
`ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --debug`


The `Base path` (`outputs/default/20250628_151326`) contains all the task execution details. After the command execution is completed, all execution details are as follows:
```shell
20250628_151326/
├── configs # A combined configuration of the configuration files corresponding to model tasks, dataset tasks, and structure presentation tasks
│   └── 20250628_151326_29317.py
├── logs # Logs during execution; if --debug is added to the command, no process logs will be saved (all logs are printed directly)
│   ├── eval
│   │   └── vllm-api-general-chat
│   │       └── demo_gsm8k.out # Logs of the accuracy evaluation process based on the inference results in the predictions/ folder
│   └── infer
│       └── vllm-api-general-chat
│           └── demo_gsm8k.out # Logs of the inference process
├── predictions
│   └── vllm-api-general-chat
│       └── demo_gsm8k.json # Inference results (all outputs returned by the inference service)
├── results
│   └── vllm-api-general-chat
│       └── demo_gsm8k.json # Original scores calculated by accuracy evaluation
└── summary
    ├── summary_20250628_151326.csv # Final accuracy scores (in table format)
    ├── summary_20250628_151326.md # Final accuracy scores (in Markdown format)
    └── summary_20250628_151326.txt # Final accuracy scores (in text format)
```
> ⚠️ **Note**: The content of the saved task execution details varies in different evaluation scenarios. For details, please refer to the guide for the specific evaluation scenario.


#### Output Results
Since there are only 8 data entries, results will be generated quickly. An example of the output is shown below:
```bash
dataset                 version  metric   mode  vllm_api_general_chat
----------------------- -------- -------- ----- ----------------------
demo_gsm8k              401e4c   accuracy gen                   62.50
```

For more tutorials, please refer to our 👉[Documentation](https://ais-bench-benchmark-rf.readthedocs.io/en/latest/)


## 🔜 Coming Soon
- [x] **\[Completed\]** ✅ AISBench has completed comprehensive refactoring, supporting plug-and-play integration of cutting-edge testing benchmarks within the AISBench framework to address the increasingly complex and diverse testing tasks in the industry; while significantly improving usability.
- [ ] **\[Planned\]** Continue to expand industry-leading multimodal evaluation capabilities, supporting more multimodal datasets and evaluation scenarios.
- [ ] **\[Planned\]** Provide evaluation capabilities for mainstream industry Agents, supporting Agent task chains and tool calling in complex scenarios.


## 🤝 Acknowledgements
- The code of this project is developed based on 🔗 [OpenCompass](https://github.com/open-compass/opencompass) with extensions.
- Some datasets and prompt implementations in this project are modified from [simple-evals](https://github.com/openai/simple-evals).
- The performance metrics tracked in this project’s code are aligned with [VLLM Benchmark](https://github.com/vllm-project/vllm/tree/main/benchmarks).
- The BFCL function calling capability evaluation feature of this project is implemented based on the [Berkeley Function Calling Leaderboard (BFCL)](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard).


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