# 计划：提升自定义配置文件方式的文档可见性与完善度

## 目标

让用户充分感知并理解"自定义配置文件"这一强大能力，它本质上可以覆盖 AISBench 所有主要评测场景，并且支持完整的 Python 语法灵活性。

---

## 一、完善 `run_custom_config.md` 文档

### 1.1 新增"为什么使用自定义配置文件"章节
- 对比 CLI 参数方式（`--models` + `--datasets`）与自定义配置文件方式的优劣
- 强调自定义配置文件的优势：
  - 可复用，一次编写多次执行
  - 支持 Python 全部语法（循环、条件、函数、列表推导等）
  - 精确控制模型-数据集组合（`model_dataset_combinations`）
  - 可在一个文件中组合任意数量的模型和数据集
  - 支持自定义 `infer`/`eval` 分区器和运行器配置
  - 方便版本管理和团队共享

### 1.2 新增"配置文件即 Python 脚本"章节
- 明确说明：配置文件本质上是 Python 脚本，支持所有 Python 语法
- 提供实用示例：
  - 使用 for 循环批量生成模型配置（如不同端口、不同 IP）
  - 使用列表推导式批量修改数据集 abbr
  - 使用条件判断动态选择配置
  - 使用 `.copy()` 复用配置并修改特定字段
  - 从外部文件读取配置参数

### 1.3 新增"配置文件完整变量参考"章节
- `models`：模型配置列表，每个元素为 dict
- `datasets`：数据集配置列表，每个元素为 dict
- `summarizer`：汇总器配置 dict
- `model_dataset_combinations`：可选，精确控制模型-数据集配对
- `work_dir`：输出目录
- `infer`：可选，推理阶段的分区器和运行器配置
- `eval`：可选，评估阶段的分区器和运行器配置
- 说明每种变量的类型、必填字段、可选字段

### 1.4 新增"各场景自定义配置文件示例"章节
为每个主要场景提供完整的自定义配置文件示例：

| 场景 | 说明 |
|------|------|
| 服务化精度测评 | API 模型 + 开源数据集精度测评 |
| 纯模型精度测评 | HuggingFace 本地模型精度测评 |
| 服务化性能测评 | API 流式模型 + 数据集性能测评 |
| 合成数据集性能测评 | API 流式模型 + synthetic 数据集性能测评 |
| 多模型多数据集组合 | 多个模型 × 多个数据集的笛卡尔积测评 |
| 自定义模型-数据集配对 | 使用 `model_dataset_combinations` 精确配对 |
| 裁判模型测评 | 被测模型 + 裁判模型的精度测评 |
| 稳态性能测评 | 使用 `stable_stage` summarizer 的性能测评 |
| 多轮对话性能测评 | ShareGPT/MTBench 多轮对话性能测评 |
| 自定义数据集测评 | 使用自定义 CSV/JSONL 数据集的测评 |
| 多模态测评 | 多模态数据集 + 多模态模型测评 |

### 1.5 完善现有内容
- 更新"预设自定义配置文件样例列表"表格，补充新增的样例文件
- 修正文档中可能存在的过时或不准确描述

---

## 二、在整个文档体系中接入自定义配置文件引用

### 2.1 修改 `index.rst`（文档首页）
- 在"推荐上手路径"中，将自定义配置文件的推荐提前或加重
- 在 toctree 中可以考虑将 `run_custom_config` 提升到更显眼的位置，或添加一个独立的引导章节

### 2.2 修改 `get_started/quick_start.md`（快速入门）
- 在执行命令章节末尾添加"💡 进阶提示"，引导用户了解自定义配置文件方式
- 说明：当需要重复执行或复杂组合时，推荐使用自定义配置文件

### 2.3 修改各场景文档，添加"通过自定义配置文件实现"小节

在以下文档的适当位置（建议在"主要功能场景"之后或每个功能场景末尾）添加引用：

| 文档 | 添加位置 | 引用内容 |
|------|---------|---------|
| `base_tutorials/scenes_intro/accuracy_benchmark.md` | 多任务测评章节后 | 说明如何用自定义配置文件实现多任务精度测评 |
| `base_tutorials/scenes_intro/accuracy_benchmark_local.md` | 主要功能章节后 | 说明如何用自定义配置文件实现纯模型精度测评 |
| `base_tutorials/scenes_intro/performance_benchmark.md` | 多任务测评章节后 | 说明如何用自定义配置文件实现多任务性能测评 |
| `advanced_tutorials/multiturn_benchmark.md` | 快速入门章节后 | 说明如何用自定义配置文件实现多轮对话测评 |
| `advanced_tutorials/synthetic_dataset.md` | 使用指导章节后 | 说明如何用自定义配置文件实现合成数据集测评 |
| `advanced_tutorials/custom_dataset.md` | 配置文件方式章节后 | 说明如何用自定义配置文件实现自定义数据集测评 |
| `advanced_tutorials/judge_model_evaluate.md` | 快速上手章节后 | 说明如何用自定义配置文件实现裁判模型测评 |
| `advanced_tutorials/stable_stage.md` | 快速入门章节后 | 说明如何用自定义配置文件实现稳态性能测评 |
| `advanced_tutorials/rps_distribution.md` | 文件配置章节后 | 说明 RPS 分布控制参数在自定义配置文件中同样适用 |
| `extended_benchmark/agent/swe_bench.md` | 适当位置 | 说明已有预设配置文件样例 |
| `extended_benchmark/agent/tau2_bench.md` | 适当位置 | 说明已有预设配置文件样例 |
| `extended_benchmark/agent/harbor_bench.md` | 适当位置 | 说明已有预设配置文件样例 |
| `extended_benchmark/lmm_generate/vbench.md` | 适当位置 | 说明已有预设配置文件样例 |
| `extended_benchmark/lmm_generate/gedit_bench.md` | 适当位置 | 说明已有预设配置文件样例 |
| `best_practices/practice_nvidia.md` | 适当位置 | 说明推荐使用自定义配置文件 |
| `best_practices/practice_ascend.md` | 适当位置 | 说明推荐使用自定义配置文件 |

每个引用小节的统一格式：
```markdown
### 通过自定义配置文件实现

> 💡 上述场景同样可以通过 [自定义配置文件](../advanced_tutorials/run_custom_config.md) 实现，将模型、数据集、summarizer 等配置写入一个 Python 文件，一次编写、多次复用。详见 [自定义配置文件运行AISBench](../advanced_tutorials/run_custom_config.md#各场景自定义配置文件示例) 中对应场景的示例。
```

---

## 三、强调配置文件支持 Python 全部语法

### 3.1 在 `run_custom_config.md` 中重点强调
- 新增独立章节"配置文件即 Python 脚本"（如 1.2 所述）
- 在文档开头添加醒目的提示框

### 3.2 在 `index.rst` 中提及
- 在推荐上手路径中增加一句描述，强调配置文件的灵活性

### 3.3 在各场景文档的引用小节中提及
- 在"通过自定义配置文件实现"小节中简要提及 Python 语法的灵活性

---

## 四、在代码仓库中预置不同场景的自定义配置文件样例

在 `ais_bench/configs/` 下新增/完善以下样例文件：

### 4.1 精度测评场景（`ais_bench/configs/api_examples/`）

| 文件名 | 说明 |
|--------|------|
| `demo_infer_vllm_api.py` | ✅ 已有，需检查完善 |
| `infer_vllm_api_general.py` | ✅ 已有 |
| `infer_vllm_api_general_chat.py` | ✅ 已有 |
| `infer_vllm_api_stream_chat.py` | ✅ 已有 |
| `infer_mindie_stream_api_general.py` | ✅ 已有 |
| `infer_vllm_api_multi_model_multi_dataset.py` | 🆕 多模型多数据集精度测评示例 |
| `infer_vllm_api_with_model_dataset_combinations.py` | 🆕 自定义模型-数据集配对示例 |
| `infer_vllm_api_with_judge_model.py` | 🆕 裁判模型测评示例 |

### 4.2 性能测评场景（`ais_bench/configs/api_examples/`）

| 文件名 | 说明 |
|--------|------|
| `demo_infer_vllm_api_perf.py` | ✅ 已有，需检查完善 |
| `perf_vllm_api_synthetic.py` | 🆕 合成数据集性能测评示例 |
| `perf_vllm_api_stable_stage.py` | 🆕 稳态性能测评示例 |
| `perf_vllm_api_multiturn.py` | 🆕 多轮对话性能测评示例 |
| `perf_vllm_api_custom_dataset.py` | 🆕 自定义数据集性能测评示例 |
| `perf_vllm_api_rps_distribution.py` | 🆕 RPS 分布控制性能测评示例 |

### 4.3 纯模型测评场景（`ais_bench/configs/hf_example/`）

| 文件名 | 说明 |
|--------|------|
| `infer_hf_chat_model.py` | ✅ 已有 |
| `infer_hf_base_model.py` | ✅ 已有 |
| `infer_hf_multi_model_multi_dataset.py` | 🆕 多模型多数据集纯模型测评示例 |

### 4.4 多模态场景（`ais_bench/configs/lmm_example/`）

| 文件名 | 说明 |
|--------|------|
| `multi_device_run_qwen_image_edit.py` | ✅ 已有 |
| `infer_lmm_multi_dataset.py` | 🆕 多模态多数据集测评示例 |

### 4.5 Agent 场景（已有，需检查）

| 文件名 | 说明 |
|--------|------|
| `agent_example/harbor_terminal_bench_2_task.py` | ✅ 已有 |
| `agent_example/tau2_bench_task.py` | ✅ 已有 |
| `swe_bench_examples/*.py` | ✅ 已有 |
| `swe_bench_pro_examples/*.py` | ✅ 已有 |

### 4.6 VBench 场景（已有，需检查）

| 文件名 | 说明 |
|--------|------|
| `vbench_examples/eval_vbench_standard.py` | ✅ 已有 |
| `vbench_examples/eval_vbench_custom.py` | ✅ 已有 |

### 4.7 更新 `all_dataset_configs.py`
- 确保所有常用数据集的导入都已包含
- 添加注释说明每个导入对应的数据集

---

## 五、实施步骤

### 步骤 1：完善 `run_custom_config.md`
- 新增"为什么使用自定义配置文件"章节
- 新增"配置文件即 Python 脚本"章节（含 Python 语法示例）
- 新增"配置文件完整变量参考"章节
- 新增"各场景自定义配置文件示例"章节（含所有主要场景的完整示例代码）
- 更新"预设自定义配置文件样例列表"

### 步骤 2：创建新的预设配置文件样例
- 在 `ais_bench/configs/api_examples/` 下创建 4.1 和 4.2 列出的新文件
- 在 `ais_bench/configs/hf_example/` 下创建 4.3 列出的新文件
- 在 `ais_bench/configs/lmm_example/` 下创建 4.4 列出的新文件
- 更新 `all_dataset_configs.py`

### 步骤 3：在文档体系中添加交叉引用
- 修改 `index.rst`
- 修改 `get_started/quick_start.md`
- 修改 `base_tutorials/scenes_intro/accuracy_benchmark.md`
- 修改 `base_tutorials/scenes_intro/accuracy_benchmark_local.md`
- 修改 `base_tutorials/scenes_intro/performance_benchmark.md`
- 修改 `advanced_tutorials/multiturn_benchmark.md`
- 修改 `advanced_tutorials/synthetic_dataset.md`
- 修改 `advanced_tutorials/custom_dataset.md`
- 修改 `advanced_tutorials/judge_model_evaluate.md`
- 修改 `advanced_tutorials/stable_stage.md`
- 修改 `advanced_tutorials/rps_distribution.md`
- 修改 `extended_benchmark/agent/swe_bench.md`
- 修改 `extended_benchmark/agent/tau2_bench.md`
- 修改 `extended_benchmark/agent/harbor_bench.md`
- 修改 `extended_benchmark/lmm_generate/vbench.md`
- 修改 `extended_benchmark/lmm_generate/gedit_bench.md`
- 修改 `best_practices/practice_nvidia.md`
- 修改 `best_practices/practice_ascend.md`

### 步骤 4：验证
- 检查所有新增/修改的 Markdown 文件中的内部链接是否正确
- 确保新增的配置文件样例语法正确、可执行
- 确保文档中的代码示例与实际代码逻辑一致
