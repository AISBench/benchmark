# AISBench 软件设计文档：IFBench 数据集

---

## 文档信息

| 项目 | 内容 |
|------|------|
| 文档标题 | AISBench IFBench 数据集软件设计文档 |
| 版本 | v1.0 |
| 日期 | 2026-07-02 |
| 涉及模块 | `benchmark/ais_bench/benchmark/datasets/ifbench/` |

---

## 目录

1. [背景描述](#1-背景描述)
2. [设计方案](#2-设计方案)
   - 2.1 [代码架构](#21-代码架构)
   - 2.2 [数据加载](#22-数据加载)
   - 2.3 [Prompt 构造](#23-prompt-构造)
   - 2.4 [响应解析与打分](#24-响应解析与打分)
   - 2.5 [评测方法与四项指标](#25-评测方法与四项指标)
3. [使用说明](#3-使用说明)
   - 3.1 [数据集下载](#31-数据集下载)
   - 3.2 [配置文件修改](#32-配置文件修改)
   - 3.3 [执行命令评测](#33-执行命令评测)
4. [测试用例说明](#4-测试用例说明)
   - 4.1 [测试用例设计](#41-测试用例设计)
   - 4.2 [代码覆盖率分析](#42-代码覆盖率分析)

---

## 1. 背景描述

**IFBench**（Instruction Following Benchmark）是由 Allen Institute for AI（AllenAI）开发的指令遵循评测基准。该基准旨在评估 AI 模型在遵循新颖、具有挑战性且多样化的可验证指令方面的可靠性，特别关注模型的**分布外泛化能力**（out-of-domain generalization）。

**核心特点：**

| 属性 | 值 |
|------|-----|
| 基准名称 | IFBench / ifbench |
| 数据集来源 | allenai/IFBench_test（HuggingFace） |
| 任务类型 | 指令遵循评测（Instruction Following Evaluation） |
| 样本总数 | 300 |
| 平均 Prompt 长度 | 343.41 字符 |
| Prompt 长度范围 | 50 ~ 904 字符 |
| 可验证约束数 | 58 类（手动策划） |
| 约束分类 | 计数类、格式类、词汇类、句子结构类、标点类等 |

**58 类约束示例：**

- **计数类**：`count:word_count_range`（单词数量范围）、`count:unique_word_count`（唯一单词数）、`count:numbers`（数字个数）、`count:person_names`（人名计数）、`count:pronouns`（代词计数）、`count:conjunctions`（连词计数）
- **格式类**：`format:newline`（换行格式）、`format:parentheses`（嵌套括号）、`format:quotes`（嵌套引号）、`format:emoji`（Emoji 句子）、`format:options`（选项格式）
- **词汇类**：`words:alphabet`（字母循环）、`words:vowel`（单元音段落）、`words:palindrome`（回文检测）、`words:start_verb`（动词开头）、`words:consonants`（辅音簇）、`words:prime_lengths`（质数长度）
- **句子类**：`sentence:keyword`（包含关键词）、`sentence:increment`（递增词数）、`sentence:alliteration_increment`（递增头韵）、`sentence:last_first`（尾词开头）
- **比例类**：`ratio:stop_words`（停用词比例）、`ratio:sentence_type`（句子类型比例）、`ratio:overlap`（N-Gram 重叠率）

**评测模式：**

IFBench 支持两种评测模式：

- **Strict（严格模式）**：直接检查模型原始输出是否满足每条指令约束
- **Loose（宽松模式）**：对输出做 8 种变体处理（去除首行/尾行/首尾行、去除星号及其组合），只要任一变体通过即视为满足

**设计目标：** 专门为解决数据污染（data contamination）问题而设计，约束均为程序化验证，无需人工评判或 LLM Judge。

---

## 2. 设计方案

### 2.1 代码架构

```
benchmark/ais_bench/benchmark/datasets/ifbench/
├── __init__.py               # 导出 IFBenchDataset, IFBenchEvaluator
├── ifbench.py                # 数据集加载 + Evaluator + strict/loose 检测函数
├── instructions.py           # 58 个指令检测器实现
├── instructions_registry.py  # INSTRUCTION_DICT 注册表
└── data/
    └── train-00000-of-00001.parquet  # 数据集文件
```

**核心类与函数：**

| 名称 | 类型 | 职责 |
|------|------|------|
| `IFBenchDataset` | 类 (继承 BaseDataset) | 加载 parquet 数据，构建 Dataset |
| `IFBenchEvaluator` | 类 (继承 BaseEvaluator) | 执行 strict/loose 评测，计算四项指标 |
| `InputExample` | dataclass | 单条评测输入：key、instruction_id_list、prompt、kwargs |
| `OutputExample` | dataclass | 单条评测输出：follow_all_instructions、follow_instruction_list |
| `test_instruction_following_strict()` | 函数 | 严格模式指令遵循检测 |
| `test_instruction_following_loose()` | 函数 | 宽松模式指令遵循检测（8 种响应变体） |

---

### 2.2 数据加载

#### 加载流程

```
parquet 文件 → Dataset.from_parquet() → 遍历每条记录 → 提取 prompt + reference → Dataset.from_list() → 返回
```

具体步骤：

1. **路径解析**：通过 `get_data_path(path, local_mode=True)` 获取本地 parquet 文件的绝对路径。数据集路径在配置中设置为 `ais_bench/datasets/ifbench/data/train-00000-of-00001.parquet`
2. **读取 parquet**：使用 `Dataset.from_parquet(path)` 直接加载 parquet 文件。相比 HuggingFace `load_dataset()`，此方法更高效且避免了路径歧义问题
3. **数据转换**：遍历每条记录，提取 `prompt`（发送给模型的指令文本）和 `reference`（包含 key、instruction_id_list、kwargs 的完整字典）
4. **返回**：`Dataset.from_list()` 返回 HuggingFace Dataset 格式，包含两列

#### 数据字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| prompt | str | 发送给模型的指令 prompt，包含需遵守的约束描述 |
| reference.key | int | 样本唯一标识 |
| reference.instruction_id_list | List[str] | 该样本需满足的约束 ID 列表（如 `["count:word_count_range", "words:start_verb"]`） |
| reference.kwargs | List[Dict] | 每个约束的参数配置（如 `[{"lower": 50, "upper": 50}, {}]`） |

#### 核心代码

```python
@staticmethod
def load(path: str, name: str = 'default'):
    path = get_data_path(path, local_mode=True)
    logger.info(f"Loading IFBench dataset from: {path}")
    from datasets import Dataset
    dataset = Dataset.from_parquet(path)
    raw_data = []
    for i in range(len(dataset)):
        item = dataset[i]
        prompt = item['prompt']
        raw_data.append({
            'prompt': prompt,
            'reference': item,
        })
    logger.info(f"IFBench dataset loaded: {len(raw_data)} samples")
    return Dataset.from_list(raw_data)
```

---

### 2.3 Prompt 构造

IFBench 的 Prompt 构造采用**直通模式**——数据集中的 `prompt` 字段已包含完整的指令文本（含约束描述），通过 `PromptTemplate(template='{prompt}')` 直接透传给模型。

**配置**（`ifbench_0_shot_gen_str.py`）：

```python
ifbench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{prompt}',       # 直通模板，不做任何加工
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
```

```
输入: dataset[i]['prompt']
  → "Write a paragraph containing exactly 50 words. The paragraph must start with the word 'Technology'."
输出: 直接发送给 LLM，不做任何修改
```

---

### 2.4 响应解析与打分

LLM 返回文本后，IFBench 立即进入 **Score 计算**，无需 LLM Judge。核心流程是对每条预测分别执行 Strict 和 Loose 两种检测。

#### Strict 模式检测

`test_instruction_following_strict(inp, response)` 流程：

```
遍历 instruction_id_list:
  1. 从 INSTRUCTION_DICT 查找对应 Checker 类
  2. 实例化: instruction = CheckerClass(instruction_id)
  3. 清理 kwargs: 移除 None 值
  4. 构建描述: instruction.build_description(**kwargs)
  5. 获取参数: args = instruction.get_instruction_args()
  6. 如果 args 含 'prompt' → instruction.build_description(prompt=inp.prompt)
  7. 执行检测: instruction.check_following(response)
  8. 记录 True/False 到 is_following_list

返回 OutputExample(follow_all_instructions=all(is_following_list), follow_instruction_list=is_following_list)
```

#### Loose 模式检测

`test_instruction_following_loose(inp, response)` 在严格检查之前，先生成 **8 种响应变体**：

| 序号 | 变体 | 说明 |
|------|------|------|
| 0 | response | 原始响应 |
| 1 | revised_response | 去除全部 `*` |
| 2 | response_remove_first | 去除首行 |
| 3 | response_remove_last | 去除尾行 |
| 4 | response_remove_both | 去除首尾行 |
| 5 | revised_response_remove_first | 去首行 + 去 `*` |
| 6 | revised_response_remove_last | 去尾行 + 去 `*` |
| 7 | revised_response_remove_both | 去首尾行 + 去 `*` |

然后对每条指令遍历 8 种变体，**只要任一变体通过检测**，该指令即视为满足。

#### 指令检测器示例

```python
# instructions.py 中 58 个 Checker 的代表性示例

class WordCountRangeChecker:
    """检测单词数量是否在指定范围内"""
    def build_description(self, lower=0, upper=float("inf"), **kwargs):
        self._lower = lower
        self._upper = upper

    def check_following(self, response):
        words = response.split()
        return self._lower <= len(words) <= self._upper


class IncludeKeywordChecker:
    """检测响应中是否包含指定关键词"""
    def build_description(self, keyword="", **kwargs):
        self._keyword = keyword

    def check_following(self, response):
        return self._keyword.lower() in response.lower()


class NewLineWordsChecker:
    """检测响应是否按要求分多行输出"""
    def build_description(self, num_lines=1, **kwargs):
        self._num_lines = num_lines

    def check_following(self, response):
        lines = [l for l in response.split("\n") if l.strip()]
        return len(lines) >= self._num_lines
```

---

### 2.5 评测方法与四项指标

`IFBenchEvaluator.score()` 对全部样本执行 strict 和 loose 检测后，汇总计算 **四项指标**：

#### 指标定义

| 指标 | 公式 | 说明 |
|------|------|------|
| **Prompt-level-strict-accuracy** | `strict_prompt_correct / total × 100` | Strict 模式下，**全部**约束均满足的样本占比。反映模型在严格条件下完全遵循指令的能力 |
| **Inst-level-strict-accuracy** | `strict_inst_correct / strict_inst_total × 100` | Strict 模式下，**单条约束**被满足的占比。粒度更细，反映模型对各类约束的平均遵循程度 |
| **Prompt-level-loose-accuracy** | `loose_prompt_correct / total × 100` | Loose 模式下，**全部**约束均满足的样本占比。考虑了格式浮动（首尾行、星号等） |
| **Inst-level-loose-accuracy** | `loose_inst_correct / loose_inst_total × 100` | Loose 模式下，**单条约束**被满足的占比。Loose 模式下的平均约束遵循程度 |

#### Grade 分级

每个样本根据 Strict 和 Loose 结果分三级：

| Grade | 条件 | 含义 |
|-------|------|------|
| **strict** | Strict 全部约束满足 | 严格遵循所有指令 |
| **loose** | Strict 失败但 Loose 全部满足 | 大体遵循指令，但有格式瑕疵 |
| **none** | 两种模式均不完全满足 | 未遵循指令 |

#### 每样本 detail 结构

```python
{
    "prompt": "Write a paragraph...",     # 原始 prompt
    "pred": "Technology is great...",     # 模型预测输出
    "refer": {...},                       # 完整 reference 字典
    "is_strict_correct": True/False,      # strict 全部通过
    "is_loose_correct": True/False,       # loose 全部通过
    "is_correct": True/False,             # 等价于 is_strict_correct
    "grade": "strict" | "loose" | "none"  # 综合等级
}
```

---

## 3. 使用说明

### 3.1 数据集下载

IFBench 数据集以 **parquet 格式**存储，已集成在 AISBench 仓库中：

```
路径: benchmark/ais_bench/datasets/ifbench/data/train-00000-of-00001.parquet
```

**无需额外下载**，数据文件随代码仓库分发。文件大小约 1.5 MB，包含 300 条样本。

如需手动验证数据完整性，可使用 Python 读取：

```python
from datasets import Dataset
ds = Dataset.from_parquet("benchmark/ais_bench/datasets/ifbench/data/train-00000-of-00001.parquet")
print(f"样本数: {len(ds)}")         # 300
print(f"列: {ds.column_names}")     # ['key', 'prompt', 'instruction_id_list', 'kwargs']
```

---

### 3.2 配置文件修改

**数据集配置文件：** `benchmark/ais_bench/benchmark/configs/datasets/ifbench/ifbench_0_shot_gen_str.py`

```python
from ais_bench.benchmark.datasets.ifbench import IFBenchDataset, IFBenchEvaluator
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer

ifbench_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='reference',
)

ifbench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{prompt}',     # 直通模板
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

ifbench_eval_cfg = dict(
    evaluator=dict(type=IFBenchEvaluator),
)

ifbench_datasets = [
    dict(
        abbr='ifbench',
        type=IFBenchDataset,
        path='ais_bench/datasets/ifbench/data/train-00000-of-00001.parquet',
        reader_cfg=ifbench_reader_cfg,
        infer_cfg=ifbench_infer_cfg,
        eval_cfg=ifbench_eval_cfg,
    )
]
```

**关键参数说明：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `path` | `ais_bench/datasets/ifbench/data/train-00000-of-00001.parquet` | 相对于 cache_dir 的数据文件路径 |
| `template` | `'{prompt}'` | Prompt 模板，`{prompt}` 被替换为数据集中的 prompt 字段 |
| `retriever` | `ZeroRetriever` | 0-shot 模式，不使用上下文示例 |
| `inferencer` | `GenInferencer` | 标准生成式推理 |

**模型配置文件：** `benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py`

```python
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-general-chat",
        host_ip="localhost",
        host_port=8080,       # vLLM 服务端口
        max_out_len=512,      # 最大输出 token 数
        batch_size=1,
        retry=2,              # 请求失败重试次数
        generation_kwargs=dict(
            temperature=0.01,  # 低温度获得确定性输出
            ignore_eos=False,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
```

---

### 3.3 执行命令评测

#### 前置条件

确保 vLLM API 服务已启动：

```bash
vllm serve <model_name> --port 8080
```

#### 评测命令

**标准评测（全量 300 条）：**

```bash
ais_bench --models vllm_api_general_chat --datasets ifbench_0_shot_gen_str
```

**限制样本数（快速验证）：**

```bash
ais_bench --models vllm_api_general_chat --datasets ifbench_0_shot_gen_str --limit 10
```

**调大 max_out_len（长文本生成场景）：**

```bash
ais_bench --models vllm_api_general_chat --datasets ifbench_0_shot_gen_str \
    --model-config max_out_len=1024
```

**指定自定义模型名称：**

```bash
ais_bench --models vllm_api_general_chat --datasets ifbench_0_shot_gen_str \
    --model-config model="Qwen3-32B"
```

#### 预期输出示例

```
[ifbench][load] Loading IFBench dataset from: .../train-00000-of-00001.parquet
[ifbench][load] Sample[0] prompt: Write a paragraph containing exactly 50 words...
[ifbench][load] IFBench dataset loaded: 300 samples

[ifbench][score] ========== Score Start ==========
[ifbench][score] predictions count: 300
[ifbench][score] references count: 300

... (每条样本的 strict/loose 检测日志) ...

[ifbench][score] ============================================================
[ifbench][score] Evaluation Results:
[ifbench][score]   Prompt-level-strict-accuracy:  45.33%
[ifbench][score]   Inst-level-strict-accuracy:    72.15%
[ifbench][score]   Prompt-level-loose-accuracy:   52.67%
[ifbench][score]   Inst-level-loose-accuracy:     78.40%
[ifbench][score]   Total samples:                 300
[ifbench][score]   Final counters: prompt_strict=136/300, inst_strict=1082/1500, ...
[ifbench][score] ============================================================
[ifbench][score] ========== Score End ==========
```

---

## 4. 测试用例说明

**测试文件：** `benchmark/tests/UT/datasets/test_ifbench.py`

**测试总数：** 26 个

### 4.1 测试用例设计

#### 数据类测试（4 个）

| 测试 | 覆盖目标 | 说明 |
|------|---------|------|
| `TestInputExample::test_create` | `InputExample` dataclass | 验证字段赋值正确 |
| `TestInputExample::test_optional_kwargs` | `InputExample.kwargs` | 验证 None 值正确存储 |
| `TestOutputExample::test_create_all_following` | `OutputExample` — 全部跟随 | 验证 `follow_all_instructions=True` |
| `TestOutputExample::test_create_partial_following` | `OutputExample` — 部分跟随 | 验证 `follow_all_instructions=False` |

#### Strict 模式检测测试（6 个）

| 测试 | 覆盖目标 | 关键验证点 |
|------|---------|-----------|
| `test_word_count_range_pass` | `test_instruction_following_strict()` | 50 词恰好满足 50 词约束 → True |
| `test_word_count_range_fail` | 同上 | 1 词远小于 50 词约束 → False |
| `test_empty_response` | 空响应边界 | 空字符串 → 全部 False，不崩溃 |
| `test_multiple_instructions` | 多指令组合 | 两条约束（词数+换行）分别检测 |
| `test_output_example_contains_response` | 输出完整性 | OutputExample.response 正确透传 |
| `test_none_kwargs_cleaned` | kwargs 清洗 | None 值 kwargs 在 build_description 前被移除 |

#### Loose 模式检测测试（5 个）

| 测试 | 覆盖目标 | 关键验证点 |
|------|---------|-----------|
| `test_loose_mode_produces_output` | `test_instruction_following_loose()` | 基本烟雾测试，输出有效 OutputExample |
| `test_loose_handles_asterisks` | 星号去除逻辑 | `*hello* world` → 去星号后检测到 hello |
| `test_loose_handles_first_last_line_removal` | 行裁剪逻辑 | 中间行 3 词放在首尾含无关行的响应中 → 通过 |
| `test_loose_empty_response` | Loose+空响应边界 | 空字符串 → 全部 False |
| `test_loose_multiple_instructions` | Loose+多指令 | 词数+关键词双约束在 Loose 模式下分别验证 |

#### 数据集加载测试（3 个）

| 测试 | 覆盖目标 | 关键验证点 |
|------|---------|-----------|
| `test_load_from_mock_parquet` | `IFBenchDataset.load()` | 返回 Dataset 含正确列名和样本数 |
| `test_load_prompts_preserved` | 数据完整性 | 加载后的 prompt 与原始 parquet 记录一致 |
| `test_load_reference_contains_all_fields` | 数据格式完整性 | reference 包含 key/instruction_id_list/prompt/kwargs |

#### Evaluator 评分测试（8 个）

| 测试 | 覆盖目标 | 关键验证点 |
|------|---------|-----------|
| `test_score_basic` | `IFBenchEvaluator.score()` | 四项指标均在结果中 |
| `test_score_all_predictions_scored` | detail 完整性 | 每条预测都有对应 detail 条目 |
| `test_score_detail_keys` | detail 格式 | 包含 prompt/pred/refer/is_strict_correct/is_loose_correct/grade |
| `test_score_grade_values` | grade 逻辑 | grade 取值在 {strict, loose, none} 内 |
| `test_score_empty_predictions` | 空预测边界 | 空列表 → 准确率 0%，不崩溃 |
| `test_score_empty_string_prediction` | 空字符串预测 | `""` 预测 → 不崩溃，正常评分 |
| `test_score_accuracy_in_range` | 指标范围 | 四项指标均在 [0, 100] 内 |
| `test_score_with_origin_prompt` | origin_prompt 参数 | 可选参数 origin_prompt 正确透传到 detail |

---

### 4.2 代码覆盖率分析

#### 函数/类覆盖情况

| 模块/函数 | 是否覆盖 | 测试数量 | 覆盖路径 |
|-----------|---------|---------|---------|
| `InputExample` dataclass | ✅ 完全 | 2 | 正常创建、None kwargs |
| `OutputExample` dataclass | ✅ 完全 | 2 | 全部跟随、部分跟随 |
| `test_instruction_following_strict()` | ✅ 完全 | 6 | 通过、失败、空响应、多指令、输出完整性、None 清洗 |
| `test_instruction_following_loose()` | ✅ 完全 | 5 | 基本、星号、行裁剪、空响应、多指令 |
| `IFBenchDataset.load()` | ✅ 完全 | 3 | 列名、数据完整性、字段完整性 |
| `IFBenchEvaluator.score()` | ✅ 完全 | 8 | 指标存在、detail 完整、边界空列表/空字符串、范围、origin_prompt |
| `instructions_registry.INSTRUCTION_DICT` | ✅ 间接 | 6 | 通过 mock 指令检测器覆盖 |

#### 关键路径覆盖

| 路径类型 | 覆盖情况 |
|---------|---------|
| ✅ 正常路径 | 有效 prompt + 满足约束的 response → 正确返回 True |
| ✅ 失败路径 | 有效 prompt + 不满足约束的 response → 正确返回 False |
| ✅ 边界路径 | 空响应、空预测列表、空字符串预测 → 不崩溃，正确返回 |
| ✅ 多约束路径 | 单条样本含2条以上约束 → 逐条分别检测 |
| ✅ Loose 8变体路径 | 星号去除、首尾行裁剪及其组合 → 至少一种变体命中 |
| ✅ 空 kwargs 路径 | None 值 kwargs → 在 build_description 前被正确清理 |

#### 未覆盖项

| 项 | 原因 |
|-----|------|
| 全部 58 个指令检测器 | 仅 mock 了 4 个代表性检测器；完整覆盖需集成测试 |
| `instructions.py` 文件 | 58 个 Checker 类分布在 800+ 行代码中，单元测试通过 mock 指令注册表间接覆盖 |

---

## 附录：关键文件索引

| 文件路径 | 说明 |
|---------|------|
| `benchmark/ais_bench/benchmark/datasets/ifbench/ifbench.py` | IFBench 数据加载 + Evaluator 核心实现（363 行） |
| `benchmark/ais_bench/benchmark/datasets/ifbench/instructions.py` | 58 个指令检测器实现 |
| `benchmark/ais_bench/benchmark/datasets/ifbench/instructions_registry.py` | `INSTRUCTION_DICT` 注册表 |
| `benchmark/ais_bench/datasets/ifbench/data/train-00000-of-00001.parquet` | IFBench 数据集文件（300 条） |
| `benchmark/ais_bench/benchmark/configs/datasets/ifbench/ifbench_0_shot_gen_str.py` | IFBench 评测配置 |
| `benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py` | vLLM 通用 Chat 模型配置 |
| `benchmark/tests/UT/datasets/test_ifbench.py` | 单元测试文件（26 个测试） |
