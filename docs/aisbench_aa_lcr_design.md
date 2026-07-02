# AISBench 软件设计文档：AA-LCR 数据集

---

## 文档信息

| 项目 | 内容 |
|------|------|
| 文档标题 | AISBench AA-LCR 数据集软件设计文档 |
| 版本 | v1.0 |
| 日期 | 2026-07-02 |
| 涉及模块 | `benchmark/ais_bench/benchmark/datasets/aa_lcr.py` |

---

## 目录

1. [背景描述](#1-背景描述)
2. [设计方案](#2-设计方案)
   - 2.1 [代码架构](#21-代码架构)
   - 2.2 [数据加载](#22-数据加载)
   - 2.3 [Prompt 构造（两阶段）](#23-prompt-构造两阶段)
   - 2.4 [响应解析与 LLM Judge](#24-响应解析与-llm-judge)
   - 2.5 [评测方法](#25-评测方法)
3. [使用说明](#3-使用说明)
   - 3.1 [数据集下载](#31-数据集下载)
   - 3.2 [配置文件修改](#32-配置文件修改)
   - 3.3 [执行命令评测](#33-执行命令评测)
4. [测试用例说明](#4-测试用例说明)
   - 4.1 [测试用例设计](#41-测试用例设计)
   - 4.2 [代码覆盖率分析](#42-代码覆盖率分析)

---

## 1. 背景描述

**AA-LCR**（Artificial Analysis Long Context Retrieval）是由 Artificial Analysis 发布的长文本检索与推理评测基准。该任务要求模型在多个文档中搜索并综合信息来回答问题，考察模型的**长上下文检索**和**跨文档推理**能力。

**核心特点：**

| 属性 | 值 |
|------|-----|
| 基准名称 | AA-LCR / aa_lcr |
| 数据集来源 | evalscope/AA-LCR（ModelScope） |
| 任务类型 | 长上下文问答（Long-Context Question Answering） |
| 样本总数 | 100 |
| 平均 Prompt 长度 | 414,674.06 字符（约 10 万 token） |
| Prompt 长度范围 | 240,709 ~ 548,771 字符 |
| 评测方式 | LLM Judge（大模型评判） |
| 主要指标 | Accuracy（准确率） |

**数据组成：**

- **文档语料库**（ZIP 压缩包 ~4MB）：231 个 `.txt` 文档，分布在三大类下
- **元数据 CSV**：100 条问答记录，每条记录关联若干文档

**文档类别分布：**

| 大类 | 文档集（部分） | 说明 |
|------|-------------|------|
| **Academia** | ac_hack, ac_markets | 学术论文、市场研究报告 |
| **Company_Documents** | co_dc_2Q23 ~ co_pro_transcripts_2024 | 公司财报（Equinix/Digital Realty/NEXTDC）、电话会议记录（Asana/Smartsheet/Atlassian） |
| **Government_Consultations** | gc_bnpl | 政府关于 "先买后付" 的咨询文件 |

**评测流程特殊之处：**

AA-LCR 采用**两阶段评测**：

1. **第一阶段（模型推理）**：将包含多文档内容 + 问题的 Prompt 发送给被测模型，模型生成答案
2. **第二阶段（LLM Judge）**：将被测模型的答案、参考答案、原始问题一起发送给独立的 Judge Model，由 Judge Model 输出 `CORRECT` 或 `INCORRECT`

这种设计的优势是：参考答案可能有多样化的表述方式，简单的字符串匹配无法准确评判；使用 LLM Judge 可以理解语义等价，给出更准确的评判。

---

## 2. 设计方案

### 2.1 代码架构

```
benchmark/ais_bench/benchmark/datasets/
├── aa_lcr.py                 # AA-LCR 全部实现（单文件，474 行）

benchmark/ais_bench/datasets/aa_lcr/
├── AA-LCR_Dataset.csv        # 100 条问答元数据
└── extracted_text/
    └── AA-LCR_extracted-text.zip  # 231 个文档，~4MB

benchmark/ais_bench/benchmark/configs/datasets/aa_lcr/
└── aa_lcr_llmjudge.py        # 评测配置（含 Judge Model 配置）
```

**核心类与函数：**

| 名称 | 类型 | 职责 |
|------|------|------|
| `AALCRDataset` | 类 (继承 BaseDataset) | 加载 CSV 元数据 + 关联文档，构造完整 Prompt |
| `AALCRJGDataset` | 类 (继承 LLMJudgeDataset) | 将模型预测合并到数据集项，构造 Judge Prompt |
| `AALCRJudgeEvaluator` | 类 (继承 BaseEvaluator) | 解析 Judge Model 输出（正则匹配 CORRECT/INCORRECT），计算 Accuracy |
| `_get_context()` | 函数 | 根据 record 元数据读取并格式化关联文档 |
| `_ensure_text_dir_downloaded()` | 函数 | 将文档 ZIP 解压到缓存目录（`~/.cache/ais_bench/aa_lcr/lcr/`） |
| `PROMPT_TEMPLATE` | 常量 | 模型推理 Prompt 模板 |
| `JUDGE_PROMPT` | 常量 | LLM Judge Prompt 模板 |

---

### 2.2 数据加载

#### 加载总体流程

```
CSV 元数据文件 → csv.DictReader 读取 → 遍历每条 record
  └→ _get_context() 读取关联文档 → PROMPT_TEMPLATE 构造 Prompt
  └→ 收集到 raw_data 列表 → Dataset.from_list() 返回
```

#### 详细步骤

**步骤 1：CSV 元数据加载**

使用 Python 标准库 `csv.DictReader` 读取 `AA-LCR_Dataset.csv`，避免 HuggingFace `datasets` 的网络依赖。CSV 格式如下：

| 列名 | 类型 | 示例 |
|------|------|------|
| question_id | str | `q1` |
| question | str | `What is the revenue growth?` |
| answer | str | `10%` |
| document_category | str | `Company_Documents` |
| document_set_id | str | `co_dc_2Q23` |
| data_source_filenames | str | `Copy of Equinix Q2 2023 Press Release and Financials.txt;Copy of Digital-Realty-2Q23-Earnings-Press-Release-FINAL.txt` |
| data_source_urls | str | （可选）原始数据来源 URL |

**步骤 2：文档语料准备**

`_ensure_text_dir_downloaded()` 负责解压文档 ZIP：

- ZIP 路径：`benchmark/ais_bench/datasets/aa_lcr/extracted_text/AA-LCR_extracted-text.zip`
- 缓存目录：`~/.cache/ais_bench/aa_lcr/lcr/`
- 若缓存目录已存在，直接返回（幂等操作）
- 可通过环境变量 `AIS_BENCH_DATASETS_CACHE` 自定义缓存根目录

**步骤 3：文档上下文读取（`_get_context()`）**

```
_get_context(text_dir, record):
  1. 定位文档目录: text_dir / document_category / document_set_id
  2. 解析 data_source_filenames:
     - 若为分号分隔字符串 → 拆分为文件名列表
     - 若已是列表 → 直接使用
     - 若为空 → 回退：遍历目录全部文件（按名称排序，确定性输出）
  3. 读取每个文件内容，包裹在标记中:
     BEGIN DOCUMENT 1:
     <文件内容>
     END DOCUMENT 1

     BEGIN DOCUMENT 2:
     <文件内容>
     END DOCUMENT 2
  4. 容错处理:
     - 文档目录不存在 → 返回空字符串 + 日志警告
     - 单个文件不存在 → 跳过并继续
     - 文件读取错误（编码/IO） → 跳过并日志警告
     - 所有文件都读取失败 → 返回空字符串
```

**步骤 4：构造完整 Prompt**

```python
prompt = PROMPT_TEMPLATE.format(
    documents_text=context,
    question=record['question'],
)
```

#### 核心代码

```python
@staticmethod
def load(path: str, name: str = 'default', **kwargs):
    csv_path = _CSV_PATH
    # 优先使用本地 CSV，避免网络依赖
    if path and os.path.isabs(path):
        candidate = os.path.join(path, 'AA-LCR_Dataset.csv')
        if os.path.exists(candidate):
            csv_path = candidate

    text_dir = _ensure_text_dir_downloaded()

    # 从 CSV 加载记录
    records = []
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    # 逐条构造 Prompt
    raw_data = []
    for record in records:
        context = _get_context(text_dir, record)
        prompt = PROMPT_TEMPLATE.format(
            documents_text=context,
            question=record['question'],
        )
        raw_data.append({
            'input': prompt,
            'answers': record['answer'],
            'question': record['question'],
            'document_category': record.get('document_category', ''),
            'document_set_id': record.get('document_set_id', ''),
            'data_source_urls': record.get('data_source_urls', ''),
        })

    return Dataset.from_list(raw_data)
```

#### 数据输出字段

| 字段 | 类型 | 说明 |
|------|------|------|
| input | str | 完整 Prompt（含全部文档 + 问题），约 24~55 万字符 |
| answers | str | 参考答案 |
| question | str | 问题文本（不含文档） |
| document_category | str | 文档大类 |
| document_set_id | str | 文档集 ID |
| data_source_urls | str | 数据来源 URL |

---

### 2.3 Prompt 构造（两阶段）

AA-LCR 有两类 Prompt，分别用于不同的阶段。

#### 第一阶段：模型推理 Prompt

由 `AALCRDataset.load()` 在数据加载时完整构造，模板为 `PROMPT_TEMPLATE`：

```
BEGIN INPUT DOCUMENTS

BEGIN DOCUMENT 1:
<文档1的完整文本内容>
END DOCUMENT 1

BEGIN DOCUMENT 2:
<文档2的完整文本内容>
END DOCUMENT 2

...

BEGIN DOCUMENT N:
<文档N的完整文本内容>
END DOCUMENT N

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

<问题文本>

END QUESTION
```

配置文件中直接透传：

```python
aa_lcr_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{input}',  # 直通，input 字段已包含完整 prompt
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
```

#### 第二阶段：Judge Model Prompt

模型推理完成后，`AALCRJGDataset` 为每条预测构造 Judge Prompt，模板为 `JUDGE_PROMPT`：

```
Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the
OFFICIAL ANSWER.

The question, for reference only: {question}
The OFFICIAL ANSWER: {answers}
CANDIDATE ANSWER TO ASSESS: {model_answer}

Reply only with CORRECT or INCORRECT.
```

**关键设计：**

- Judge Prompt 只给出 question（参考）、answers（正确答案）、model_answer（待评判答案），**不包含文档全文**。Judge Model 基于语义一致性判断，而非检索文档
- **temperature=0.0**：确保 Judge 输出确定性的 CORRECT/INCORRECT
- **enable_thinking=False**：关闭推理链，只需简单判断，节省 token

#### Judge 配置详解

```python
aa_lcr_judge_infer_cfg = dict(
    judge_reader_cfg=dict(
        input_columns=['question', 'answers', 'model_answer'],
        output_column='model_pred_uuid',
    ),
    judge_model=dict(
        type=VLLMCustomAPIChat,
        host_ip='localhost',
        host_port=8005,           # Judge Model 独立端口
        max_out_len=512,
        generation_kwargs=dict(
            chat_template_kwargs=dict(
                max_tokens=1024,
                temperature=0.0,       # 确定性输出
                enable_thinking=False,  # 关闭推理链
            ),
        ),
    ),
    judge_dataset_type=AALCRJGDataset,
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[dict(role='HUMAN', prompt=JUDGE_PROMPT)],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
```

---

### 2.4 响应解析与 LLM Judge

#### 完整评测流程

```
                                 ┌─────────────┐
第1阶段（模型推理）               │ 被测 Model   │
  Prompt ──────────────────────→ │ (port 8080)  │
                                 └──────┬──────┘
                                        │ prediction
                                        ▼
                                 ┌─────────────┐
第2阶段（LLM Judge）              │ Judge Model  │
  JUDGE_PROMPT ────────────────→ │ (port 8005)  │
                                 └──────┬──────┘
                                        │ "CORRECT" / "INCORRECT"
                                        ▼
                                 ┌─────────────┐
第3阶段（结果判定）               │ Evaluator    │
  AALCRJudgeEvaluator.score()    │              │
  └─ re.search(r'\bCORRECT\b')  │              │
  └─ 计算 Accuracy               └─────────────┘
```

#### Judge 结果判定

`AALCRJudgeEvaluator.score()` 使用**单词边界正则表达式**解析 Judge Model 输出：

```python
is_correct = bool(re.search(r'\bCORRECT\b', judge_output, re.IGNORECASE))
```

**为什么使用 `\bCORRECT\b` 而非简单子串搜索：**

| 方法 | 风险 |
|------|------|
| `"CORRECT" in output` | `INCORRECT` 会被误判为正确（包含 CORRECT 子串） |
| `output == "CORRECT"` | 过于严格，Judge 可能输出 "CORRECT." 或多行文本 |
| `re.search(r'\bCORRECT\b', output)` | ✅ 正确：单词边界确保只匹配完整单词 CORRECT |

**边界情况处理：**

| 场景 | 行为 |
|------|------|
| `len(predictions) != len(references)` | 返回 `{"error": "predictions and references have different length..."}` |
| 空预测列表 | Accuracy = 0% |
| Judge 输出包含换行 | 正则跨行匹配，如 `"Some text.\nCORRECT\nMore text."` → 正确识别 |
| 大小写混合 | `re.IGNORECASE` 同时匹配 `correct`, `Correct`, `CORRECT` |

---

### 2.5 评测方法

#### 评测指标

**Accuracy（准确率）**

```
Accuracy = (CORRECT 判定数 / 总样本数) × 100%
```

这是 AA-LCR 的唯一核心指标。不同于 IFBench 的多指标设计，AA-LCR 依赖 LLM Judge 的二元判定，最终只输出正确率。

#### 每样本 detail 结构

```python
{
    "judge_output": "CORRECT",      # Judge Model 原始输出文本
    "answer": "10%",                # CSV 中的参考答案
    "correct": True                 # 正则判定结果
}
```

#### 核心代码

```python
def score(self, predictions: List, references: List) -> Dict[str, Any]:
    if len(predictions) != len(references):
        return {
            'error': (
                'predictions and references have different length. '
                f'len(predictions): {len(predictions)}, '
                f'len(references): {len(references)}'
            )
        }

    details = {}
    correct = 0
    total = 0

    for index, (judge_output, ref) in enumerate(zip(predictions, references)):
        total += 1
        is_correct = bool(
            re.search(r'\bCORRECT\b', judge_output, re.IGNORECASE)
        )
        if is_correct:
            correct += 1
        details[str(index)] = {
            'judge_output': judge_output,
            'answer': ref,
            'correct': is_correct,
        }

    accuracy = correct / total * 100 if total > 0 else 0
    return {'accuracy': accuracy, 'details': details}
```

---

## 3. 使用说明

### 3.1 数据集下载

AA-LCR 数据集已集成在 AISBench 仓库中，**无需额外下载**：

| 文件 | 路径 | 大小 | 说明 |
|------|------|------|------|
| 元数据 CSV | `benchmark/ais_bench/datasets/aa_lcr/AA-LCR_Dataset.csv` | ~15KB | 100 条问答元数据 |
| 文档语料库 ZIP | `benchmark/ais_bench/datasets/aa_lcr/extracted_text/AA-LCR_extracted-text.zip` | ~4MB | 231 个文档 |

**首次运行时的自动操作：**

`_ensure_text_dir_downloaded()` 会将 ZIP 自动解压到：

```
~/.cache/ais_bench/aa_lcr/lcr/
├── Academia/
│   ├── ac_hack/        # 8 个学术论文
│   └── ac_markets/     # 4 个市场报告
├── Company_Documents/  # 200+ 个公司文档
│   ├── co_dc_2Q23/
│   ├── co_dc_3Q23/
│   ├── co_dc_4Q23/
│   ├── co_dc_ann_sup_a/
│   ├── co_dc_dr_press/
│   ├── co_dc_eq_history/
│   ├── co_dc_nxt_history/
│   ├── co_dc_press_a/
│   ├── co_dc_press_b/
│   ├── co_pro_asana_rel_tran/
│   ├── co_pro_rel_ann/
│   ├── co_pro_ssheet_rel_tran/
│   ├── co_pro_transcripts_2023/
│   └── co_pro_transcripts_2024/
└── Government_Consultations/
    └── gc_bnpl/         # 4 个政府文件
```

**自定义缓存目录：**

```bash
export AIS_BENCH_DATASETS_CACHE=/your/custom/cache/path
```

---

### 3.2 配置文件修改

**配置文件：** `benchmark/ais_bench/benchmark/configs/datasets/aa_lcr/aa_lcr_llmjudge.py`

#### 数据集配置

```python
aa_lcr_datasets = [
    dict(
        abbr='aa_lcr',
        type=AALCRDataset,
        path='benchmark/ais_bench/datasets/aa_lcr/',
        reader_cfg=aa_lcr_reader_cfg,
        infer_cfg=aa_lcr_infer_cfg,
        judge_infer_cfg=aa_lcr_judge_infer_cfg,  # 关键：LLM Judge 配置
        eval_cfg=aa_lcr_eval_cfg,
    )
]
```

#### 模型推理配置

```python
aa_lcr_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{input}',          # 直通 AALCRDataset.load() 构造的完整 prompt
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
```

**注意：** `max_out_len` 需在模型配置中设置得足够大（建议 ≥ 512），因为 AA-LCR 的答案可能需要较长的推理过程。

#### Judge Model 配置

```python
aa_lcr_judge_infer_cfg = dict(
    judge_model=dict(
        type=VLLMCustomAPIChat,
        host_ip='localhost',
        host_port=8005,              # Judge 使用独立端口
        max_out_len=512,
        generation_kwargs=dict(
            chat_template_kwargs=dict(
                max_tokens=1024,
                temperature=0.0,      # 确定性输出，Judge 不可随机
                enable_thinking=False, # 关闭推理链
            ),
        ),
    ),
    # ...
)
```

**Judge Model 关键参数：**

| 参数 | 推荐值 | 原因 |
|------|--------|------|
| `host_port` | 8005（与被测模型 8080 分离） | 避免端口冲突 |
| `temperature` | 0.0 | Judge 评判需要确定性，不能有随机性 |
| `enable_thinking` | False | 只需输出 CORRECT/INCORRECT，思考链浪费资源 |
| `max_tokens` | 1024 | 足够输出 CORRECT/INCORRECT + 简短理由 |

#### 模型配置（通用）

`benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py`：

```python
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-general-chat",
        host_ip="localhost",
        host_port=8080,
        max_out_len=512,
        batch_size=1,
        retry=2,
        generation_kwargs=dict(
            temperature=0.01,
            ignore_eos=False,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
```

---

### 3.3 执行命令评测

#### 前置条件

**AA-LCR 需要启动两个模型服务：**

```bash
# 1. 被测模型（端口 8080）
vllm serve <model_name> --port 8080

# 2. Judge Model（端口 8005）
vllm serve <judge_model_name> --port 8005
```

> **Judge Model 选择建议：** 选用能力较强的模型（如 GPT-4 级别）以获得更可靠的评判。Judge Model 只需要理解 question + answer + model_answer 的语义一致性，不涉及长文本处理。

#### 评测命令

**标准评测（全量 100 条）：**

```bash
ais_bench --models vllm_api_general_chat --datasets aa_lcr_llmjudge
```

**限制样本数（快速验证流程）：**

```bash
ais_bench --models vllm_api_general_chat --datasets aa_lcr_llmjudge --limit 5
```

**自定义 Judge Model 端口：**

在 `aa_lcr_llmjudge.py` 中修改 `host_port` 后运行即可。如临时覆盖：
```bash
# 可通过修改配置文件中的 host_port 实现
# aa_lcr_llmjudge.py → judge_model.host_port
```

#### 预期输出示例

```
[INFO] AA-LCR documents found in cache: ~/.cache/ais_bench/aa_lcr/lcr/
[INFO] Loading AA-LCR dataset metadata from: .../AA-LCR_Dataset.csv
[INFO] Loaded 100 records from AA-LCR CSV
[INFO] ========== CONSTRUCTED PROMPT (question_id=q1) ==========
[INFO] QUESTION: What is the revenue growth?
[INFO] ANSWER:   10%
[INFO] PROMPT (251234 chars):
BEGIN INPUT DOCUMENTS
BEGIN DOCUMENT 1:
...
END DOCUMENT 1
END INPUT DOCUMENTS
...
[INFO] ========== CONSTRUCTED PROMPT END ==========

... (模型推理阶段) ...

[INFO] ========== JUDGE ITEM (pred_uuid=xxx) ==========
[INFO] QUESTION:      What is the revenue growth?
[INFO] ANSWERS:       10%
[INFO] MODEL_ANSWER:  The revenue growth is 10%.
[INFO] JUDGE_PROMPT (312 chars):
Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT...
[INFO] ========== JUDGE ITEM END ==========

... (Judge 评判阶段) ...

[INFO] ========== EVAL RESULT (index=0) ==========
[INFO] JUDGE_OUTPUT: CORRECT
[INFO] REFERENCE:    10%
[INFO] IS_CORRECT:   True
[INFO] ========== EVAL RESULT END ==========

[INFO] ========== EVAL SUMMARY (correct=75/100, accuracy=75.00%) ==========
```

---

## 4. 测试用例说明

**测试文件：** `benchmark/tests/UT/datasets/test_aa_lcr.py`

**测试总数：** 19 个

### 4.1 测试用例设计

#### Prompt 模板测试（2 个）

| 测试 | 覆盖目标 | 验证内容 |
|------|---------|---------|
| `test_judge_prompt_format` | `JUDGE_PROMPT` 常量 | 包含 question、OFFICIAL ANSWER、CANDIDATE ANSWER、以 "CORRECT or INCORRECT." 结尾 |
| `test_prompt_template_format` | `PROMPT_TEMPLATE` 常量 | 包含 BEGIN/END INPUT DOCUMENTS、START/END QUESTION，文档内容和问题正确嵌入 |

#### 文档上下文读取测试（`_get_context()`—5 个）

| 测试 | 覆盖目标 | 验证内容 |
|------|---------|---------|
| `test_get_context_with_filenames_list` | 文件名列表模式 | 按指定文件名顺序读取，`BEGIN DOCUMENT 1` 包裹 |
| `test_get_context_with_semicolon_separated_string` | CSV 分号分隔模式 | `"a.txt;b.txt"` → 正确拆分为两个文件 |
| `test_get_context_missing_folder_returns_empty` | 缺失目录容错 | 不存在的 `document_category` → 返回空字符串，不崩溃 |
| `test_get_context_missing_file_skipped` | 缺失文件容错 | 文件名存在但文件缺失 → 跳过，继续处理 |
| `test_get_context_fallback_directory_iteration` | 无文件名回退模式 | `data_source_filenames=[]` → 遍历目录全部文件（排序） |

#### Judge Evaluator 测试（8 个）

| 测试 | 覆盖目标 | 验证内容 |
|------|---------|---------|
| `test_all_correct` | 正确率计算 | 全部 CORRECT → 100% |
| `test_all_incorrect` | 正确率计算 | 全部 INCORRECT → 0% |
| `test_mixed_results` | 准确率计算 | 2/3 CORRECT → 66.67% |
| `test_word_boundary_matching` | 正则单词边界 | `INCORRECT` / `INCORRECTLY WORDED` → 不误判为 CORRECT |
| `test_case_insensitive` | 大小写不敏感 | `correct` / `Correct` / `CORRECT` → 均识别 |
| `test_newline_embedded_correct` | 换行嵌入 | `"Some text.\nCORRECT\nMore text."` → 正确识别 |
| `test_length_mismatch_returns_error` | 错误处理 | `len(pred) != len(ref)` → 返回 error 字典 |
| `test_empty_predictions` | 边界健壮性 | 空列表 → accuracy=0%，不崩溃 |

#### 数据集加载测试（3 个）

| 测试 | 覆盖目标 | 验证内容 |
|------|---------|---------|
| `test_load_returns_dataset` | `AALCRDataset.load()` | 返回 Dataset 含 6 个列名、2 条样本 |
| `test_load_prompt_contains_documents` | Prompt 构造 | 第一条 Prompt 包含 `BEGIN INPUT DOCUMENTS` + 文档内容 + `START QUESTION` |
| `test_load_answers_match_csv` | 数据完整性 | 第一条 answers=`10%`、第二条=`A summary here.` |

#### Judge Dataset 测试（1 个）

| 测试 | 覆盖目标 | 验证内容 |
|------|---------|---------|
| `test_aalcr_jg_dataset_get_class` | `AALCRJGDataset._get_dataset_class()` | 返回 `AALCRDataset` 类引用 |

---

### 4.2 代码覆盖率分析

#### 函数/类覆盖情况

| 模块/函数 | 是否覆盖 | 测试数量 | 覆盖路径 |
|-----------|---------|---------|---------|
| `PROMPT_TEMPLATE` | ✅ 完全 | 1 | 格式验证 |
| `JUDGE_PROMPT` | ✅ 完全 | 1 | 格式验证 |
| `_get_context()` | ✅ 完全 | 5 | 列表模式、CSV 字符串模式、缺失目录、缺失文件、回退目录遍历 |
| `_ensure_text_dir_downloaded()` | ⚠️ 间接 | 3 | 通过 Dataset load 测试间接覆盖（mock 返回值） |
| `AALCRJudgeEvaluator.score()` | ✅ 完全 | 8 | 全部正确、全部错误、混合、单词边界、大小写、换行、长度不匹配、空列表 |
| `AALCRDataset.load()` | ✅ 完全 | 3 | 列名结构、Prompt 内容、answers 完整性 |
| `AALCRJGDataset._get_dataset_class()` | ✅ 完全 | 1 | 返回正确类引用 |
| `AALCRJGDataset._modify_dataset_item()` | ❌ 未直接覆盖 | — | 需端到端集成测试 |

#### 关键路径覆盖

| 路径类型 | 覆盖情况 |
|---------|---------|
| ✅ 正常路径 | CSV 加载 → 文档读取 → Prompt 构造 → 正确输出 |
| ✅ 文件名解析路径 | 列表格式 / 分号分隔字符串 / 无文件名回退 |
| ✅ 容错路径 | 缺失目录 → 空字符串；缺失文件 → 跳过；读取错误 → 跳过 |
| ✅ Judge 判定路径 | CORRECT/INCORRECT 识别、单词边界保护、大小写不敏感、多行匹配 |
| ✅ 边界路径 | 长度不匹配 error、空预测列表 0%、空语料 |

#### 未覆盖项及原因

| 项 | 原因 | 建议 |
|-----|------|------|
| `_ensure_text_dir_downloaded()` 真实解压 | 单元测试使用 mock 返回值 | 需集成测试覆盖真实 ZIP 解压流程 |
| `AALCRJGDataset._modify_dataset_item()` | 涉及 Judge 全流程 | 需端到端集成测试 |
| 完整 100 条样本加载 | 单元测试使用 2 条 mock CSV | 集成测试验证全量数据 |

---

## 附录：关键文件索引

| 文件路径 | 说明 |
|---------|------|
| `benchmark/ais_bench/benchmark/datasets/aa_lcr.py` | AA-LCR 完整实现（单文件，474 行） |
| `benchmark/ais_bench/datasets/aa_lcr/AA-LCR_Dataset.csv` | 100 条问答元数据 |
| `benchmark/ais_bench/datasets/aa_lcr/extracted_text/AA-LCR_extracted-text.zip` | 231 个文档语料库 |
| `benchmark/ais_bench/benchmark/configs/datasets/aa_lcr/aa_lcr_llmjudge.py` | AA-LCR 评测配置（含 LLM Judge） |
| `benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py` | vLLM 通用 Chat 模型配置 |
| `benchmark/tests/UT/datasets/test_aa_lcr.py` | 单元测试文件（19 个测试） |
