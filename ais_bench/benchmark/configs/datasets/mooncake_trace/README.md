# Mooncake Trace

中文 | [English](README_en.md)

## 数据集简介

Mooncake Trace 数据集是一个用于性能评测的 trace 数据集，支持从包含 hash_id 和 trace 数据的 JSONL 文件生成 prompt。该数据集实现了与 AIPerf 兼容的 prompt 生成机制，包括可复现的随机数生成、语料库采样、hash_id 缓存等核心特性，同时支持通过 timestamp 字段精确控制请求的发送时间。

**参考与实现**：语料库来源自 [AIPerf](https://github.com/ai-dynamo/aiperf)，prompt 生成机制（可复现随机数、语料库采样、hash_id 缓存等）参考其实现原理。

主要特性：

- **Hash ID 缓存机制**：相同 hash_id 生成相同的 prompt 内容，确保可复现性
- **时间戳控制**：支持通过 timestamp 字段精确控制请求的发送时间，模拟真实请求的时间分布
- **input_text 支持**：支持直接通过 input_text 字段指定 prompt，支持混合模式（部分使用 input_text，部分使用生成的 prompt）
- **固定调度控制**：支持时间戳自动偏移、开始偏移和结束偏移，灵活控制请求时间窗口
- **缓存机制**：支持 prompt 生成结果缓存，避免重复生成，提高加载效率

## 数据集格式

数据集文件为 JSONL 格式，每行一个 JSON 对象。支持的字段如下：

### 必需字段（二选一）

- `input_length` (int): 输入 token 数量，用于生成 prompt
- `input_text` (str, 可选): 直接指定的 prompt 文本。如果存在此字段，`input_length` 和 `hash_ids` 将被忽略

### 可选字段

- `output_length` (int): 最大输出 token 数量，默认为 1
- `hash_ids` (list[int]): Hash ID 列表，用于缓存复用。每个 hash_id 对应一个 token 块（默认 512 tokens）
- `timestamp` (int 或 float，可选): 请求时间戳（毫秒），用于控制请求的发送时间；必须 >= 0。不提供则该条无 timestamp，不参与按时间戳的调度与过滤

### Hash IDs 与 Input Length 的关系规则

当使用 `hash_ids` 时，`input_length` 必须满足以下规则：

- **每个 hash_id 对应一个 token 块**：默认每个 hash_id 对应 512 个 tokens（block_size = 512）
- **计算规则**：
  - 如果有 n 个 hash_ids，前 n-1 个 hash_ids 每个对应 512 个 tokens
  - 最后一个 hash_id 对应 `final_block_size` 个 tokens
  - `final_block_size = input_length - (n-1) * 512`
  - 必须满足：`0 < final_block_size <= 512`

- **有效范围**：
  - 对于 n 个 hash_ids，`input_length` 的有效范围是：`[(n-1) * 512 + 1, n * 512]`
  - 最小值：`(n-1) * 512 + 1`
  - 最大值：`n * 512`

- **示例**：
  - 1 个 hash_id：`input_length` 范围是 `[1, 512]`
  - 2 个 hash_ids：`input_length` 范围是 `[513, 1024]`
  - 3 个 hash_ids：`input_length` 范围是 `[1025, 1536]`
  - 4 个 hash_ids：`input_length` 范围是 `[1537, 2048]`

- **验证公式**：

  ```
  final_block_size = input_length - (len(hash_ids) - 1) * 512
  必须满足：0 < final_block_size <= 512
  ```

### 数据格式示例

```jsonl
# 使用 hash_ids 生成 prompt（1 个 hash_id，input_length 范围：1-512）
{"timestamp": 0, "input_length": 256, "output_length": 10, "hash_ids": [1]}

# 使用 hash_ids 生成 prompt（2 个 hash_ids，input_length 范围：513-1024）
{"timestamp": 200, "input_length": 1024, "output_length": 100, "hash_ids": [2, 3]}

# 使用 hash_ids 生成 prompt（3 个 hash_ids，input_length 范围：1025-1536）
{"timestamp": 500, "input_length": 1536, "output_length": 200, "hash_ids": [4, 5, 6]}

# 直接使用 input_text（不需要 hash_ids）
{"timestamp": 1000, "input_text": "What is the capital of France?", "output_length": 10}
```

**注意**：示例中的 `input_length` 值都在对应 hash_ids 数量的有效范围内，确保数据格式正确。

## 数据集部署

1. 准备 JSONL 格式的 trace 数据文件
2. 将文件放置在合适的位置（支持相对路径和绝对路径）
3. 在配置文件中指定 `path` 参数指向数据文件

## 配置参数说明

### 基础参数

| 参数 | 类型 | 说明 | 默认值 |
| ---- | ---- | ---- | ---- |
| `path` | str | 原始包含 hash_id 和 trace 数据的 JSONL 文件路径。使用相对路径时相对于源码根路径，支持绝对路径 | 必需 |
| `model_path` | str | 模型路径，用于加载 tokenizer | 必需 |
| `random_seed` | int | 随机数种子，用于可复现性 | None |
| `generated_prompts_path` | str | 生成的 prompt 缓存路径。如果为空，将自动生成缓存文件名。使用相对路径时相对于源码根路径，支持绝对路径 | "" |

### 固定调度参数

| 参数 | 类型 | 说明 | 默认值 |
| ---- | ---- | ---- | ---- |
| `fixed_schedule_auto_offset` | bool | 是否自动偏移时间戳（使第一个时间戳为 0） | False |
| `fixed_schedule_start_offset` | int | 开始偏移量（毫秒），过滤掉小于该偏移的时间戳 | 0 |
| `fixed_schedule_end_offset` | int | 结束偏移量（毫秒），-1 表示不限制，>=0 时过滤掉大于该偏移的时间戳 | -1 |

### 参数使用说明

1. **fixed_schedule_auto_offset**:
   - 当设置为 `True` 时，会自动计算最小时间戳，并将所有时间戳减去最小值，使第一个时间戳为 0
   - 适用于时间戳不是从 0 开始的场景

2. **fixed_schedule_start_offset** 和 **fixed_schedule_end_offset**:
   - 用于截取 trace 数据的特定时间窗口
   - `start_offset`: 只保留时间戳 >= 该值的请求
   - `end_offset`: 只保留时间戳 <= 该值的请求（-1 表示不限制）
   - 例如：`start_offset=1000, end_offset=5000` 只处理时间戳在 [1000, 5000] 范围内的请求

3. **缓存机制**:
   - 当使用 `fixed_schedule` 参数时，缓存文件名会自动包含这些参数，确保参数变化时使用不同的缓存文件
   - 如果缓存文件已存在，会直接加载缓存，跳过 prompt 生成过程
   - 如需重新生成，请删除缓存文件后重新运行

## 可用数据集任务

| 任务名称 | 简介 | 评估指标 | few-shot | prompt格式 | 对应源码配置文件路径 |
| --- | --- | --- | --- | --- | --- |
| mooncake-trace | Mooncake trace 数据集生成式任务 | 性能测评 | 0-shot | 字符串格式 | [mooncake_trace_gen.py](mooncake_trace_gen.py) |

## 使用示例

### 基础配置

```python
mooncake_trace_datasets = [
    dict(
        abbr='mooncake-trace',
        type=MooncakeTraceDataset,
        path='path/to/trace.jsonl',
        model_path='path/to/model',
        random_seed=1234,
        generated_prompts_path='',  # 自动生成缓存路径
        fixed_schedule_auto_offset=False,
        fixed_schedule_start_offset=0,
        fixed_schedule_end_offset=-1,
        reader_cfg=mooncake_trace_reader_cfg,
        infer_cfg=mooncake_trace_infer_cfg,
        eval_cfg=mooncake_trace_eval_cfg
    )
]
```

### 使用时间戳自动偏移

```python
mooncake_trace_datasets = [
    dict(
        abbr='mooncake-trace',
        type=MooncakeTraceDataset,
        path='path/to/trace.jsonl',
        model_path='path/to/model',
        random_seed=1234,
        fixed_schedule_auto_offset=True,  # 启用自动偏移
        fixed_schedule_start_offset=0,
        fixed_schedule_end_offset=-1,
        # ... 其他配置
    )
]
```

### 截取特定时间窗口

```python
mooncake_trace_datasets = [
    dict(
        abbr='mooncake-trace',
        type=MooncakeTraceDataset,
        path='path/to/trace.jsonl',
        model_path='path/to/model',
        random_seed=1234,
        fixed_schedule_auto_offset=False,
        fixed_schedule_start_offset=1000,  # 从 1 秒开始
        fixed_schedule_end_offset=5000,    # 到 5 秒结束
        # ... 其他配置
    )
]
```

### 使用 input_text 字段

在 JSONL 文件中直接指定 prompt：

```jsonl
{"timestamp": 0, "input_text": "What is the capital of France?", "output_length": 10}
{"timestamp": 200, "input_length": 655, "output_length": 2, "hash_ids": [46, 47]}
```

支持混合模式：同一数据集中部分数据使用 `input_text`，部分使用 `input_length` 和 `hash_ids` 生成 prompt。

## 特性说明

### Hash ID 缓存机制

- 每个 `hash_id` 对应一个 token 块（默认 512 tokens）
- 相同 `hash_id` 会生成相同的 token 序列，确保可复现性
- 缓存机制避免重复生成，提高效率

### 时间戳控制

- `timestamp` 字段用于精确控制请求的发送时间
- 在性能测评场景中，系统会根据 `timestamp` 计算延迟时间并控制请求发送
- 是否按 timestamp 调度由模型配置中的 **use_timestamp** 决定：当 **use_timestamp 为 True** 且数据集中包含 timestamp 时，按 timestamp 发送请求，**request_rate 被忽略**；当 **use_timestamp 为 False** 时，按 **request_rate** 调度，数据集中 timestamp 不参与调度

### input_text 支持

- 当 trace 数据中存在 `input_text` 字段时，直接使用它作为 prompt
- 此时 `input_length` 和 `hash_ids` 将被忽略
- 支持混合模式：同一数据集中部分数据使用 `input_text`，部分使用生成的 prompt

## 注意事项

1. **语料库文件**：语料库来源自 [AIPerf](https://github.com/ai-dynamo/aiperf)。系统需要 Shakespeare 文本作为语料库（`assets/shakespeare.txt`），用于生成 prompt。请确保该文件位于 **`ais_bench/third_party/aiperf/assets/shakespeare.txt`**（以 ais_bench 包根为基准）。若报错，错误信息中会列出实际查找的路径列表。

2. **缓存文件**：生成的 prompt 会缓存到文件中，如果数据文件或配置参数发生变化，建议删除缓存文件以重新生成。

3. **时间戳单位**：所有时间戳相关参数的单位都是**毫秒**。

4. **参数验证**：`fixed_schedule_start_offset` 必须 <= `fixed_schedule_end_offset`（当 `end_offset >= 0` 时）。

5. **数据格式验证**：当使用 `hash_ids` 时，必须确保 `input_length` 与 `hash_ids` 数量匹配：
   - **常见错误**：`input_length` 值不在有效范围内
     - 错误示例：`{"input_length": 512, "hash_ids": [1, 2]}`
     - 原因：2 个 hash_ids 时，`input_length` 应该是 513-1024，但实际是 512
     - 正确示例：`{"input_length": 1024, "hash_ids": [1, 2]}`

   - **验证方法**：

     ```
     final_block_size = input_length - (len(hash_ids) - 1) * 512
     如果 final_block_size <= 0 或 final_block_size > 512，则数据格式错误
     ```

   - **快速检查公式**：
     - 对于 n 个 hash_ids，确保：`(n-1) * 512 + 1 <= input_length <= n * 512`

   - **错误提示**：如果数据格式不正确，会触发 **ParameterValueError**（错误码 **DSET-PARAM-004**），错误信息形如：

     ```
     Input length: XXX, Hash IDs: [...], Block size: 512 are not compatible.
     Final block size: XXX must be > 0 and <= 512.
     ```

     详见错误码 **DSET-PARAM-004** 的说明（文档《错误码说明》）。
