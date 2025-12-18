# Retriever

在数据集配置文件中，有一个 `retriever` 字段，该字段表示如何从数据集中召回样本作为上下文样例（in-context examples）。其中最常用的是 `FixKRetriever`，表示固定使用某 k 个样本，因此即为 k-shot。另外还有 `ZeroRetriever`，表示不使用任何样本，这在大多数情况下意味着 0-shot。

另一方面，in-context 的样本也可以直接在数据集的模板中指定，在该情况下也会搭配使用 `ZeroRetriever`，但此时的评测并不是 0-shot，而需要根据具体的模板来进行确定。具体请参考 [prompt_template](./prompt_template.md)。

当前 AISBench 支持的 `Retriever` 如下：

- **`ZeroRetriever`**：不使用任何样本作为上下文样例
- **`FixKRetriever`**：固定使用某 k 个样本作为上下文样例
- **`RandomRetriever`**：随机使用某 k 个样本作为上下文样例

## ZeroRetriever

`ZeroRetriever` 是一个零样本检索器，它不会从训练集中召回任何样本作为上下文。对于每个测试样本，它都返回空的索引列表，因此通常用于实现 0-shot 评测。

### 配置方式

```python
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever

infer_cfg = dict(
    retriever=dict(type=ZeroRetriever),
    # ... 其他配置
)
```

### 功能说明

- **返回值**：对于所有测试样本，返回空的索引列表 `[]`
- **使用场景**：
  - 0-shot 评测场景
  - 当上下文样本已经在 prompt 模板中硬编码时（此时虽然使用 `ZeroRetriever`，但实际不是 0-shot）

### 实际示例

假设我们有一个问答数据集，训练集包含以下样本：

**训练集（train）**：

- 样本 0: `{"question": "什么是人工智能？", "answer": "人工智能是计算机科学的一个分支"}`
- 样本 1: `{"question": "Python是什么？", "answer": "Python是一种编程语言"}`

**测试集（test）**：

- 样本 0: `{"question": "什么是机器学习？", "answer": "机器学习是AI的一个子领域"}`

使用 `ZeroRetriever` 时，对于测试样本 0，不会召回任何训练样本，生成的 prompt 中不包含任何上下文示例。

## FixKRetriever

`FixKRetriever` 是一个固定 K 个样本的检索器，它会为所有测试样本固定使用训练集中指定的 K 个样本作为上下文示例。这是实现 k-shot 评测最常用的方式。

### 配置方式

```python
from ais_bench.benchmark.openicl.icl_retriever import FixKRetriever

infer_cfg = dict(
    retriever=dict(
        type=FixKRetriever,
        fix_id_list=[0, 1, 2, 3, 4]  # 指定使用训练集中索引为 0,1,2,3,4 的样本
    ),
    # ... 其他配置
)
```

### 参数说明

- **`fix_id_list`** (`List[int]`): 必需参数，指定要使用的训练样本索引列表。所有测试样本都会使用相同的这些样本作为上下文。

### 功能说明

- **返回值**：对于所有测试样本，返回相同的索引列表（即 `fix_id_list`）
- **使用场景**：
  - k-shot 评测场景（如 5-shot、10-shot 等）
  - 需要确保所有测试样本使用相同的上下文示例，保证评测的一致性

### 实际示例

假设我们有一个阅读理解数据集：

**训练集（train）**：

- 样本 0: `{"article": "文章A...", "question": "问题1", "answer": "A"}`
- 样本 1: `{"article": "文章B...", "question": "问题2", "answer": "B"}`
- 样本 2: `{"article": "文章C...", "question": "问题3", "answer": "C"}`
- 样本 3: `{"article": "文章D...", "question": "问题4", "answer": "D"}`
- 样本 4: `{"article": "文章E...", "question": "问题5", "answer": "A"}`

**测试集（test）**：

- 样本 0: `{"article": "文章X...", "question": "测试问题1", "answer": "B"}`
- 样本 1: `{"article": "文章Y...", "question": "测试问题2", "answer": "C"}`

配置示例（5-shot）：

```python
retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4])
```

**工作流程**：

1. 对于测试样本 0：
   - 召回训练样本 [0, 1, 2, 3, 4]
   - 使用 `ice_template` 将这些样本格式化为上下文示例
   - 将上下文示例插入到测试样本的 prompt 中

2. 对于测试样本 1：
   - 同样召回训练样本 [0, 1, 2, 3, 4]（与测试样本 0 相同）
   - 使用相同的上下文示例

**生成的 prompt 示例**（假设使用简单的模板）：

```text
Read the article, and answer the question by replying A, B, C or D.

Article:
文章A...

Q: 问题1
Answer: A
Read the article, and answer the question by replying A, B, C or D.

Article:
文章B...

Q: 问题2
Answer: B
...（更多示例）
Read the article, and answer the question by replying A, B, C or D.

Article:
文章X...

Q: 测试问题1
Answer:
```

### 配置示例

以下是一些实际配置文件中的使用示例：

#### 示例 1：5-shot 配置

```python
# ais_bench/benchmark/configs/datasets/race/race_middle_gen_5_shot_chat.py
retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4])
```

#### 示例 2：使用 range 生成索引列表

```python
# ais_bench/benchmark/configs/datasets/triviaqa/triviaqa_gen_5_shot_chat_prompt.py
k = 5
retriever=dict(type=FixKRetriever, fix_id_list=list(range(k)))
```

#### 示例 3：10-shot 配置

```python
# ais_bench/benchmark/configs/datasets/hellaswag/hellaswag_gen_10_shot_chat_prompt.py
retriever=dict(type=FixKRetriever, fix_id_list=list(range(10)))
```

#### 示例 4：25-shot 配置

```python
# ais_bench/benchmark/configs/datasets/ARC_c/ARC_c_gen_25_shot_chat_prompt.py
retriever=dict(type=FixKRetriever, fix_id_list=[i for i in range(25)])
```

### 注意事项

1. **索引范围检查**：`fix_id_list` 中的索引必须在训练集的有效范围内（`[0, len(train))`），否则会抛出 `AISBenchValueError` 异常。

2. **索引顺序**：`fix_id_list` 中的顺序决定了上下文示例在 prompt 中的出现顺序。

3. **与 ice_template 配合使用**：使用 `FixKRetriever` 时，通常需要配置 `ice_template` 来格式化召回的样本。

## RandomRetriever

`RandomRetriever` 是一个随机检索器，它会为每个测试样本从训练集中随机选择 K 个样本作为上下文示例。与 `FixKRetriever` 不同，每个测试样本使用的上下文示例是不同的，都是随机选择的。

### 配置方式

```python
from ais_bench.benchmark.openicl.icl_retriever.icl_random_retriever import RandomRetriever

infer_cfg = dict(
    retriever=dict(
        type=RandomRetriever,
        ice_num=5,  # 指定每个测试样本召回的样本数量
        seed=43     # 随机种子，用于保证结果可重复性，默认为 43
    ),
    # ... 其他配置
)
```

### 参数说明

- **`ice_num`** (`int`): 必需参数，指定每个测试样本要召回的样本数量。默认为 1。
- **`seed`** (`Optional[int]`): 可选参数，随机种子，用于保证结果的可重复性。默认为 43。如果设置相同的种子，多次运行会得到相同的结果。

### 功能说明

- **返回值**：对于每个测试样本，返回一个随机选择的索引列表，长度为 `ice_num`
- **随机性**：每个测试样本的上下文示例都是独立随机选择的
- **可重复性**：通过设置 `seed` 参数，可以保证在相同配置下得到可重复的结果
- **使用场景**：
  - 需要为每个测试样本使用不同的上下文示例时
  - 研究不同上下文示例对模型性能的影响
  - 需要随机采样来减少过拟合风险时

### 实际示例

假设我们有一个分类数据集：

**训练集（train）**：

- 样本 0: `{"text": "这是一篇关于科技的文章", "label": "科技"}`
- 样本 1: `{"text": "这是一篇关于体育的文章", "label": "体育"}`
- 样本 2: `{"text": "这是一篇关于娱乐的文章", "label": "娱乐"}`
- 样本 3: `{"text": "这是一篇关于财经的文章", "label": "财经"}`
- 样本 4: `{"text": "这是一篇关于教育的文章", "label": "教育"}`
- 样本 5: `{"text": "这是一篇关于健康的文章", "label": "健康"}`

**测试集（test）**：

- 样本 0: `{"text": "这是一篇关于AI的文章", "label": "科技"}`
- 样本 1: `{"text": "这是一篇关于足球的文章", "label": "体育"}`

配置示例（3-shot，seed=123）：

```python
retriever=dict(type=RandomRetriever, ice_num=3, seed=123)
```

**工作流程**：

1. 对于测试样本 0：
   - 从训练集中随机选择 3 个样本（例如：[1, 3, 5]）
   - 使用 `ice_template` 将这些样本格式化为上下文示例
   - 将上下文示例插入到测试样本的 prompt 中

2. 对于测试样本 1：
   - 从训练集中随机选择 3 个样本（例如：[0, 2, 4]，可能与测试样本 0 不同）
   - 使用 `ice_template` 将这些样本格式化为上下文示例
   - 将上下文示例插入到测试样本的 prompt 中

**生成的 prompt 示例**（假设测试样本 0 随机选择了训练样本 [1, 3, 5]）：

```text
</E>
Text: 这是一篇关于体育的文章
Label: 体育
</E>
Text: 这是一篇关于财经的文章
Label: 财经
</E>
Text: 这是一篇关于健康的文章
Label: 健康
</E>
Text: 这是一篇关于AI的文章
Label:
```

**可重复性说明**：

如果使用相同的 `seed` 值，多次运行会得到相同的随机结果。例如：

```python
# 第一次运行
retriever1 = RandomRetriever(dataset, ice_num=3, seed=123)
result1 = retriever1.retrieve()

# 第二次运行（相同配置）
retriever2 = RandomRetriever(dataset, ice_num=3, seed=123)
result2 = retriever2.retrieve()

# result1 和 result2 完全相同
```

### 注意事项

1. **未充分测试**：`RandomRetriever` 类在代码中标注了警告，表示它还没有被充分测试，使用时需要谨慎。

2. **与 FixKRetriever 的区别**：
   - `FixKRetriever`：所有测试样本使用相同的上下文示例
   - `RandomRetriever`：每个测试样本使用不同的随机上下文示例

3. **随机种子**：如果不指定 `seed` 或每次使用不同的 `seed`，每次运行的结果都会不同，这可能会影响评测结果的可重复性。

4. **与 ice_template 配合使用**：使用 `RandomRetriever` 时，通常需要配置 `ice_template` 来格式化召回的样本。

5. **导入路径**：`RandomRetriever` 没有在 `__init__.py` 中导出，需要直接从模块路径导入：

   ```python
   from ais_bench.benchmark.openicl.icl_retriever.icl_random_retriever import RandomRetriever
   ```

## 完整配置示例

以下是一个完整的数据集配置示例，展示了如何同时使用 `ice_template` 和 `FixKRetriever`：

```python
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import FixKRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer

reader_cfg = dict(
    input_columns=['article', 'question', 'A', 'B', 'C', 'D'],
    output_column='answer',
)

infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='</E>',
            round=[
                dict(role='HUMAN', prompt='Read the article, and answer the question by replying A, B, C or D.\n\nArticle:\n{article}\n\nQ: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:'),
                dict(role='BOT', prompt='{answer}'),
            ]
        ),
        ice_token='</E>',  # 用于标识上下文示例的位置
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),  # 5-shot
    inferencer=dict(type=GenInferencer),
)
```

## 完整配置示例（RandomRetriever）

以下是一个使用 `RandomRetriever` 的完整配置示例：

```python
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever.icl_random_retriever import RandomRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer

reader_cfg = dict(
    input_columns=['text'],
    output_column='label',
)

infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='</E>',
            round=[
                dict(role='HUMAN', prompt='Text: {text}'),
                dict(role='BOT', prompt='Label: {label}'),
            ]
        ),
        ice_token='</E>',
    ),
    retriever=dict(type=RandomRetriever, ice_num=3, seed=123),  # 3-shot，随机选择
    inferencer=dict(type=GenInferencer),
)
```
