# 用户自定义数据评测任务

本项目基于 `ais\_bench` 构建用户自定义数据评测任务，适用于对语言大模型在自定义选择题（MCQ）数据上的回答能力进行评估。

当前示例配置支持两种评测方式：

* `SelfData_gen_0_shot.py`：0-shot 评测，不提供示例，模型直接根据题目和选项作答。\n* `SelfData_gen_5_shot.py`：5-shot 评测，固定选取数据集前 5 条样本作为上下文示例，再让模型回答当前样本。

\---

## 1\. 项目文件说明

```text
SelfData\_gen\_0\_shot.py   # 0-shot 自定义数据集评测配置
SelfData\_gen\_5\_shot.py   # 5-shot 自定义数据集评测配置
YourData.csv         # 用户自定义评测数据，需要自行准备
```

\---

## 2\. 适用场景

本项目适用于以下场景：

* 用户自定义分类任务评测
* 用户自定义选择题任务评测
* 大模型 0-shot 能力评估
* 大模型 few-shot 能力评估
* 不同 Prompt 效果对比
* 不同模型在同一数据集上的准确率对比
* 自定义业务场景下的大模型判断能力评估

例如：

* 文本分类
* 知识问答选择题
* 多选项决策判断

\---

## 3\. 数据格式

数据文件建议放在：

```text
ais\_bench/datasets/SelfData/YourData.csv
```

默认代码使用 CSV 格式，并要求包含以下字段：

```csv
question,A,B,answer
请判断这句话的情感倾向：这个产品很好用,正向,负向,A
请判断这句话的情感倾向：这个服务体验很差,正向,负向,B
```

字段含义如下：

|字段名|含义|
|-|-|
|`question`|待评测的问题、文本、样本或任务输入|
|`A`|选项 A 的内容|
|`B`|选项 B 的内容|
|`answer`|正确答案，默认填写 `A` 或 `B`|

注意：当前配置默认是 A/B 二分类选择题。如果需要支持 A/B/C/D 等多选项任务，需要同步修改 Prompt 模板、输入字段和答案匹配规则。

\---

## 4\. Prompt 模板说明

配置文件中的 `QUERY\_TEMPLATE` 用于定义最终输入给模型的提示词。

默认模板结构如下：

```python
QUERY\_TEMPLATE = """
请根据题目内容，从给定选项中选择一个最合适的答案。

请只输出最终答案，格式为：
答案: A
或
答案: B

题目：
{question}

A) {A}
B) {B}
""".strip()
```

其中：

* `{question}` 会被替换为数据中的 `question` 字段。
* `{A}` 会被替换为数据中的 `A` 字段。
* `{B}` 会被替换为数据中的 `B` 字段。

如果你的任务有更明确的业务定义，可以在模板中补充任务说明。例如：

```python
QUERY\_TEMPLATE = """
请完成以下文本分类任务。

任务说明：
根据输入文本判断其所属类别，并从选项中选择一个最合适的答案。

输出要求：
你回答的最后一行必须是：答案: A 或 答案: B

输入：
{question}

A) {A}
B) {B}
""".strip()
```

\---

## 5\. 0-shot 评测说明

0-shot 表示不向模型提供示例，模型仅根据当前题目、选项和 Prompt 完成判断。

对应配置文件：

```text
SelfData\_0\_shot.py
```

运行命令：

```bash
ais\_bench --models vllm\_api\_general\_chat --datasets SelfData\_gen\_0\_shot --work-dir path/to/result/dir
```

参数说明：

|参数|含义|
|-|-|
|`--models vllm\_api\_general\_chat`|指定使用已配置的模型服务|
|`--datasets SelfData\_gen\_0\_shot`|指定使用 0-shot 自定义数据集配置|
|`--work-dir`|指定评测结果保存目录|

\---

## 6\. 5-shot 评测说明

5-shot 表示先向模型提供 5 条示例样本，再让模型回答当前题目。

对应配置文件：

```text
SelfData\_5\_shot.py
```

运行命令：

```bash
ais\_bench --models vllm\_api\_general\_chat --datasets SelfData\_gen\_5\_shot --work-dir path/to/result/dir
```

当前配置默认固定选取数据集前 5 条样本作为上下文示例：

```python
fix\_id\_list=\[0, 1, 2, 3, 4]
```

如果需要更换示例样本，可以修改 `fix\_id\_list` 中的样本编号。例如：

```python
fix\_id\_list=\[5, 8, 12, 20, 31]
```

建议选择质量较高、格式规范、类别覆盖均衡的样本作为 few-shot 示例。

\---

## 7\. 评测配置说明

当前配置使用准确率作为评价指标：

```python
evaluator=dict(type=AccEvaluator)
```

模型输出会通过正则表达式抽取最终答案：

```python
answer\_pattern=r'(?i)答案\\s\*\[:：]\\s\*\[\\W]\*(\[AB])\[\\W]\*'
```

因此，Prompt 中应明确要求模型按照以下格式输出：

```text
答案: A
```

或：

```text
答案: B
```

如果模型没有按照指定格式输出，可能会导致答案解析失败，从而影响评测结果。

\---

## 8\. 如何替换为自己的数据集

如果要使用自己的数据，只需要修改配置文件中的数据路径和数据集简称。

默认配置：

```python
path='ais\_bench/datasets/SelfData/YourData.csv'
abbr='YourData\_0\_shot'
```

可以修改为：

```python
path='ais\_bench/datasets/SelfData/my\_task.csv'
abbr='my\_task\_0\_shot'
```

如果使用 5-shot 配置，也需要同步修改：

```python
path='ais\_bench/datasets/SelfData/my\_task.csv'
abbr='my\_task\_5\_shot'
```

\---

## 9\. 常见修改项

### 9.1 修改任务提示词

修改：

```python
QUERY\_TEMPLATE
```

如果使用 5-shot 配置，还可以同步修改：

```python
ICE\_TEMPLATE
```

`QUERY\_TEMPLATE` 用于当前待评测样本，`ICE\_TEMPLATE` 用于 few-shot 示例样本。

\---

### 9.2 修改数据路径

修改：

```python
path='ais\_bench/datasets/SelfData/YourData.csv'
```

确保该路径能正确找到你的 CSV 或 JSONL 数据文件。

\---

### 9.3 修改输入字段

如果你的 CSV 字段名不是 `question`、`A`、`B`，需要修改：

```python
input\_columns=\['question', 'A', 'B']
```

例如你的字段是：

```csv
text,label\_a,label\_b,answer
```

则应修改为：

```python
input\_columns=\['text', 'label\_a', 'label\_b']
```

同时 Prompt 中也需要使用对应字段名：

```python
{text}
{label\_a}
{label\_b}
```

\---

### 9.4 修改答案字段

如果答案列不叫 `answer`，需要修改：

```python
output\_column='answer'
```

例如答案列叫 `label`，则修改为：

```python
output\_column='label'
```

\---

### 9.5 修改选项数量

当前代码默认支持 A/B 二分类。如果要改为 A/B/C/D 四选项，需要修改三处。

第一处，修改 Prompt 模板：

```python
QUERY\_TEMPLATE = """
请根据题目内容，从给定选项中选择一个最合适的答案。

请只输出最终答案，格式为：
答案: A
答案: B
答案: C
或
答案: D

题目：
{question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()
```

第二处，修改输入字段：

```python
input\_columns=\['question', 'A', 'B', 'C', 'D']
```

第三处，修改答案匹配规则：

```python
answer\_pattern=r'(?i)答案\\s\*\[:：]\\s\*\[\\W]\*(\[ABCD])\[\\W]\*'
```

\---

## 10\. 推荐目录结构

```text
ais\_bench/
├── datasets/
│   └── SelfData/
│       └── YourData.csv
├── configs/
│   └── datasets/
│       ├── SelfData\_0\_shot.py
│       └── SelfData\_5\_shot.py
```

实际目录结构可以根据项目要求调整，但需要保证配置文件中的 `path` 能正确找到数据文件。

\---

## 11\. 完整使用流程

```text
准备自定义 CSV 数据
→ 检查字段名和答案格式
→ 修改数据路径 path
→ 修改数据集简称 abbr
→ 修改 Prompt 模板 QUERY\_TEMPLATE
→ 选择 0-shot 或 5-shot 配置
→ 执行 ais\_bench 评测命令
→ 查看评测结果
```

\---

## 12\. 示例命令汇总

0-shot：

```bash
ais\_bench --models vllm\_api\_general\_chat --datasets SelfData\_gen\_0\_shot --work-dir path/to/result/dir
```

5-shot：

```bash
ais\_bench --models vllm\_api\_general\_chat --datasets SelfData\_gen\_5\_shot --work-dir path/to/result/dir
```

\---

