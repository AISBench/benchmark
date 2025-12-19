# Prompt 模板

## 背景

在语言模型的评测中，我们通常会将原始数据集按照一定的规则构造成提示词（Prompt），以便模型能够按照要求回答问题。

通常，我们会在提示词开头放入指令，接着放入几个 in-context example（上下文样例，也称为 shot case），最后放入待测试的题目。例如：

```text
Solve the following questions.
1+1=?
2
3+9=?
12
5+6=?
```

大量的实验表明，即便测试的原始题目相同，不同的提示词构造方式也会对模型的表现产生显著影响。可能影响的因素包括：

- **提示词的构成方式**：包括指令、in-context example、题目的写法
- **in-context example 的选择**：包括选择的数量和选择策略
- **提示词的使用方式**：是让模型基于提示词进行补全，还是从候选的提示词中选择一个最好的作为答案

AISBench 将提示词的构建策略定义在数据集配置中的 `infer_cfg` 部分。一个典型的 `infer_cfg` 配置如下所示：

```python
infer_cfg=dict(
    ice_template=dict(  # 用于构造 In Context Example (ice) 的模板
        type=PromptTemplate,
        template='{question}\n{answer}'
    ),
    prompt_template=dict(  # 用于构造主干 prompt 的模板
        type=PromptTemplate,
        template='Solve the following questions.\n</E>{question}\n{answer}',
        ice_token="</E>"
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1]),  # 定义 in context example 的获取方式
    inferencer=dict(type=GenInferencer),  # 使用何种推理器进行推理
)
```

本文档主要介绍 `ice_template` 和 `prompt_template` 的定义方法。关于其他组件的说明：

- **`retriever`（检索器）**：用于指定 shot case 的提取策略，详细介绍请参考 [retriever](./retriever.md)
- **`inferencer`（推理器）**：推理任务的具体实现，包含模型的调用和推理结果的保存逻辑，详细介绍请参考 [inferencer](../develop_guide/inferencer.md)

我们首先介绍 prompt 的基本语法。

## 字符串式 prompt

字符串式的模板是比较经典的模板形式，考虑下面的模板：

```python
reader_cfg=dict(
    input_columns=['anything', 'question'],
    output_column='answer',
)

prompt_template=dict(
    type=PromptTemplate,
    template="{anything}\nQuestion: {question}\nAnswer: {answer}"
)
```

其中 `reader_cfg` 需要定义清楚从数据集中读取哪些字段，以及将哪个字段作为答案。通常 `input_columns` 可以包含多个字段（列表或字符串），但 `output_column` 只能包含一个字段（字符串）。

运行时，花括号`{}`内的字段会被替换成数据样本内的对应字段。如果数据样本中没有对应的字段，则会保持原样输出。

例如我们有一个数据 example 如下:

```python
example = {
    'anything': 'blabla', # 假设 anything 被写在了 reader_cfg.input_columns 中
    'question': '1+1=?', # 假设 question 被写在了 reader_cfg.input_columns 中
    'answer': '2',  # 假设 answer 被写在了 reader_cfg.output_column 中
}
```

则填入模板后的结果为：

```text
blabla
Question: 1+1=?
Answer:
```

可以看到，问题的实际答案 `answer` 并没有出现在生成的结果中。这是因为 AISBench 会自动遮盖 `reader_cfg.output_column` 中指定的字段，避免答案泄露影响评测结果。

## 固定对话式 prompt

在实际的评测中，简单的补全式测试并不能很好地评估对话式模型的性能，因此我们希望提示词能够以对话的格式输入到模型中。另外，不同的模型对对话格式的定义也不一样，因此我们需要数据集侧产生的提示词更加通用，在测试时再结合具体模型的配置生成符合模型需求的提示词。

因此，AISBench 在字符串式模板之上，增加了对对话式模板的支持。对话式模板更加灵活，它可以结合模型侧不同的 [meta_template](./meta_template.md) 生成不同对话形式的提示词，同时适用于基座和对话模型，但定义也相对复杂。

现在，让我们假设有一个数据样本如下：

```python
example = {
    'anything': 'blabla', # 假设 anything 被写在了 reader_cfg.input_columns 中
    'question': '1+1=?', # 假设 question 被写在了 reader_cfg.input_columns 中
    'answer': '2',  # 假设 answer 被写在了 reader_cfg.output_column 中
}
```

接下来，我们来展示几个例子：

### 普通对话

```python
prompt_template=dict(
    type=PromptTemplate,
    template=dict(
        round=[
            dict(role="HUMAN", prompt="Question: {question}"),
            dict(role="BOT", prompt="Answer: {answer}"),
        ]
    )
)
infer_cfg = dict(
    prompt_template=prompt_template,
    retriever=...,
    inferencer=dict(type=GenInferencer, **args))
```

AISBench 把数据填入模板后得到的中间结果为：

```python
PromptList([
    dict(role='HUMAN', prompt='Question: 1+1=?'),
    dict(role='BOT', prompt='Answer: '),
])
```

### 多轮对话

```python
prompt_template=dict(
    type=PromptTemplate,
    template=dict(
        round=[
            dict(role="HUMAN", prompt="Question: 2+2=?"),
            dict(role="BOT", prompt="Answer: 4"),
            dict(role="HUMAN", prompt="Question: 3+3=?"),
            dict(role="BOT", prompt="Answer: 6"),
            dict(role="HUMAN", prompt="Question: {question}"),
            dict(role="BOT", prompt="Answer: {answer}"),
        ]
    )
)

infer_cfg = dict(
    prompt_template=prompt_template,
    retriever=...,
    inferencer=dict(type=GenInferencer, **args))
```

AISBench 把数据填入模板后得到的中间结果为：

```python
PromptList([
    dict(role='HUMAN', prompt='Question: 2+2=?'),
    dict(role='BOT', prompt='Answer: 4'),
    dict(role='HUMAN', prompt='Question: 3+3=?'),
    dict(role='BOT', prompt='Answer: 6'),
    dict(role='HUMAN', prompt='Question: 1+1=?'),
    dict(role='BOT', prompt='Answer: '),
])
```

### 带 SYSTEM 的对话

```python
prompt_template=dict(
    type=PromptTemplate,
    template=dict(
        begin=[
            dict(role='SYSTEM', fallback_role='HUMAN', prompt='Solve the following questions.'),
        ],
        round=[
            dict(role="HUMAN", prompt="Question: {question}"),
            dict(role="BOT", prompt="Answer: {answer}"),
        ]
    )
)
```

AISBench 把数据填入模板后得到的中间结果为：

```python
PromptList([
    dict(role='SYSTEM', fallback_role='HUMAN', prompt='Solve the following questions.'),
    dict(role='HUMAN', prompt='Question: 1+1=?'),
    dict(role='BOT', prompt='Answer: '),
])
```

在具体的 meta template 处理时，如果模型定义中存在 SYSTEM 角色，则会调用 SYSTEM 的模板进行处理。否则，会调用 `fallback_role` 指定的角色模板进行处理，也就是这个例子中的 HUMAN 角色。

可以看到，在对话式模板中，提示词是以不同角色（`role`）的对话形式进行组织的。在 AISBench 的预定义数据集配置中，常用的角色包括：

- **`HUMAN`**：人类角色，通常为提问的一方
- **`BOT`**：语言模型角色，通常为回答的一方
- **`SYSTEM`**：系统角色，通常用在提示词的开头，负责下达系统级别的指令

另外，与字符串式模板不同，经过对话式模板生成的提示词从固定的字符串变成了一个中间结构 `PromptList`。这个结构会进一步与模型侧的 [meta template](./meta_template.md) 相结合，最终拼装成模型所需的提示词格式（PromptType）。当前 meta template 已在模型实现中预设，同时也支持用户通过模型参数 `meta_template` 进行自定义。

```{note}
上面例子中 `PromptList` 的内容并非模型最终的输入，最终输入格式取决于 meta template 的处理。一个容易产生误解的地方是，在生成式评测中，最后一个 `BOT` 角色的 prompt（如 `Answer: `）**不会**实际输入到模型。这是因为 API 模型通常无法自定义模型回复的开头，因此这一设定保持了本地模型与 API 模型在评测行为上的一致性。更多信息可以参考 [meta template](./meta_template.md) 的文档。
```

<details>
<summary>点击查看完整参数介绍</summary>

- **`begin`、`end`**（list，可选）：提示词的开头和结尾，通常是一些系统级别的指令。列表中的每一项**可以是字典或字符串**。

- **`round`**（list）：对话的模板格式。列表中的每一项**必须是字典**。

每个字典的参数如下：

- **`role`**（str）：参与对话的角色名，用于与 `meta_template` 中的名称进行关联，不会影响实际生成的提示词内容。

- **`fallback_role`**（str，可选）：缺省角色名。如果 `meta_template` 中找不到 `role` 对应的角色，则会尝试使用 `fallback_role` 进行关联。默认为 `None`。

- **`prompt`**（str）：角色的对话内容。

</details>

## 多轮对话 prompt

多轮对话的数据样本中通常包含多个问题，需要按照顺序构造多轮对话的 prompt。AISBench 提供了多轮对话式模板（MultiTurnPromptTemplate）来支持多轮对话的构造。其构造方式与[固定对话式prompt](#固定对话式-prompt)一致，只是将round固定为：

```python
reader_cfg=dict(
    input_columns=['question'],
    output_column='answer',
)

prompt_template=dict(
    type=MultiTurnPromptTemplate,
    template=dict(round=[
    dict(role="HUMAN", prompt="{question}"),
    dict(role="BOT", prompt="{answer}"),
]))

infer_cfg = dict(
    prompt_template=prompt_template,
    retriever=...,
    inferencer=dict(type=MultiTurnGenInferencer, infer_mode="every", **args))

```

其中 `infer_mode` 表示多轮对话的请求拼接模式，当前支持三种配置：

- **`every`**：每次 `BOT` 都采用上一次模型的推理结果进行拼接
- **`every_with_gt`**：每次 `BOT` 都采用答案的 ground truth 进行拼接
- **`last`**：只采用最后一次模型的推理结果进行拼接

`question` 需要在数据样本中被定义为列表，每个元素为一个问题。

假设有如下样本：

```python
example = {
    'question': ['1+1=?', '2+2=?', '3+3=?'],
    'answer': ['2', '4', '6'],
}
```

则填入模板后的中间结果为：

`infer_mode=every` 时，总共包含三次推理过程：

```python
PromptList([
    dict(role='HUMAN', prompt='1+1=?')
])
# 模型返回：answer1
PromptList([
    dict(role='HUMAN', prompt='1+1=?'),
    dict(role='BOT', prompt='answer1'),
    dict(role='HUMAN', prompt='2+2=?')
])

# 模型返回：answer2
PromptList([
    dict(role='HUMAN', prompt='1+1=?'),
    dict(role='BOT', prompt='answer1')
    dict(role='HUMAN', prompt='2+2=?')
    dict(role='BOT', prompt='answer2'),
    dict(role='HUMAN', prompt='3+3=?')
])
```

`infer_mode=every_with_gt` 时，不关注模型上一轮推理结果，只会将答案的 ground truth 作为拼接，总共包含三次推理过程：

```python
PromptList([
    dict(role='HUMAN', prompt='1+1=?'),
])

PromptList([
    dict(role='HUMAN', prompt='1+1=?'),
    dict(role='BOT', prompt='2'),
    dict(role='HUMAN', prompt='2+2=?'),
])

PromptList([
    dict(role='HUMAN', prompt='1+1=?'),
    dict(role='BOT', prompt='2'),
    dict(role='HUMAN', prompt='2+2=?'),
    dict(role='BOT', prompt='4'),
    dict(role='HUMAN', prompt='3+3=?')
])
```

`infer_mode=last` 时，将除最后一轮答案外其他每一轮的文本和答案进行拼接，总共包含一次推理过程：

```python
PromptList([
    dict(role='HUMAN', prompt='1+1=?'),
    dict(role='BOT', prompt='2'),
    dict(role='HUMAN', prompt='2+2=?'),
    dict(role='BOT', prompt='4'),
    dict(role='HUMAN', prompt='3+3=?')
])
```

## 多模态 prompt

多模态模型不仅包含文本信息（text），还可能包含图像（image）、音频（audio）、视频（video）等多模态信息。这些信息无法采用字符串式模板进行构造，因此 AISBench 提供了多模态式模板（MMPromptTemplate）来支持多模态信息的构造。

多模态式模板的语法与对话式模板一致，只是将 `prompt` 替换为 `prompt_mm`。`prompt_mm` 是一个字典，键为模态类型，值为模态信息。

```python
reader_cfg=dict(
    input_columns=['anything', 'question'],
    output_column='answer',
)

# 多模态数据类型为url：
prompt_template=dict(
    type=MMPromptTemplate,
    template=dict(round=[dict(role="HUMAN",  prompt_mm={
                    "text": {"type": "text", "text": "{anything}\nQuestion: {question}"},
                    "image": {"type": "image_url", "image_url": {"url": "file://{image}"}},
                    "video": {"type": "video_url", "video_url": {"url": "file://{video}"}},
                    "audio": {"type": "audio_url", "audio_url": {"url": "file://{audio}"}}})])
)

# 多模态数据类型为base64：
prompt_template=dict(
    type=MMPromptTemplate,
    template=dict(round=[dict(role="HUMAN",  prompt_mm={
                    "text": {"type": "text", "text": "{anything}\nQuestion: {question}"},
                    "image": {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}},
                    "video": {"type": "video_url", "video_url": {"url": "data:video/jpeg;base64,{video}"}},
                    "audio": {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,{audio}"}}})])
)

infer_cfg = dict(
    prompt_template=prompt_template,
    retriever=...,
    inferencer=dict(type=GenInferencer, **args))
```

其中，`prompt_mm` 中的键为模态类型，值为模态信息。当前支持的模态类型为 `text`、`image`、`audio`、`video`。每个键值内部是一个包含 `type` 和对应类型值的字典。数据集中加载的数据类型需要与 `prompt_mm` 中的键值类型一致。此外，样本数据需要采用 [prompt.py](../../../ais_bench/benchmark/utils/prompt/prompt.py) 中定义的标识符进行拼接，标识不同数据内容的类别，如下所示：

- **text**：`<AIS_TEXT_START>{text_content}<AIS_CONTENT_TAG>`
- **image**：`<AIS_IMAGE_START>{image_content}<AIS_CONTENT_TAG>`
- **audio**：`<AIS_AUDIO_START>{audio_content}<AIS_CONTENT_TAG>`
- **video**：`<AIS_VIDEO_START>{video_content}<AIS_CONTENT_TAG>`

数据填充的内容根据模型实际输入的要求进行调整，例如对于图片输入，可以采用如下格式：
<AIS_IMAGE_START>image.jpg<AIS_CONTENT_TAG>
<AIS_IMAGE_START>file://image.jpg<AIS_CONTENT_TAG>
<AIS_IMAGE_START>data:image/jpeg;base64,base64_data<AIS_CONTENT_TAG>
<AIS_IMAGE_START>https://xxx.com/image.jpg<AIS_CONTENT_TAG>

让我们假设有一个数据样本如下：

```python
example = {
    'anything': 'blabla', # 假设 anything 被写在了 reader_cfg.input_columns 中
    'question': '<AIS_TEXT_START>What is this?<AIS_CONTENT_TAG><AIS_IMAGE_START>{image_data}<AIS_CONTENT_TAG><AIS_AUDIO_START>{audio_data}<AIS_CONTENT_TAG><AIS_VIDEO_START>{video_data}<AIS_CONTENT_TAG>', # 假设 question 被写在了 reader_cfg.input_columns 中，image_data为图片的base64数据或URL，audio_data为音频的base64数据或URL，video_data为视频的base64数据或URL
    'answer': 'a cat',  # 假设 answer 被写在了 reader_cfg.output_column 中
}
```

则填入模板后的结果会根据 `template` 格式进行填充。

多模态数据类型为 URL 时，中间结果为：

```json
PromptList([
    {"role": "HUMAN",
    "prompt": [
    {"type": "text", "text": "blabla\nQuestion: What is this?"},
    {"type": "image_url", "image_url": {"url": "{image_data}"}},
    {"type": "audio_url", "audio_url": {"url": "{audio_data}"}},
    {"type": "video_url", "video_url": {"url": "{video_data}"}}]
    }
])
```

多模态数据类型为 base64 时，中间结果如下：

```json

PromptList([
    {"role": "HUMAN",
    "prompt": [
    {"type": "text", "text": "blabla\nQuestion: What is this?"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_data}"}},
    {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,{audio_data}"}},
    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,{video_data}"}}]
    }
])
```

## `ice_template` 与 `prompt_template`

在 AISBench 中，对于 0-shot 的评测，我们通常只需要定义 `prompt_template` 字段，即可完成提示词的构造。但对于 few-shot（有 shot case 的评测），我们还需要定义 `ice_template` 字段，用于管理上下文学习中样例所对应的提示词模板。

`ice_template` 和 `prompt_template` 两者遵循的语法和规则一致，完整提示词的构造流程可以使用如下的伪代码表示：

```python
def build_prompt():
    ice = ice_template.format(*ice_example) # 根据ice_template格式化shot case
    prompt = prompt_template.replace(prompt_template.ice_token, ice).format(*prompt_example) # 将shot case的内容填充到prompt_template中
    return prompt
```

现在，让我们假设有两个训练数据作为shot case (ex1, ex2) 和一个测试数据 (ex3):

```python
ex1 = {
    'question': '2+2=?',
    'answer': '4',
    'irrelavent_infos': 'blabla',
}
ex2 = {
    'question': '3+3=?',
    'answer': '6',
    'irrelavent_infos': 'blabla',
}
ex3 = {
    'question': '1+1=?',
    'answer': '2',  # 假设 answer 被写在了 reader_cfg.output_column 中
    'irrelavent_infos': 'blabla',
}
```

接下来，我们看一下不同的 prompt 构造方法对应的实际效果：

### 字符串式模板

模板配置如下：

```python
infer_cfg=dict(
    ice_template=dict(
        type=PromptTemplate,
        template='{question}\n{answer}'
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template='Solve the following questions.\n</E>{question}\n{answer}',
        ice_token='</E>',
    )
)
```

会得到以下字符串：

```text
Solve the following questions.
2+2=?
4
3+3=?
6
1+1=?
```

### 对话式模板

模板配置如下：

```python
infer_cfg=dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt="{question}"),
                dict(role="BOT", prompt="{answer}"),
            ]
        )
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt='Solve the following questions.'),
                '</E>',
            ],
            round=[
                dict(role="HUMAN", prompt="{question}"),
                dict(role="BOT", prompt="{answer}"),
            ],
        ),
        ice_token='</E>',
    )
)
```

AISBench 把数据填入模板后得到的中间结果为：

```python
PromptList([
    dict(role='SYSTEM', fallback_role='HUMAN', prompt='Solve the following questions.'),
    dict(role='HUMAN', prompt='2+2=?'),
    dict(role='BOT', prompt='4'),
    dict(role='HUMAN', prompt='3+3=?'),
    dict(role='BOT', prompt='6'),
    dict(role='HUMAN', prompt='1+1=?'),
    dict(role='BOT', prompt=''),
])
```

### 省略式使用方法

值得一提的是，为了简化配置文件，`prompt_template` 这一字段是可被省略的。当 `prompt_template` 字段被省略时，`ice_template` 会同时被作为 `prompt_template`，用于拼装得到完整的提示词。以下两份 `infer_cfg` 配置是等价的：

<table class="docutils">
  <thead>
  <tr>
      <th>完整写法</th>
      <th>省略写法</th>
  <tbody>
  <tr>
  <td>

```python
infer_cfg=dict(
    ice_template=dict(
        type=PromptTemplate,
        template="Q: {question}\nA: {answer}",
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template="</E>Q: {question}\nA: {answer}",
        ice_token="</E>",
    ),
    # ...
)
```

</td>
  <td>

```python
infer_cfg=dict(
    ice_template=dict(
        type=PromptTemplate,
        template="</E>Q: {question}\nA: {answer}",
        ice_token="</E>",
    ),
    # ...
)
```

</td>
  </tr>
  </thead>
  </table>

更一般地，即便在 0-shot learning 的情况下（即 `retriever` 为 `ZeroRetriever`）时，这一机制依然生效。因此以下配置也是合法的：

```python
datasets = [
    dict(
        infer_cfg=dict(
            ice_template=dict(
                type=PromptTemplate,
                template="Q: {question}\nA: {answer}",
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        )
    ),
]
```
