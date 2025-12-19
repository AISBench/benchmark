# Prompt Template

## Background

In language model evaluation, we usually construct prompts according to certain rules from raw datasets so that models can answer questions according to requirements.

Usually, we put instructions at the beginning of prompts, then put a few in-context examples (also called shot cases), and finally put the test question. For example:

```text
Solve the following questions.
1+1=?
2
3+9=?
12
5+6=?
```

Extensive experiments show that even when the raw test questions are the same, different prompt construction methods can significantly affect model performance. Factors that may affect include:

- **Prompt composition method**: Including the writing of instructions, in-context examples, and questions
- **Selection of in-context examples**: Including the number and selection strategy
- **Usage of prompts**: Whether to let the model complete based on prompts, or select the best one from candidate prompts as the answer

AISBench defines prompt construction strategies in the `infer_cfg` section of dataset configuration. A typical `infer_cfg` configuration is as follows:

```python
infer_cfg=dict(
    ice_template=dict(  # Template for constructing In Context Example (ice)
        type=PromptTemplate,
        template='{question}\n{answer}'
    ),
    prompt_template=dict(  # Template for constructing main prompt
        type=PromptTemplate,
        template='Solve the following questions.\n</E>{question}\n{answer}',
        ice_token="</E>"
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1]),  # Define how to get in context examples
    inferencer=dict(type=GenInferencer),  # Which inferencer to use for inference
)
```

This document mainly introduces the definition methods of `ice_template` and `prompt_template`. For descriptions of other components:

- **`retriever` (Retriever)**: Used to specify shot case extraction strategy, detailed introduction please refer to [retriever](./retriever.md)
- **`inferencer` (Inferencer)**: Specific implementation of inference tasks, including model invocation and inference result saving logic, detailed introduction please refer to [inferencer](../develop_guide/inferencer.md)

We first introduce the basic syntax of prompts.

## String-style Prompt

String-style templates are a classic template form. Consider the following template:

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

Where `reader_cfg` needs to clearly define which fields to read from the dataset and which field to use as the answer. Usually `input_columns` can contain multiple fields (list or string), but `output_column` can only contain one field (string).

At runtime, fields within curly braces `{}` will be replaced with corresponding fields in data samples. If there is no corresponding field in the data sample, it will remain unchanged in the output.

For example, we have a data example as follows:

```python
example = {
    'anything': 'blabla', # Assume anything is written in reader_cfg.input_columns
    'question': '1+1=?', # Assume question is written in reader_cfg.input_columns
    'answer': '2',  # Assume answer is written in reader_cfg.output_column
}
```

The result after filling the template is:

```text
blabla
Question: 1+1=?
Answer:
```

It can be seen that the actual answer `answer` does not appear in the generated result. This is because AISBench automatically masks the field specified in `reader_cfg.output_column` to avoid answer leakage affecting evaluation results.

## Fixed Dialogue-style Prompt

In actual evaluation, simple completion tests cannot well evaluate the performance of dialogue models, so we hope prompts can be input to models in dialogue format. In addition, different models have different definitions of dialogue formats, so we need prompts generated on the dataset side to be more universal, and then combine with specific model configurations at test time to generate prompts that meet model requirements.

Therefore, AISBench adds support for dialogue-style templates on top of string-style templates. Dialogue-style templates are more flexible. They can combine with different [meta_template](./meta_template.md) on the model side to generate prompts in different dialogue forms, and are suitable for both base and dialogue models, but the definition is relatively complex.

Now, let's assume we have a data sample as follows:

```python
example = {
    'anything': 'blabla', # Assume anything is written in reader_cfg.input_columns
    'question': '1+1=?', # Assume question is written in reader_cfg.input_columns
    'answer': '2',  # Assume answer is written in reader_cfg.output_column
}
```

Next, we show several examples:

### Ordinary Dialogue

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

The intermediate result after AISBench fills the template with data is:

```python
PromptList([
    dict(role='HUMAN', prompt='Question: 1+1=?'),
    dict(role='BOT', prompt='Answer: '),
])
```

### Multi-turn Dialogue

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

The intermediate result after AISBench fills the template with data is:

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

### Dialogue with SYSTEM

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

The intermediate result after AISBench fills the template with data is:

```python
PromptList([
    dict(role='SYSTEM', fallback_role='HUMAN', prompt='Solve the following questions.'),
    dict(role='HUMAN', prompt='Question: 1+1=?'),
    dict(role='BOT', prompt='Answer: '),
])
```

In specific meta template processing, if the SYSTEM role exists in the model definition, the SYSTEM template will be called for processing. Otherwise, the role template specified by `fallback_role` will be called for processing, which is the HUMAN role in this example.

It can be seen that in dialogue-style templates, prompts are organized in the form of dialogues from different roles (`role`). In AISBench's predefined dataset configurations, commonly used roles include:

- **`HUMAN`**: Human role, usually the questioning party
- **`BOT`**: Language model role, usually the answering party
- **`SYSTEM`**: System role, usually used at the beginning of prompts, responsible for issuing system-level instructions

In addition, unlike string-style templates, prompts generated through dialogue-style templates change from fixed strings to an intermediate structure `PromptList`. This structure will further combine with the model-side [meta template](./meta_template.md) to eventually assemble into the prompt format required by the model (PromptType). Currently, meta templates are preset in model implementations and also support user customization through the model parameter `meta_template`.

```{note}
The content of `PromptList` in the examples above is not the final input to the model. The final input format depends on meta template processing. One place that can easily cause misunderstanding is that in generative evaluation, the prompt of the last `BOT` role (such as `Answer: `) will **not** actually be input to the model. This is because API models usually cannot customize the beginning of model replies, so this setting maintains consistency in evaluation behavior between local models and API models. For more information, please refer to the [meta template](./meta_template.md) documentation.
```

<details>
<summary>Click to view complete parameter introduction</summary>

- **`begin`、`end`** (list, optional): Beginning and end of prompts, usually some system-level instructions. Each item in the list **can be a dictionary or string**.

- **`round`** (list): Dialogue template format. Each item in the list **must be a dictionary**.

Parameters for each dictionary are as follows:

- **`role`** (str): Name of the role participating in the dialogue, used to associate with names in `meta_template`, does not affect actual generated prompt content.

- **`fallback_role`** (str, optional): Default role name. If the `role` corresponding role is not found in `meta_template`, it will try to use `fallback_role` for association. Default is `None`.

- **`prompt`** (str): Dialogue content of the role.

</details>

## Multi-turn Dialogue Prompt

Multi-turn dialogue data samples usually contain multiple questions and need to construct multi-turn dialogue prompts in order. AISBench provides multi-turn dialogue templates (MultiTurnPromptTemplate) to support multi-turn dialogue construction. Its construction method is consistent with [fixed dialogue-style prompt](#fixed-dialogue-style-prompt), except that round is fixed as:

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

Where `infer_mode` represents the request concatenation mode for multi-turn dialogues. Currently, three configurations are supported:

- **`every`**: Each `BOT` uses the previous model inference result for concatenation
- **`every_with_gt`**: Each `BOT` uses the ground truth of the answer for concatenation
- **`last`**: Only uses the last model inference result for concatenation

`question` needs to be defined as a list in data samples, with each element being a question.

Assume we have the following sample:

```python
example = {
    'question': ['1+1=?', '2+2=?', '3+3=?'],
    'answer': ['2', '4', '6'],
}
```

The intermediate result after filling the template is:

When `infer_mode=every`, it contains three inference processes in total:

```python
PromptList([
    dict(role='HUMAN', prompt='1+1=?')
])
# Model returns: answer1
PromptList([
    dict(role='HUMAN', prompt='1+1=?'),
    dict(role='BOT', prompt='answer1'),
    dict(role='HUMAN', prompt='2+2=?')
])

# Model returns: answer2
PromptList([
    dict(role='HUMAN', prompt='1+1=?'),
    dict(role='BOT', prompt='answer1')
    dict(role='HUMAN', prompt='2+2=?')
    dict(role='BOT', prompt='answer2'),
    dict(role='HUMAN', prompt='3+3=?')
])
```

When `infer_mode=every_with_gt`, it does not care about the model's previous round inference result, only concatenates the ground truth of answers, containing three inference processes in total:

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

When `infer_mode=last`, it concatenates the text and answers of each round except the last answer, containing one inference process in total:

```python
PromptList([
    dict(role='HUMAN', prompt='1+1=?'),
    dict(role='BOT', prompt='2'),
    dict(role='HUMAN', prompt='2+2=?'),
    dict(role='BOT', prompt='4'),
    dict(role='HUMAN', prompt='3+3=?')
])
```

## Multimodal Prompt

Multimodal models not only contain text information (text), but may also contain multimodal information such as images (image), audio (audio), and videos (video). This information cannot be constructed using string-style templates, so AISBench provides multimodal templates (MMPromptTemplate) to support multimodal information construction.

The syntax of multimodal templates is consistent with dialogue templates, except that `prompt` is replaced with `prompt_mm`. `prompt_mm` is a dictionary where keys are modality types and values are modality information.

```python
reader_cfg=dict(
    input_columns=['anything', 'question'],
    output_column='answer',
)

# Multimodal data type is url:
prompt_template=dict(
    type=MMPromptTemplate,
    template=dict(round=[dict(role="HUMAN",  prompt_mm={
                    "text": {"type": "text", "text": "{anything}\nQuestion: {question}"},
                    "image": {"type": "image_url", "image_url": {"url": "file://{image}"}},
                    "video": {"type": "video_url", "video_url": {"url": "file://{video}"}},
                    "audio": {"type": "audio_url", "audio_url": {"url": "file://{audio}"}}})])
)

# Multimodal data type is base64:
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

Where keys in `prompt_mm` are modality types and values are modality information. Currently supported modality types are `text`, `image`, `audio`, `video`. Each key-value pair internally is a dictionary containing `type` and the corresponding type value. The data types loaded in the dataset need to be consistent with the key-value types in `prompt_mm`. In addition, sample data needs to use identifiers defined in [prompt.py](../../../ais_bench/benchmark/utils/prompt/prompt.py) for concatenation to identify different data content categories, as shown below:

- **text**: `<AIS_TEXT_START>{text_content}<AIS_CONTENT_TAG>`
- **image**: `<AIS_IMAGE_START>{image_content}<AIS_CONTENT_TAG>`
- **audio**: `<AIS_AUDIO_START>{audio_content}<AIS_CONTENT_TAG>`
- **video**: `<AIS_VIDEO_START>{video_content}<AIS_CONTENT_TAG>`

The data filling content is adjusted according to the actual input requirements of the model. For example, for image input, the following formats can be used:
<AIS_IMAGE_START>image.jpg<AIS_CONTENT_TAG>
<AIS_IMAGE_START>file://image.jpg<AIS_CONTENT_TAG>
<AIS_IMAGE_START>data:image/jpeg;base64,base64_data<AIS_CONTENT_TAG>
<AIS_IMAGE_START>https://xxx.com/image.jpg<AIS_CONTENT_TAG>

Let's assume we have a data sample as follows:

```python
example = {
    'anything': 'blabla', # Assume anything is written in reader_cfg.input_columns
    'question': '<AIS_TEXT_START>What is this?<AIS_CONTENT_TAG><AIS_IMAGE_START>{image_data}<AIS_CONTENT_TAG><AIS_AUDIO_START>{audio_data}<AIS_CONTENT_TAG><AIS_VIDEO_START>{video_data}<AIS_CONTENT_TAG>', # Assume question is written in reader_cfg.input_columns, image_data is base64 data or URL of image, audio_data is base64 data or URL of audio, video_data is base64 data or URL of video
    'answer': 'a cat',  # Assume answer is written in reader_cfg.output_column
}
```

The result after filling the template will be filled according to the `template` format.

When multimodal data type is URL, the intermediate result is:

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

When multimodal data type is base64, the intermediate result is as follows:

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

## `ice_template` and `prompt_template`

In AISBench, for 0-shot evaluation, we usually only need to define the `prompt_template` field to complete prompt construction. But for few-shot (evaluation with shot cases), we also need to define the `ice_template` field to manage the prompt template corresponding to in-context learning examples.

`ice_template` and `prompt_template` follow the same syntax and rules. The complete prompt construction process can be represented by the following pseudo-code:

```python
def build_prompt():
    ice = ice_template.format(*ice_example) # Format shot case according to ice_template
    prompt = prompt_template.replace(prompt_template.ice_token, ice).format(*prompt_example) # Fill shot case content into prompt_template
    return prompt
```

Now, let's assume we have two training data as shot cases (ex1, ex2) and one test data (ex3):

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
    'answer': '2',  # Assume answer is written in reader_cfg.output_column
    'irrelavent_infos': 'blabla',
}
```

Next, let's look at the actual effects corresponding to different prompt construction methods:

### String-style Template

Template configuration is as follows:

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

Will get the following string:

```text
Solve the following questions.
2+2=?
4
3+3=?
6
1+1=?
```

### Dialogue-style Template

Template configuration is as follows:

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

The intermediate result after AISBench fills the template with data is:

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

### Omitted Usage Method

It is worth mentioning that to simplify configuration files, the `prompt_template` field can be omitted. When the `prompt_template` field is omitted, `ice_template` will be used as `prompt_template` at the same time to assemble the complete prompt. The following two `infer_cfg` configurations are equivalent:

<table class="docutils">
  <thead>
  <tr>
      <th>Complete Writing</th>
      <th>Omitted Writing</th>
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

More generally, this mechanism still works even in 0-shot learning cases (i.e., when `retriever` is `ZeroRetriever`). Therefore, the following configuration is also valid:

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

