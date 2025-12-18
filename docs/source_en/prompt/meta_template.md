# Meta Template

## Background

In the supervised fine-tuning (SFT) process of language models (LLMs), we often inject some predefined strings into conversations according to actual needs, so that models can output content according to certain requirements. For example, in the fine-tuning of some chat models, we may add system-level instructions at the beginning of each conversation and agree on a format to represent the dialogue between users and models. In a conversation, the text format expected by the model might be as follows:

```Bash
Meta instruction: You are now a helpful and harmless AI assistant.
HUMAN: Hi!<eoh>\n
Bot: Hello! How may I assist you?<eob>\n
```

During evaluation, we also need to input questions in the agreed format so that models can perform at their best.

In addition, API models have similar situations. Generally, API chat models allow users to pass conversation history when calling, and some models also allow passing SYSTEM-level instructions. To better evaluate the capabilities of API models, we hope that when evaluating API models, we can make the data more aligned with the multi-turn dialogue template of the API model itself, rather than stuffing all content into a single instruction.

Therefore, we need to specify different parsing templates for different models. In AISBench, we call this set of parsing templates **Meta Template**. Meta Template is bound to model configuration and combined with the dataset's dialogue template at runtime to ultimately produce prompts most suitable for the current model.

```python
# When specifying, just pass the meta_template field to model configuration
models = [
    dict(
        type='AnyModel',
        meta_template=...,  # meta template
    )
]
```

Next, we will introduce the configuration methods of Meta Template for two types of models. It is recommended that readers understand the basic syntax of [prompt_template](./prompt_template.md) before reading this chapter.

```{note}
In some cases (such as testing base models), we don't need to inject any instructions in normal conversations. In this case, we can leave the meta template empty. In this situation, the prompt received by the model is only defined by the dataset configuration and is an ordinary string. If the dataset configuration uses a dialogue template, speeches from different roles will be concatenated by `\n`.
```

## Application to Language Models

The following figure shows several cases of data from datasets going through prompt template and meta template to eventually construct prompts in the case of 2-shot learning. Readers can use this figure as a reference to facilitate understanding of subsequent chapters.

![Prompt Template and Meta Template Processing Flow](https://user-images.githubusercontent.com/22607038/251195073-85808807-6359-44df-8a19-9f5d00c591ec.png)

We will explain the definition methods of meta template with several examples.

Assume that according to the dataset's dialogue template, the following PromptList is generated:

```python
PromptList([
    dict(role='HUMAN', prompt='1+1=?'),
    dict(role='BOT', prompt='2'),
    dict(role='HUMAN', prompt='2+2=?'),
    dict(role='BOT', prompt='4'),
])
```

We hope to pass this conversation to a model that has been fine-tuned with SFT. In the conversation format agreed by the model, speeches from different roles start with `<role name>:` and end with a special token and `\n`. The following is the complete string expected by the model:

```Plain
<HUMAN>: 1+1=?<eoh>
<BOT>: 2<eob>
<HUMAN>: 2+2=?<eoh>
<BOT>: 4<eob>
```

In the meta template, we only need to abstract the format of each round of dialogue into the following configuration:

```python
# model meta template
meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<HUMAN>: ', end='<eoh>\n'),
        dict(role='BOT', begin='<BOT>: ', end='<eob>\n'),
    ],
)
```

______________________________________________________________________

Some datasets may introduce SYSTEM-level roles:

```python
PromptList([
    dict(role='SYSTEM', fallback_role='HUMAN', prompt='Solve the following math questions'),
    dict(role='HUMAN', prompt='1+1=?'),
    dict(role='BOT', prompt='2'),
    dict(role='HUMAN', prompt='2+2=?'),
    dict(role='BOT', prompt='4'),
])
```

Assume the model also accepts the SYSTEM role and expects input as:

```Bash
<SYSTEM>: Solve the following math questions<eosys>\n
<HUMAN>: 1+1=?<eoh>\n
<BOT>: 2<eob>\n
<HUMAN>: 2+2=?<eoh>\n
<BOT>: 4<eob>\n
end of conversation
```

We can put the SYSTEM role definition in `reserved_roles`. Roles in `reserved_roles` do not appear in regular conversations but allow the dataset configuration's dialogue template to call them in `begin` or `end`.

```python
# model meta template
meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<HUMAN>: ', end='<eoh>\n'),
        dict(role='BOT', begin='<BOT>: ', end='<eob>\n'),
    ],
    reserved_roles=[dict(role='SYSTEM', begin='<SYSTEM>: ', end='<eosys>\n')],
)
```

If the model does not accept the SYSTEM role, it is **not necessary** to configure this, and it can still run normally. In this case, the string received by the model becomes:

```plain
<HUMAN>: Solve the following math questions<eoh>\n
<HUMAN>: 1+1=?<eoh>\n
<BOT>: 2<eob>\n
<HUMAN>: 2+2=?<eoh>\n
<BOT>: 4<eob>\n
end of conversation
```

This is because in AISBench's predefined datasets, each `SYSTEM` speech has a `fallback_role='HUMAN'`, meaning if the `SYSTEM` role does not exist in the meta template, the speaker will be switched to the `HUMAN` role.

______________________________________________________________________

Some models may also need to embed other strings at the beginning or end of conversations, such as system instructions:

```Bash
Meta instruction: You are now a helpful and harmless AI assistant.
<SYSTEM>: Solve the following math questions<eosys>\n
<HUMAN>: 1+1=?<eoh>\n
<BOT>: 2<eob>\n
<HUMAN>: 2+2=?<eoh>\n
<BOT>: 4<eob>\n
end of conversation
```

At this time, we can specify these strings by specifying `begin` and `end` parameters.

```python
meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<HUMAN>: ', end='<eoh>\n'),
        dict(role='BOT', begin='<BOT>: ', end='<eob>\n'),
    ],
    reserved_roles=[dict(role='SYSTEM', begin='<SYSTEM>: ', end='<eosys>\n')],
    begin="Meta instruction: You are now a helpful and harmless AI assistant.",
    end="end of conversation",
)
```

______________________________________________________________________

In **generative** task evaluation, we also do not directly input answers to the model, but truncate the prompt, keeping the context while leaving the model's output answer blank.

```Bash
Meta instruction: You are now a helpful and harmless AI assistant.
<SYSTEM>: Solve the following math questions<eosys>\n
<HUMAN>: 1+1=?<eoh>\n
<BOT>: 2<eob>\n
<HUMAN>: 2+2=?<eoh>\n
<BOT>:
```

We only need to set the `generate` field to `True` in the BOT configuration, and AISBench will leave the last BOT sentence for the model to generate:

```python
meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<HUMAN>: ', end='<eoh>\n'),
        dict(role='BOT', begin='<BOT>: ', end='<eob>\n', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', begin='<SYSTEM>: ', end='<eosys>\n')],
    begin="Meta instruction: You are now a helpful and harmless AI assistant.",
    end="end of conversation",
)
```

Note that `generate` only affects generative inference. When performing discriminative inference (such as perplexity calculation), the prompt received by the model is still complete.

### Complete Field Introduction

```Bash
models = [
    dict(meta_template = dict(
            begin="Meta instruction: You are now a helpful and harmless AI assistant.",
            round=[
                    dict(role='HUMAN', begin='HUMAN: ', end='<eoh>\n'),  # begin and end can be strings or integer lists
                    dict(role='THOUGHTS', begin='THOUGHTS: ', end='<eot>\n', prompt='None'), # Default prompt can be set here, may be overridden by specific dataset configuration
                    dict(role='BOT', begin='BOT: ', generate=True, end='<eob>\n'),
            ],
            end="end of conversion",
            reserved_roles=[dict(role='SYSTEM', begin='SYSTEM: ', end='\n'),],
            eos_token_id=10000,
         ),
     )
]
```

meta_template is a dictionary that can contain the following fields:

- **`begin`、`end`** (str, optional): Beginning and end of prompts, usually some system-level instructions.

- **`round`** (list): Template format for each round of dialogue. The prompt content for each round of dialogue is controlled by the dataset configuration's dialogue template.

- **`reserved_roles`** (list, optional): Specify roles that do not appear in `round` but may be used in dataset configuration, such as the `SYSTEM` role.

- **`eos_token_id`** (int, optional): Specify the id of the model's eos token. If not set, it defaults to the eos token id in the tokenizer. Its main function is to truncate the model's output results in generative tasks, so it should generally be set to the first token id of the `end` corresponding to the item with `generate=True`.

meta_template's `round` specifies the format of each role speaking in a round of dialogue, accepting a list of dictionaries, with each dictionary's fields as follows:

- **`role`** (str): Name of the role participating in the dialogue, this string does not affect the actual prompt content.

- **`begin`、`end`** (str): Specify the fixed beginning or end when this role speaks.

- **`prompt`** (str, optional): The role's prompt. It can be left empty in the meta template, but must be specified in the dataset configuration's prompt.

- **`generate`** (bool): When set to `True`, this role is the role played by the model. In generative tasks, the prompt received by the model will be truncated at this role's `begin`, and the remaining content will be completed by the model.

## Application to API Models

The meta template for API models is similar to that of regular models, but the configuration is simpler. Users can directly use one of the following two configurations according to the situation to evaluate API models in a multi-turn dialogue manner:

```python
# If API model does not support system instructions
meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True)
    ],
)

# If API model supports system instructions
meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True)
    ],
    reserved_roles=[
        dict(role='SYSTEM', api_role='SYSTEM'),
    ],
)
```

### Principle

Although different API models accept different data structures, there are commonalities overall. Interfaces that accept conversation history usually allow users to pass prompts for the following three roles:

- **User** (HUMAN)
- **Bot** (BOT)
- **System** (SYSTEM, optional)

Based on this, AISBench presets three `api_role` values for API models: `HUMAN`, `BOT`, `SYSTEM`, and also agrees that API models accept input in addition to ordinary strings, there is also an intermediate format represented by the `PromptList` structure for conversations. API models will repackage conversations in multi-turn dialogue format and send them to the backend. But to activate this function, users need to map the `role` in the dataset prompt template to the corresponding `api_role` in the meta template above. The following figure shows the relationship between the input accepted by API models and Prompt Template and Meta Template.

![API Model Prompt Template and Meta Template Relationship](https://user-images.githubusercontent.com/22607038/251195872-63aa7d30-045a-4837-84b5-11b09f07fb18.png)

