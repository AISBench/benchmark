# Reproducing Dataset Evaluation Results from Large Language Model (LLM) Papers (Technical Reports) — Taking the GPQA Dataset Used by DeepSeek R1 as an Example

## Preface - Methodology
To reproduce the accuracy results reported in papers using the AISBench evaluation tool, it is essential to align with the testing methodology for the dataset as described in the model’s technical report or paper. The following configurations in the evaluation tool need to be aligned accordingly:

### Model - Related Configurations
- Select the appropriate model task corresponding to the endpoint
- Fully align the maximum output length
- Fully align the post - processing parameters

### Dataset - Related Configurations
- Fully align the prompt engineering
- Fully align the answer extraction method
- Align the accuracy evaluation metrics

---

## Example: Reproducing the Evaluation Results of the DeepSeek R1 Model on the GPQA Dataset
### Select the Appropriate Model Configuration File Corresponding to the Endpoint
For execution efficiency, inference services are generally used as the subjects under test when reproducing model accuracy. Inference services can be accessed via various endpoints, and the industry standard mainly adopts OpenAI - style endpoints. There are two primary OpenAI endpoints: `v1/completions` and `v1/chat/completions`.

- **v1/completions**:
    The model generates text based on a "prefix continuation" logic and does not inherently distinguish between "instructions" and "content". Strong guidance through prompt engineering (e.g., adding "Please answer:") is required; otherwise, it may produce imitative outputs rather than executing instructions. For instance, inputting "Translate the following English to Chinese: Hello" might result in the continuation "Translate the following Chinese to English: Nihao" instead of a direct translation.
    Therefore, it is suitable for single - turn text generation tasks (such as code completion, short - text writing, text continuation, and simple text classification) or scenarios that need to be compatible with legacy base models.

- **v1/chat/completions**:
    The model natively understands the semantic roles of system/user/assistant, prioritizes executing user instructions, and ensures more stable dialogue consistency and intent alignment. It can complete tasks like translation and summarization without complex prompt wrapping.
    Hence, it is ideal for modern LLM application scenarios such as multi - turn dialogues (customer service, chatbots), instruction - driven tasks (translation, summarization, data analysis), tool integration (function calling, retrieval - augmented generation), and multimodal interactions.

💡 As of January 2025, nearly all newly released LLM models support the `v1/chat/completions` endpoint, and the `v1/completions` endpoint has been largely deprecated. Consequently, model configuration files typically only use the model tasks for accessing the `v1/chat/completions` endpoint: **vllm_api_general_chat** (accessing the service via a non - streaming interface) and **vllm_api_stream_chat** (accessing the service via a streaming interface).

Taking the model task `vllm_api_general_chat` as an example, the absolute path to its corresponding model configuration file can be obtained by running the following command:
```bash
ais_bench --models vllm_api_general_chat --search
```

⚠️ All subsequent model - related configurations will be modified in this configuration file.

---

### Fully Align the Maximum Output Length
The following description can be found on the [DeepSeek R1 Hugging Face Model Card](https://huggingface.co/deepseek - ai/DeepSeek - R1):
> ## 4. Evaluation Results
> ### DeepSeek - R1 - Evaluation
> For all our models, the maximum generation length is set to 32,768 tokens....

This indicates that the maximum output length of the DeepSeek R1 model is set to 32,768 tokens.

Taking the model task `vllm_api_general_chat` as an example, the configuration for the maximum output length is as follows:
```python
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm - api - general - chat',
        # ......
        max_out_len=32768,          # Maximum number of tokens output by the inference service
        # ......
    )
]
```

---

### Fully Align the Post - processing Parameters
The following description is available on the [DeepSeek R1 Hugging Face Model Card](https://huggingface.co/deepseek - ai/DeepSeek - R1):
> ## 4. Evaluation Results
> ### DeepSeek - R1 - Evaluation
> ..., For benchmarks requiring sampling, we use a temperature of $0.6$, a top - p value of $0.95$, ...

It can be seen from this that the post - processing parameters of the DeepSeek R1 model include a temperature of 0.6 and a top - p value of 0.95.

Taking the model task `vllm_api_general_chat` as an example, the configuration for the post - processing parameters is as follows:
```python
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm - api - general - chat',
        # ......
        temperature=0.6,           # Sampling temperature for text generation
        top_p=0.95,                # Top - p sampling parameter
        # ......
    )
]
```

---

### Fully Align Prompt Engineering
In the [DeepSeek R1 technical report](https://github.com/deepseek - ai/DeepSeek - R1/blob/main/DeepSeek_R1.pdf), the prompt format for the GPQA dataset is specified as follows:
> For GPQA, we use the 0 - shot chain - of - thought (CoT) prompt from the original GPQA paper. The prompt template is as follows:
> Q: [question]
> A: Let's think step by step.

In the AISBench dataset configuration file, the prompt engineering can be aligned by modifying the reader configuration, as shown below:
```python
# https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/gpqa/gpqa_gen_0_shot_cot_chat_prompt.py

gpqa_reader_cfg = dict(
    # ......
    prompt_template='Q: {question}\nA: Let\'s think step by step.',
    # ......
)
```

---

### Fully Align the Answer Extraction Method
The answer format in the GPQA dataset is option - based (options A, B, C, D). In the DeepSeek R1 paper, the answer extraction method is to extract the final answer option (A/B/C/D) from the model - generated reasoning process.

Therefore, in AISBench, a custom post - processing function for answer extraction needs to be implemented in the dataset configuration file, as shown below:
```python
# https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/gpqa/gpqa_gen_0_shot_cot_chat_prompt.py

import re

def gpqa_extract_answer(text):
    """
    Extract the final answer option (A/B/C/D) from the model - generated reasoning text
    """
    ANSWER_PATTERN = r"Answer[ \t]*:[ \t]*\$?([A - D])\$?"
    match = re.search(ANSWER_PATTERN, text)
    if match:
        return match.group(1)
    return None

from ais_bench.benchmark.datasets import GPQADataset, GPQA_Simple_Eval_postprocess, GPQAEvaluator

gpqa_eval_cfg = dict(evaluator=dict(type=GPQAEvaluator),
                     pred_postprocessor=dict(type=GPQA_Simple_Eval_postprocess, func=gpqa_extract_answer)) # Pass in the custom answer extraction function, which can also be directly defined in the dataset configuration file
```

---

### Align the Accuracy Evaluation Metrics
Typically, model evaluation results are presented in a table. Take the results from DeepSeek as an example:

| Model | AIME 2024 pass@1 | AIME 2024 cons@64 | MATH - 500 pass@1 | GPQA Diamond pass@1 | LiveCodeBench pass@1 | CodeForces rating |
| ----- | ---------------- | ----------------- | ----------------- | ------------------- | -------------------- | ----------------- |
| GPT - 4o - 0513 | 9.3              | 13.4              | 74.6              | 49.9                | 32.9                 | 759               |
| Claude - 3.5 - Sonnet - 1022 | 16.0             | 26.7              | 78.3              | 65.0                | 38.9                 | 717               |
| o1 - mini | 63.6             | 80.0              | 90.0              | 60.0                | 53.8                 | 1820              |

Here, `cons@64` and `pass@1` represent accuracy evaluation metrics. For detailed explanations of these metrics, refer to [Accuracy Metric Description](../base_tutorials/results_intro/accuracy_metric.md#ii - definition - and - relationship - between - passk - consk - and - avgn).

Taking GPQA as an example, the table shows that `pass@1` is used as the accuracy evaluation metric. The description of pass@1 in the DeepSeek R1 paper is as follows:
> ..., and report pass@1 using a non - zero temperature. Specifically, we use a sampling temperature of 0.6 and a top - 𝑝 value of 0.95 to generate 𝑘 responses (typically between 4 and 64, depending on the test set size) for each question. Pass@1 is then calculated as
>  ${\text{pass@1}} = \frac{1}{n} \sum_{i = 1}^{n} p_i$

Then in AISBench, configure the model configuration file as follows:
```python
# https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py

models = [
    dict(
        ... # Other parameters
        generation_kwargs = dict(
            num_return_sequences = 4, # n=4~64
            ... # Other parameters
        ),
        ...
    )
]

```
Under normal circumstances, `n == k` or `k=1`. In scenarios where `n == k`, the inferred metric is `path@k`; in scenarios where `k=1` (i.e., `pass@1` in the DeepSeek formula), it is essentially `avg@n`. Configuring `n` alone is sufficient, so the 20251219 version of the AISBench evaluation tool does not yet support configuring `k` independently.

After the precision evaluation phase, the results will be recorded in the logs and printed to the running window, following the format in the example below (data is for reference only):

```bash
| dataset   | version   | metric                    | mode | vllm-api-stream-chat |
| --------- | --------- | ------------------------- | ---- | -------------------- |
| GPQA_diamond | 604a78    | accuracy (4 runs average) | gen  | 18.00                |
| GPQA_diamond | 604a78    | avg@4                     | gen  | 18.00                |
| GPQA_diamond | 604a78    | pass@4                    | gen  | 53.33                |
| GPQA_diamond | 604a78    | cons@4                    | gen  | 13.33                |
```
Among them, `avg@4` has the same meaning as `pass@1` (average over 4 runs) in DeepSeek.


> ⚠️ While `n` only affects the fluctuation range of the evaluation results and not the mathematical expectation, a larger `n` means more repeated runs for each test case, leading to higher resource consumption. When reproducing accuracy, adjustments should be made based on the actual resource availability.

> 💡 If a paper does not specify the accuracy evaluation metric for a dataset, `pass@1` is generally used by default. Thus, omitting the configuration of `n` and `k` in the AISBench dataset configuration file defaults to `pass@1`.

---

## References
- DeepSeek R1 Hugging Face Model Card: https://huggingface.co/deepseek - ai/DeepSeek - R1
- DeepSeek R1 Paper: https://github.com/deepseek - ai/DeepSeek - R1/blob/main/DeepSeek_R1.pdf