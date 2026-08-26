# 复现大语言模型（LLM）论文（技术报告）中的数据集测评结果（以DeepSeek R1使用的GPQA数据集为例）

> 💡 本文档中的测评命令均可以通过 [自定义配置文件方式](../advanced_tutorials/run_custom_config.md) 实现，将模型、数据集、summarizer 等配置写入一个 Python 文件，一次编写、多次复用。配置文件本质上是 Python 脚本，支持循环、条件判断、列表推导等所有 Python 语法。详见 [自定义配置文件运行AISBench](../advanced_tutorials/run_custom_config.md)。

## 前言-方法论
如果想要通过AISBench测评工具复现论文精度，需要对齐模型的技术报告或论文中对此数据集的测试方法，在评测工具这边需要对齐的如下：
**模型相关配置**：
- 选取合适的endpoint对应的模型任务
- 最大输出长度完全对齐
- 后处理参数完全对齐

**数据集相关配置**
- 提示词工程完全对齐
- 答案提取方式完全对齐
- 精度评估指标对齐

## 以复现DeepSeek R1模型在GPQA数据集上的测评结果为例
### 选取合适的endpoint对应的模型配置文件
为了确保执行效率，复现模型精度时一般也使用推理服务作为被测对象。访问推理服务可以有不同的endpoint，业界主流还是用OpenAI的endpoint。OpenAI的endpoint主要有两种：`v1/completions`和`v1/chat/completions`。
- **v1/completions**：
    模型按 “续写前缀” 逻辑生成文本，不主动区分 “指令” 与 “内容”，需通过 prompt 工程强引导（如加 “请回答：”），否则可能出现模仿式输出而非指令执行。例如输入 “将下列英文翻译成中文：Hello”，可能续写为 “将下列中文翻译成英文：你好”，而非直接翻译。
    因此适合单轮文本生成（如代码补全、短文写作、文本续写、简单文本分类），或需兼容旧版基础模型的场景。
- **v1/chat/completions**：
    模型原生理解 system/user/assistant 角色语义，优先执行用户指令，对话一致性与意图对齐更稳定，无需复杂 prompt 包装即可完成翻译、总结等任务。
    因此适合多轮对话（客服、聊天机器人）、指令驱动任务（翻译、摘要、数据分析）、工具集成（函数调用、检索增强）、多模态交互等现代 LLM 应用场景。

💡 目前（2025年1月之后），几乎所有新发布的LLM模型都支持`v1/chat/completions` endpoint，而且`v1/completions` endpoint也基本被废弃，因此模型配置文件中一般只选用访问`v1/chat/completions` endpoint的模型任务**vllm_api_general_chat**(用非流式接口访问服务)和**vllm_api_stream_chat**(用流式接口访问服务)。

以模型任务`vllm_api_general_chat`为例，其对应的模型配置文件的绝对路径可以通过执行如下命令得到：
```bash
ais_bench --models vllm_api_general_chat --search
```

⚠️ 后续模型相关配置都在此配置文件中修改。


### 最大输出长度完全对齐
在[DeepSeek R1 Huggingface模型卡片](https://huggingface.co/deepseek-ai/DeepSeek-R1)中可以看到如下描述
> ## 4. Evaluation Results
> ### DeepSeek-R1-Evaluation
> For all our models, the maximum generation length is set to 32,768 tokens....

从这段描述可以看出，DeepSeek R1模型的最大输出长度被设置为32,768 tokens。

因此以模型任务`vllm_api_general_chat`为例，最大输出长度的配置如下：
```python
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        # ......
        max_out_len=32768,          # 推理服务输出的token的最大数量
        # ......
    )
]
```

### 后处理参数完全对齐
在[DeepSeek R1 Huggingface模型卡片](https://huggingface.co/deepseek-ai/DeepSeek-R1)中可以看到如下描述
> ## 4. Evaluation Results
> ### DeepSeek-R1-Evaluation
> ..., For benchmarks requiring sampling, we use a temperature of $0.6$, a top-p value of $0.95$, ...

从这段描述可以看出，DeepSeek R1模型的后处理参数包括温度（temperature）为0.6，top-p值（top_p）为0.95。

因此以模型任务`vllm_api_general_chat`为例，后处理参数的配置如下：
```python
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        # ......
        generation_kwargs=dict( # 后处理参数都填在这里
            temperature=0.6,
            top_p=0.95,
        ),
        # ......
    )
]
```

### 提示词工程完全对齐
提示词工程对模型精度的影响是很显著的，一般模型论文或技术报告都会公开测试所使用的提示词，如果是使用第三方工具测试的，也会强调具体是什么开源工具。
在DeepSeek R1论文中，提及了GPQA数据集的测试提示词工程，如下：

> Evaluation Prompts Following the setup in DeepSeek-V3, standard benchmarks such as MMLU, DROP, GPQADiamond, and SimpleQA are evaluated using prompts from the simple-evals framework.

这说明DeepSeek R1的提示词工程是使用simple-evals这个工具的提示词模板，可以参考[simple-evals](https://github.com/openai/simple-evals)这个项目，其中GPQA用到的提示词相关的部分为(需求去代码里看)：
```python
QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()
```

因此AISBench的提示词工程应当修改为:
```python
# https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/gpqa/gpqa_gen_0_shot_cot_chat_prompt.py

## 与simple-evals完全相同的提示词模板
align_prompt = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

## ......

gpqa_infer_cfg = dict(
    prompt_template=dict( # 提示词工程
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=align_prompt), # 传入提示词模板
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))
```
更详细的提示词工程的介绍请参考[Prompt 模板介绍](../prompt/prompt_template.md)

### 答案提取方式完全对齐
如何从模型的推理结果中提取答案并评估会直接影响评测的得分。llm测评使用的数据集的答案提取总体来说分为3类：
1. 对于类似选择题或问答题类型的评测数据集（ceval、gsm8k等），答案提取方式一般是基于固定的正则表达式。
2. 对于比较复杂的诸如代码类或者要包含解题过程的数学类的数据集（例如livecodebench、humaneval、math500等），一般会有统一的数据集配套的evaluate库供调用。
3. 对于一些发散或者偏主观的数据集可能需要引入裁判模型进行评估。

一般情况下只有测评涉及到第3类数据集时，在模型的论文或技术报告里才会明确说明（例如使用GPT-4-1106的api作为裁判模型）。对于第2类数据集一般默认使用数据集配套的evaluate库进行答案提取。

对于第1类数据集，如果模型论文或技术报告说明了提示词工程出自什么工具，那么可以直接沿用这个工具中提及的答案提取方式，还是以GPQA为例，在simple-evals中，GPQA的答案提取方式就是基于固定的正则表达式:
```python
# https://github.com/openai/simple-evals/blob/main/common.py
ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"

# https://github.com/openai/simple-evals/blob/main/gpqa_eval.py
match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
```
因此AISBench的答案提取方式应当修改为与simple-evals完全相同的正则表达式：
```python
# https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/datasets/gpqa.py
@TEXT_POSTPROCESSORS.register_module() # 答案提取函数，从模型原始回答的字符串中提取A,B,C,D中一个选项
def GPQA_Simple_Eval_postprocess(text: str) -> str:
    """
    从模型原始回答的字符串中提取A,B,C,D中一个选项作为答案。

    :param text: 模型原始回答的字符串。
    :return: 提取到的答案选项（A、B、C、D），如果未找到匹配项则返回None。
    """
    ANSWER_PATTERN = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"
    match = re.search(ANSWER_PATTERN, text)
    if match:
        return match.group(1)
    return None


# https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/gpqa/gpqa_gen_0_shot_cot_chat_prompt.py

from ais_bench.benchmark.datasets import GPQADataset, GPQA_Simple_Eval_postprocess, GPQAEvaluator

gpqa_eval_cfg = dict(evaluator=dict(type=GPQAEvaluator),
                     pred_postprocessor=dict(type=GPQA_Simple_Eval_postprocess)) # 传入自定义的答案提取函数，函数本身也可以直接定义在数据集配置文件里
```

### 精度评估指标对齐
一般模型的评测结果都会有这样一张表，以DeepSeek的为例：

|Model|	AIME 2024 pass@1|	AIME 2024 cons@64|	MATH-500 pass@1|	GPQA Diamond pass@1|	LiveCodeBench pass@1|CodeForces rating|
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
|GPT-4o-0513|	9.3|	13.4|	74.6|	49.9|	32.9|	759|
|Claude-3.5-Sonnet-1022|	16.0|	26.7|	78.3|	65.0|	38.9|	717|
|o1-mini|	63.6|	80.0|	90.0|	60.0|	53.8|	1820|

其中的`cons@64`，`pass@1`就是表示精度评估指标。这些精度评估指标的详细介绍可以参考[精度评估指标说明](../base_tutorials/results_intro/accuracy_metric.md#二passk-consk-avgn-的定义与关系)

以GPQA为例，表格是指明其是使用`pass@1`作为精度评估指标的，在DeepSeek R1论文中对pass@1的描述如下：

> ... , and report pass@1 using a non-zero temperature. Specifically, we use a sampling temperature of 0.6 and a top-𝑝 value of 0.95 to generate 𝑘 responses (typically between 4 and 64, depending on the test set size) for each question. Pass@1 is then calculated as
>  ${\text{pass@1}} = \frac{1}{n} \sum_{i=1}^{n} p_i$

那么在AISBench中，模型配置文件做如下配置：
```python
# https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py

models = [
    dict(
        ... # 其它参数
        generation_kwargs = dict(
            num_return_sequences = 4, # n=4~64
            ... # 其它参数
        ),
        ...
    )
]

```
一般情况下 `n == k` 或者 `k=1`，`n == k`的场景推理出的指标为`path@k`，`k=1`的场景也就是deepseek公式中的`pass@1`本质上是`avg@n`，单独配置`n`够用，因此AISBench评测工具20251219版本下还不支持单独配置`k`。

精度评估阶段结束后，结果会记录在日志和打屏在运行窗口，格式按照以下示例内容（数据仅供参考）：

```bash
| dataset   | version   | metric                    | mode | vllm-api-stream-chat |
| --------- | --------- | ------------------------- | ---- | -------------------- |
| GPQA_diamond | 604a78    | accuracy (4 runs average) | gen  | 18.00                |
| GPQA_diamond | 604a78    | avg@4                     | gen  | 18.00                |
| GPQA_diamond | 604a78    | pass@4                    | gen  | 53.33                |
| GPQA_diamond | 604a78    | cons@4                    | gen  | 13.33                |
```
其中`avg@4`和deepseek中执行4次平均的`pass@1`含义相同。

> ⚠️ `n`本身只是影响评测结果的波动幅度不影响数学期望，但是`n`越大意味这需要反复跑同一条用例的次数越多，资源消耗越大。如果要复现精度需要依据实际资源情况做调整。

> 💡如果论文的数据集不强调是什么精度评估指标，一般默认`pass@1`，因此在AISBench的数据集配置文件中不配置`n`和`k`就是默认`pass@1`

## 参考资料
- DeepSeek R1 huggingface模型卡片：https://huggingface.co/deepseek-ai/DeepSeek-R1
- DeepSeek R1 论文：https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf