# AISBench FAQ 常见问题解答

## 1. 常见安装错误

### 1.1 ‘torch.library’ has no attribute ‘register\_fake’
![输入图片说明](https://foruda.gitee.com/images/1752115913457665110/c75f79c4_15797231.png "屏幕截图")

**问题根因：**

aisbench环境torch和torchvision版本不匹配

**建议的处理方式如下：**
参考[Previous PyTorch Versions](https://pytorch.org/get-started/previous-versions/)安装匹配版本的torch和torchvision

### 1.2 ImportError: Failed to import required modules

![输入图片说明](https://foruda.gitee.com/images/1752115955182360975/cec5312d_15797231.png "屏幕截图")

**问题根因：**

未安装依赖包，可能会导致部分数学计算或多模态精度计算功能无法使用

**建议的处理方式如下：**
执行如下命令，安装aisbench的拓展依赖包
```shell
pip3 install -r requirements/api.txt
pip3 install -r requirements/extra.txt
```
---

## 2. 参数配置问题

### 2.1 已经知道测评命令，我需要修改哪些配置文件


**问题根因：**

AisBench提供了快捷配置文件查询功能。

**建议的处理方式如下：**
只需要在原有测评命令后添加`--search`参数，终端会打印该任务涉及的配置文件路径，直接修改即可
![输入图片说明](https://foruda.gitee.com/images/1752204051848035094/9987953a_15797231.png "屏幕截图")

### 2.2 配置文件中应该修改哪些参数


**问题根因：**

Aisbench提供了各个数据集的默认配置文件，不建议直接修改，如果需要定制化template或指定数据集路径，建议先对原有配置文件进行备份，再修改数据集配置中的template字段和path字段：
![输入图片说明](https://foruda.gitee.com/images/1752204060095585759/8e45aa17_15797231.png "屏幕截图")
模型配置参数必须修改的内容如下：
![输入图片说明](https://foruda.gitee.com/images/1752204068024744757/2b26c904_15797231.png "屏幕截图")

**建议的处理方式如下：**
如果需要对请求速率、并发速度以及模型最大输出长度等参数进行配置，可参考：
[模型配置说明](../base_tutorials/all_params/models.md)

### 2.3 报错：Please pass the argument ‘trust\_remote\_code=True’ to allow custom code to run

![输入图片说明](https://foruda.gitee.com/images/1752204075220061499/7c5c2c79_15797231.png "屏幕截图")

**问题根因：**

新模型或社区模型尚未集成到官方 Transformers 库，如果模型 repo 中包含自定义架构或 tokenizer 的 Python 脚本，就必须设置 trust_remote_code=True，否则加载将失败

**建议的处理方式如下：**
在模型配置文件中配置'trust_remote_code=True告诉 Transformers：可以安全地执行下载下来的 .py 文件。

![输入图片说明](https://foruda.gitee.com/images/1752204081509799033/3e2b778c_15797231.png "屏幕截图")

### 2.4 报错：FileExistsError: Dataset path: \**** is not exist!
![输入图片说明](https://foruda.gitee.com/images/1752204088304680700/6777b51c_15797231.png "屏幕截图")


**问题根因：**

数据集配置错误，aisbench无法找到数据集路径。

**建议的处理方式如下：**
1. 参考[数据集准备指南](../base_tutorials/all_params/datasets.md)完成数据集的准备
2. 开源数据集建议放置到`{工具根路径}/ais_bench/datasets`。
3. 自定义数据集路径需要修改`--datasets`指定的配置文件，修改其中的`path`字段为实际数据集路径（请确保实际数据集路径下数据格式与要求一致）
![输入图片说明](https://foruda.gitee.com/images/1752204096428670468/faa01659_15797231.png "屏幕截图")

---

## 3. 服务化返回报错

### 3.1 Request failed: HTTP status 500. Server response：\*\*timeout**


**问题根因：**

服务端推理超时

**建议的处理方式如下：**

调高超时时间
通过模型配置中的max_out_len减小模型的最大输出长度，降低推理耗时
通过模型配置中的batch_size减小模型的最大并发数，降低服务端并行资源占用，提高请求推理效率
提高硬件配置


### 3.2 Exceeded maximum retry attempts (2) 或 Connection refused


**问题根因：**

client无法与server端建立链接失败，大概率为服务化端口配置或url配置问题。


**建议的处理方式如下：**

1. 通过命令 curl http:{url}:{port}/v1/models 校验服务端是否可以正常访问，示例如下，其中'data'字典中'id'字段即为服务端已加载的模型名称，表明服务化后端可以正常响应
```json
{
  "object": "list",
  "data": [
    {
      "id": "model-id-0",
      "object": "model",
      "created": 1686935002,
      "owned_by": "organization-owner"
    },
    {
      "id": "model-id-1",
      "object": "model",
      "created": 1686935002,
      "owned_by": "organization-owner",
    },
    {
      "id": "model-id-2",
      "object": "model",
      "created": 1686935002,
      "owned_by": "openai"
    }
  ],
  "object": "list"
}
```
2. 检查服务端是否启动并监听；
3. 查看防火墙或安全组配置。

### 3.3 HTTP error during stream response processing: Response ended prematurely


**问题根因：**

流式响应被意外中断


**建议的处理方式如下：**

检查服务器日志定位中断原因；


### 3.4 [AisBenchClientException] Error processing stream response: [StreamResponseError] Expecting value: line 1 column 65533 (char 65532)!


**问题根因：**

流式响应的内容超过默认chunk的缓存大小，导致解析失败。通常出现于非标endpoint包含了额外的服务端返回数据。


**建议的处理方式如下：**

增大 `ais_bench/benchmark/global_consts.py`中的`MAX_CHUNK_SIZE`,增大chunk的缓存空间

---

## 4. 精度测评常见问题

### 4.1 如何选择模型配置进行精度测评


**问题根因：**

对不同模型配置的特性和适用场景不清楚


**建议的处理方式如下：**

1. 阅读服务化框架官方文档确认其支持的服务化后端，例如vllm、sglang均只openai的endpoint，则只能选择`v1/chat/completions`和`v1/completions`对应的模型配置进行测评；
2. 参考[模型配置说明](../base_tutorials/all_params/models.md)选择满足服务化后端支持的模型配置；
3. 对于chat类型的api，会根据模型的配置，在原有的prompt中添加包含system角色的chat_template的结构化信息，例如
```json
[
  {
    "role": "system",
    "content": "You are a helpful assistant."
  },
  {
    "role": "user",
    "content": "{prompt}"
  }
]
```

而其他模型配置则会直接将prompt作为输入内容。因此精度结果可能会存在差异，测试时需要根据实际需求选择合适的模型配置。

### 4.2 精度测评如何查看模型的输出内容


**问题根因：**


aisbench会保存模型输出的精度结果，但不会进行直接打印。


**建议的处理方式如下：**

推理过程中会在工作目录的`predictions`子目录下缓存模型的推理结果，例如：`outputs/default/20250710_164659/predictions/vllm-api-stream-chat/gsm8k.json`;
数据格式如下：
```json
{
    "0": { # 数据id
        "origin_prompt": "What is 2 + 2?", # 数据的原始输入
        "prediction": "4", # 模型的输出
        "gold": "4" # 数据的正确答案
    },
    ...
}
```


### 4.3 如何查看每个问题的正确错误结果


**问题根因：**

aisbench对精度计算过程中默认不保存每个case的正确错误信息，但提供了参数开启 `dump-eval-details` 以便记录详细的评测结果。

**建议的处理方式如下：**

1. 在评测命令中添加`--dump-eval-details`参数；
2. 如果已有模型推理结果保存在工作目录下，可配合`--reuse`参数重新生成包含详细评测结果的文件，详情参考：[eval模式](../base_tutorials/all_params/mode.md#eval模式)；
3. 推理过程中会在工作目录的`results`子目录下缓存模型的推理结果，例如：`outputs/default/20250710_164659/results/vllm-api-stream-chat/gsm8k.json`;
数据格式如下：
```json
"details": {
    "0": { # 数据id
        "prompt": "What is 2 + 2?", # 数据的原始输入
        "origin_prediction": "The answer is 4.", # 模型的输出
        "predictions": "4", # 从模型输出中提取的答案
        "references": "4", # 数据的正确答案
        "correct": true # 是否正确
    },
}
```

### 4.4 模型输出中包含正确答案，但精度计算结果异常


**问题根因：**


aisbench基于特定的匹配规则从模型回答中提取正确答案，可能存在case，人工能从模型回答中识别正确答案，但是却不满足aisbench的匹配规则，导致精度计算结果为空。


**建议的处理方式如下：**

参考[如何查看每个问题的正确错误结果](#43-如何查看每个问题的正确错误结果)开启`dump-eval-details`参数，查看模型输出的详细评测结果，确认是否是匹配规则导致的精度计算异常。

### 4.5 CEval 和 MMLU 中的naive_average和weighted_average精度结果的区别


**问题根因：**

ceval和mmlu这类包含多个子类别数据集的精度结果会包含每个子类别的精度结果，除了会计算每个子类别数据集的精度，还包含整体的平均精度结果。整体的平均精度包含naive_average和weighted_average两种计算方式，naive_average是对每个子类别的精度结果进行简单平均，而weighted_average则是根据每个子类别的数据量进行加权平均。


**建议的处理方式如下：**

1. 论文中的精度结果通常是weighted_average的结果；如果需要复现论文中的精度结果，建议参考weighted_average的结果；
2. 如果需要对每个子类别的精度结果进行分析，建议参考naive_average的结果。



### 4.6 未在 DS 上复现论文精度（参照 AISBench Wiki）


**问题根因：**

1. 实际使用的环境配置与 AISBench wiki中的不一致；
2. 数据集经过额外处理与标准的原始数据集存在差异


**建议的处理方式如下：**

1. 对比`--models`中的配置参数与AISBench wiki中一致，包括最大输出长度`max_out_len` (防止截断），后处理参数`generation_kwargs`；
2. 参考aisbench社区[数据集准备指南](../base_tutorials/all_params/datasets.md)完成数据准备；
3. 将差异结果提交 Issue 寻求帮助。

### 4.7 FileNotFoundError: [Errno 2] No such file or directory

**问题根因：**

未按照资料说明配置数据集


**建议的处理方式如下：**

参考aisbench社区[数据集准备指南](../base_tutorials/all_params/datasets.md)完成数据准备；

---

## 5. 性能测评常见问题

### 5.1 如何选择模型配置进行性能测评


**问题根因：**

对不同模型配置的特性和适用场景不清楚

**建议的处理方式如下：**

1. 阅读服务化框架官方文档确认其支持的服务化后端，例如vllm、sglang均只支持openai的endpoint，则只能选择`v1/chat/completions`和`v1/completions`对应的模型配置进行测评；
2. 参考[模型配置说明](../base_tutorials/all_params/models.md)选择满足服务化后端支持的模型配置，需要注意只有流式接口支持性能测评，即模型配置中包含`stream`关键字；
3. 对于chat类型的api，会根据模型的配置，在原有的prompt中添加包含system角色的chat_template的结构化信息，例如
```json
[
  {
    "role": "system",
    "content": "You are a helpful assistant."
  },
  {
    "role": "user",
    "content": "{prompt}"
  }
]
```
而其他模型配置则会直接将prompt作为输入内容。因此实际输入的token长度可能会存在差异，导致性能结果出现差异。

4. `v1/chat/completions`和`v1/completions`对应的模型配置会从服务端的返回信息中提取生成的token数目，与实际服务端生成的token总数一致，而其他模型配置则会根据实际输出回答的string字符串进行encode，转化为tokenid获取输出token长度，与实际服务端生成的token总数可能存在差异。如果需要保证输出token长度的完全准确，建议选择`v1/chat/completions`和`v1/completions`对应的模型配置进行性能测评。

### 5.2 什么是稳态

**问题根因：**

稳定状态性能测试（后续简称"稳态测试"）是为了模拟真实推理服务的业务场景，测试推理服务处于稳定状态下的性能。 稳定状态，是指推理服务在并发请求数量达到最大值时，能够同时处理且保持稳定的状态。

**建议的处理方式如下：**

参考[服务化稳定状态性能测试](../advanced_tutorials/stable_stage.md)

### 5.3 什么是压测

**问题根因：**

AISBench的服务化性能压力测试是为了模拟真实推理服务的业务场景，测试特定时间段内服务化在最大并发负荷下的性能。


**建议的处理方式如下：**

参考[服务化压力测试](../advanced_tutorials/stable_stage.md#压力测试快速入门)

### 5.4 InputTokens 与合成数据集输入不一致


**问题根因：**

1. 使用了chat类型的模型配置（如`v1/chat/completions`），该配置会将输入的prompt与chat_template进行拼接，导致实际输入的token数目与合成数据集的输入不一致；

2. 合成数据集包含两种类别：`string`和`tokenid`,两者构建合成数据集的原理存在差异

`string`：将`A `认为一个token，拼接多个`A `构成输入字符串，最终的输入token数目与实际输入的字符串长度一致。示例如下：
```python
from transformers import AutoTokenizer

# 加载指定的分词器模型（Qwen2-7B-Instruct）
tokenizer_model = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    trust_remote_code=True,
    use_fast=True,
    local_files_only=False
)

token_num = 10  # 设定要生成的 token 数量
text = "A " * token_num  # 构造包含 10 个 "A " 的字符串
text = text.strip()  # 去除末尾的空格

# 将文本编码为 token id 列表
new_token_ids = tokenizer_model.encode(text)

# 将 token id 列表解码回文本
new_text = tokenizer_model.decode(new_token_ids)

print(text == new_text)                      # True
print(f"Token Count: {len(new_token_ids)}")  # len = 10
```


`tokenid`：生成token num 个数的tokenid列表，再根据tokenid列表生成对应的输入字符串。由于tokenid转化为string，再由string转化为tokenid的过程不可逆，可能会导致实际token的增多或减少，与配置的输入token数目不一致。示例如下：
```python
from transformers import AutoTokenizer

# 加载指定的分词器模型（Qwen2-7B-Instruct）
tokenizer_model = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    trust_remote_code=True,
    use_fast=True,
    local_files_only=False
)

original_token_ids = [i for i in range(100)]  # 原始的 token id 列表，长度为 100

# 将 token id 列表解码为文本
text = tokenizer_model.decode(original_token_ids)

# 将解码得到的文本再次编码为 token id 列表
new_token_ids = tokenizer_model.encode(text)

print(f"New token ids nums: {len(new_token_ids)}")  # New token ids nums: 35
```


**建议的处理方式如下：**

1. 确认模型配置是否为chat类型的模型配置，如果是，则需要注意实际输入的token数目与合成数据集的输入不一致；
2. 确认合成数据集的类型，选择合适的模型配置；


### 5.5 OutputTokens 与最大输出长度不一致


**问题根因：**

1. 服务化模型最大上下文存在限制，可能导致实际生成的token数目小于最大输出长度；
2. 模型配置的`generation_kwargs`中没有配置`ignore_eos`参数；
2. 模型配置不支持`ignore_eos`参数，导致模型在生成到`eos_token`时提前结束生成；


**建议的处理方式如下：**

1. 确认服务化模型是否存在最大上下文限制。对于支持Openai endpoint 的服务化后端，可以根据输入长度构造一个请求，参考如下命令发送请求观测返回的内容的`completion_tokens`是否达到`max_tokens`：
```shell
curl https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "your prompt"}],
    "stream": false,
    "max_tokens": 1000,
    "ignore_eos": true
  }'

```
2. 通过`--search`参数查看任务的配置文件，检查配置文件中在`generation_kwargs`中是否正确配置了`ignore_eos`参数；
3. 根据服务化后端的官方资料，确认`ignore_eos`参数的支持情况。

### 5.6 TTFT 和 TPOT 都降低，为什么总吞吐反而下降


**问题根因：**

单次延迟与并发效应相互影响，在硬件资源固定的情况下，实际并发的减小会导致并行任务间的资源竞争减弱，单个请求的时延会降低，但由于并行效率的减小，总吞吐量可能会下降。


**建议的处理方式如下：**

aisbench性能测评结束后会生成测评过程的实时并发可视化图像，参考[性能测试可视化并发图使用说明](../base_tutorials/results_intro/performance_visualization.md),可以分析测评任务的并发效率和请求实验


### 5.7 各项指标含义及公式说明


**问题根因：**

aisbench的测评指标定义与业界一致，常见的指标定义如下图所示：
![输入图片说明](https://foruda.gitee.com/images/1752204138375003502/8c9de2a8_15797231.png "屏幕截图")


**建议的处理方式如下：**

更详细介绍，参考资料：[性能测评结果说明](../base_tutorials/results_intro/performance_metric.md)


### 5.8 ValueError: Tokenizer path '' does not exist

**问题根因：**

模型配置未指定分词器路径，或路径指向的目录不存在。

**建议的处理方式如下：**
1. 使用 `--search` 参数查询任务涉及的配置文件。
2. 在相关配置文件中，将 `path` 字段设置为本地词表文件路径（通常为模型权重所在目录）。
3. 确认路径填写正确，且文件实际存在。
![输入图片说明](https://foruda.gitee.com/images/1752204145754826215/b7fe93f0_15797231.png "屏幕截图")



### 5.9 AISBench 与 vllm_benchmark 性能差异

**问题根因：**
1. 输入未对齐：AISBench 和 vllm benchmark 都支持合成数据集测试。AISBench 使用 `tokenid` 方式生成合成数据集，与 vllm benchmark 的 random 数据集构造方式类似，但由于 tokenid 转为字符串的随机性，输入无法完全对齐。
2. 测试所访问的 endpoint 不一致：vllm 通过 `--backend` 参数指定 endpoint，AISBench 通过模型配置文件指定。两者对应关系如下：

| vllm benchmark `--backend` 参数 | AISBench `--models` 配置文件示例         | 说明                                      |
|:--------------------------------|:----------------------------------------|:------------------------------------------|
| vllm/lmdeploy/openai/scalellm/sglang | vllm_api_general_stream                | `v1/completions`                          |
| openai-chat                      | vllm_api_stream_chat                    | `v1/chat/completions`                     |
| tgi                              | tgi_stream_api_general                  | `generate_stream`                         |
| tensorrt-llm                     | triton_stream_api_general               | v2/models/{model_name}/generate_stream    |

> 具体 models 配置文件名称以实际 aisbench 版本为准，通常位于 `ais_bench/benchmark/configs/models/` 目录下。

**建议的处理方式如下：**
1. 两者均支持自定义数据集测试，建议采用相同的数据集构造方式，确保输入一致。
2. 确认所访问的 endpoint 是否一致，确保测试环境对齐。
3. 性能测试建议关闭后处理，设置 `temperature = 0` 或 `temperature = 0.01`（部分 endpoint 不支持 temperature 为 0），以减少后处理对性能结果的影响。

### 5.10 同样开启 request rate 首 token 时延 aisbench 和 benchmark 差距过大

**问题根因：**
由于benchmark和aisbench请求的抓取机制不同导致的，aisbench在开始发送阶段胡抓取多个请求导致处于等待处理的状态，后面每个阶段的请求都会受影响延迟一段时间

### 5.11 稳态测试的实际并发超过最大并发

## 6. 其他

### 6.1 执行任务时不打印日志和进度条


**问题根因：**

aisbench执行并行任务时，不同进程打印的日志会导致日志错乱，因此，默认情况下会将日志保存在工作目录的日志目录下,例如：`outputs/default/20250710_164010/logs`。

**建议的处理方式如下：**

1. 通过`tail -f`命令查看日志内容，例如：
```shell
tail -f ./outputs/default/20250710_164010/logs/infer/vllm-api-stream-chat/gsm8k.out
```
2. 添加--debug参数，开启调试模式，串行执行任务并直接将日志打印在终端；

### 6.2 task OpenICLInfer [***] fail, see ***.out


**问题根因：**

推理任务执行失败，由于没有开启debug模式，提示查看子任务的日志文件。

**建议的处理方式如下：**

1. 通过`vim`或`cat`命令查看子任务的日志文件，例如：
```shell
vim outputs/default/20250711_104313/logs/infer/vllm-api-stream-chat/gsm8k.out
```

### 6.3 可视化文件无法在浏览器正常打开

当出现可视化文件通过浏览器打开后等待过长时间仍显示白色时，可通过按下键盘F12打开浏览器的开发者模式，检查导航栏的"控制台"项是否出现红色报错信息。

- 报错内容1 - "不支持file协议"
  - 原因1. 被浏览器默认为跨域请求而被CORS安全策略拦截
    - 方案1. 启用本地文件访问：在浏览器的设置中搜寻对应浏览器的"运行访问本地文件"开关，选择启用。
    - 方案2. 修改快捷方式属性：在浏览器快捷方式的“目标”中添加 `--allow-file-access-from-files --user-data-dir="可视化html文件路径" --disable-web-security`，并通过重新点击该快捷方式打开新的窗口，则可允许访问本地文件（根据浏览器的不同，可能需要重启电脑或关闭浏览器的其他窗口后重新打开等方式）。
    - 方案3. 检查扩展和缓存：禁用可能干扰本地文件访问的浏览器扩展，清除浏览器缓存。
    - 方案4. 通过Python简单启动本地服务器，通过`http://localhost`访问文件。

- 报错内容2 - "找不到plotly.min.js"
  - 原因1. 服务器安全设置过于严格，导致无法通过CDN方式加载所需的`plotly.min.js`文件
    - 方案1. 移除网络防火墙相关的设置。
    - 方案2. 通过本地加载方式:
      - 步骤1. 进入 [plotly官网的Download章节](https://www.plotly.com/javascript/getting-started/) 或 [bootcdn.plotly网站(选择大于3.0的版本)](https://www.bootcdn.cn/plotly.js/) 下载到与可视化文件html同一目录下;
      - 步骤2. 通过文本形式打开html文件，找到文件开始的以下部分：
          ```html
          <script src="https://cdn.plot.ly/plotly-3.1.0.min.js" charset="utf-8"></script>
          ```
          ，并修改为：
          ```html
          <script src="文件路径" charset="utf-8"></script>
          ```
          保存后退出；
      - 步骤3：重新通过浏览器打开该文件即可。（若仍出现空白页面，则参考“报错内容1”的方案解析）
  - 原因2. 无网络环境，无法加载所需的`plotly.min.js`文件
    - 方案1. 请参考`原因1`的`方案2`，通过在其他有网络环境中下载该文件，再放置到可视化文件所在的环境中。