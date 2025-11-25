# 错误码说明
## TMAN-CMD-001
### 错误描述
该错误表明在执行命令时，缺少必填的输入参数。
通过命令行启动ais_bench评测工具时，必须指定模型配置和数据集配置。
合法场景示例：
```bash
# 使用开源数据集，必须通过`--models`指定模型任务，通过`--datasets`指定数据集任务
ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen
# 使用自定义数据集，必须通过`--models`指定模型任务，通过`--custom_dataset_path`指定自定义数据集路径
ais_bench --models vllm_api_stream_chat --custom_dataset_path /path/to/custom/dataset
```
### 解决办法
参考合法场景示例补齐缺失参数。

## TMAN-CMD-002
### 错误描述
该报错表明命令行参数的取值不在合法范围内
### 解决办法
在本文档中搜索日志中出现的具体命令行，找到命令行说明中对参数取值的约束。<br>
例如执行`ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen --num-prompts -1 --mode perf` 出现本报错，在文档中检索`--num-prompts`，找到参数说明中的约束
| 参数| 说明| 示例 |
| ---- | ---- | ---- |
| `--num-prompts` | 指定数据集测评条数，需传入正整数，超过数据集条数或默认情况下表示对全量数据集进行测评。 | `--num-prompts 500` |

参数说明中约束为正整数，需大于0。

## TMAN-CFG-001
### 错误描述
.py配置文件中的内容存在语法错误，导致解析失败。
### 解决办法
检查日志中打印配置文件中存在的python语法错误（ais_bench评测工具可修改的配置文件均遵循python语法），例如缺少引号、括号不匹配等，并修正。

## TMAN-CFG-002
### 错误描述
.py配置文件中缺少必要的参数，导致解析失败。
例如，具体报错日志为：`Config file /path/to/vllm_api_stream_chat.py does not contain 'models' param!`，这表明配置文件中缺少`models`参数。
合法的`vllm_api_stream_chat.py`内容中包含`models`参数：
```python
# ......
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-stream-chat",
        # ......
    )
]

```
### 解决办法
在报错日志中打印的.py配置文件中，补齐日志中提示缺失的参数.

## TMAN-CFG-003
### 错误描述
.py配置文件中存在的参数类型错误，导致解析失败。
例如`vllm_api_stream_chat.py`配置文件中，相关配置为：
```python
# ......
models = dict(
    attr="service",
    type=VLLMCustomAPIChat,
    abbr="vllm-api-stream-chat",
    # ......
)
```
具体报错日志为：`In config file /path/to/vllm_api_stream_chat.py, 'models' param must be a list!`，这表明配置文件中`models`参数的类型错误，应是列表类型（实际是字典类型）。
### 解决办法
在报错日志中打印的.py配置文件中，依据日志中提示将错误的参数类型更正为要求的参数类型。

## UTILS-MATCH-001
### 错误描述
通过`--models`、`--datasets`或`--summarizer`指定的任务名称，无法匹配到与任务名称同名的.py配置文件。
### 解决办法
检查日志提示的无法匹配的任务名称，例如`xxxx`无法匹配会打印如下日志：
```
+------------------------+
| Not matched patterns   |
|------------------------|
| xxxx                   |
+------------------------+
```
#### 场景 1：未指定配置文件所在文件夹路径
先执行`pip3 show ais_bench_benchmark | grep "Location:"`，查看ais_bench评测工具安装路径，例如执行后得到如下信息：
```bash
Location: /usr/local/lib/python3.10/dist-packages
```
那么配置文件所在路径为`/usr/local/lib/python3.10/dist-packages/ais_bench/benchmark/configs`，进入该路径
1. 如果无法匹配的任务名称通过`--models`指定，那么检查`models/`路径内（含多级目录）是否存在与任务名称同名的.py配置文件。
2. 如果无法匹配的任务名称通过`--datasets`指定，那么检查`datasets/`路径内（含多级目录）是否存在与任务名称同名的.py配置文件。
3. 如果无法匹配的任务名称通过`--summarizer`指定，那么检查`summarizers/`路径内（含多级目录）是否存在与任务名称同名的.py配置文件。
#### 场景 2：指定了配置文件所在文件夹路径
如果在执行命令时，通过`--config-dir`指定了配置文件所在文件夹路径，那么进入该路径
1. 如果无法匹配的任务名称通过`--models`指定，那么检查`models/`路径内（含多级目录）是否存在与任务名称同名的.py配置文件。
2. 如果无法匹配的任务名称通过`--datasets`指定，那么检查`datasets/`路径内（含多级目录）是否存在与任务名称同名的.py配置文件。
3. 如果无法匹配的任务名称通过`--summarizer`指定，那么检查`summarizers/`路径内（含多级目录）是否存在与任务名称同名的.py配置文件。

## UTILS-CFG-001
### 错误描述
使用[随机合成数据集](../advanced_tutorials/synthetic_dataset.md)`tokenid`场景下，模型配置文件必须指定tokenizer路径。
### 解决办法
假设ais_bench评测工具命令为`ais_bench --models vllm_api_stream_chat --datasets synthetic_gen_tokenid --mode perf`，那么`vllm_api_stream_chat.py`（配置文件路径检索方式参考[任务对应配置文件修改](../get_started/quick_start.md#任务对应配置文件修改)）配置文件中`models`中所有的`path`参数须传入tokenizer路径（一般就是模型权重文件夹路径）。
```python
# ......
models = dict(
    attr="service",
    type=VLLMCustomAPIChat,
    abbr="vllm-api-stream-chat",
    path="/path/to/tokenzier", # 传入tokenizer路径
    # ......
)
```

## UTILS-CFG-002
### 错误描述
通过模型配置文件内参数初始化模型实例时，因为参数内容非法而失败。
### 解决办法
检查日志中`build failed with the following errors:{error_content}`，依据`error_content`的提示修正模型配置文件内参数。
例如模型配置文件中`batch_size`参数值为100001，`error_content`为`"batch_size must be an integer in the range (0, 100000]`，表面batch_size参数超出了合法范围（0, 100000]，那么需要将`batch_size`参数值修正为100000。

## UTILS-CFG-003
### 错误描述
模型配置文件内参数取值在工具限定范围内
### 解决办法
依据详细日志提示配置工具限定范围内的参数取值，例如配置文件内容为：
```python
# vllm_stream_api_chat.py中
models = [
    {
        attr="service1",
        # ......
    }
]
```
详细报错日志为：
```bash
Model config contain illegal attr, 'attr' in model config is 'service1', only 'local' and 'service' are supported!
```
这表明模型参数`attr`取值为`'service1'`，而工具限定范围内只支持`'local'`和`'service'`两种取值，需要将`attr`设置为合法的取值之一。

## UTILS-CFG-004
### 错误描述
模型参数的部分配置项在每个模型的配置中必须一致，不能出现不同的取值。
### 解决办法
依据详细日志的提示统一配置的取值，例如配置文件内容为;
```python
# vllm_stream_api_chat.py中
models = [
    {
        attr="service",
        # ......
    },
    {
        attr="local"
    }
]
```
详细报错日志为：
```bash
Cannot run local and service model together! Please check 'attr' parameter of models
```
因为`models`配置中包含了`'service'`和`'local'`两种参数取值，而工具只支持统一配置一种，因此需要将`models`配置中`attr`参数设置为`'service'`或`'local'`中的一种。

## PARTI-FILE-001
### 错误描述
输出路径文件的权限不足，工具无法将结果写入。
### 解决办法
例如报错日志为:
```bash
Current user can't modify /path/to/workspace/outputs/default/20250628_151326/predictions/vllm-api-stream-chat/gsm8k.json, reuse will not enable.
```
执行`ls -l /path/to/workspace/outputs/default/20250628_151326/predictions/vllm-api-stream-chat/gsm8k.json`查看此路径属主和权限，发现该文件当前用户不可写，需要给该文件添加当前用户的写权限（例如执行`chmod u+w /path/to/workspace/outputs/default/20250628_151326/predictions/vllm-api-stream-chat/gsm8k.json`即可添加当前用户的写权限）。

## CALC-MTRC-001
### 错误描述
性能结果数据无效，无法计算指标。
### 解决办法
#### 场景 1：性能结果原始数据为空
如果在执行命令时，通过`--mode perf_viz`指定了性能结果重计算，若基础输出路径为`outputs/default/20250628_151326`（查找打屏中`Current exp folder: `），那么检查该路径下`performances/`文件夹内的`*_details.jsonl`文件内容是否都为空，若为空，则需要先执行一次评测，生成性能结果数据。
#### 场景 2：性能结果原始数据不包含任何有效值
如果在执行命令时，通过`--mode perf_viz`指定了性能结果重计算，若基础输出路径为`outputs/default/20250628_151326`（查找打屏中`Current exp folder: `），那么检查该路径下`performances/`文件夹内的`*_details.jsonl`文件内容不包含任何有效字段（可能被篡改了），则需要重新执行性能测评，生成新的数据。

## CALC-FILE-001
### 错误描述
落盘性能结果数据失败
### 解决办法
若详细的报错日志为；
```bash
Failed to write request level performance metrics to csv file '{/path/to/workspace/outputs/default/20250628_151326/performances/vllm-api-stream-chat/gsm8k.csv': XXXXXX
```
其中`XXXXXX`为具体落盘失败的原因，例如`Permission denied`表示该文件已存在且当前用户没有写权限，可以选择删除该文件或者给已存在的文件添加当前用户的写权限。

## CALC-DATA-001
### 错误描述
所有结束的推理请求都没有获取到有效的性能指标数据，无法计算指标。
### 解决办法
若具体日志为：
```bash
All requests failed, cannot calculate performance results. Please check the error logs from responses!
```
这表明推理过程中的所有请求都失败了，需要进一步去查看请求失败的日志，定位请求失败的原因。
1. 如果命令中包含`--debug`，请求失败的日志将直接打屏，可以在打屏记录中查看
2. 如果命令中不包含`--debug`，打屏记录中会有`[ERROR] [RUNNER-TASK-001]task failed. OpenICLApiInfervllm-api-stream-chat/synthetic failed with code 1, see outputs/default/20251125_160128/logs/infer/vllm-api-stream-chat/synthetic.out`类似的日志，可以在`outputs/default/20251125_160128/logs/infer/vllm-api-stream-chat/synthetic.out`中查看具体请求失败的原因。

## CALC-DATA-002
### 错误描述
计算稳态性能指标时，在所有请求信息中找不到属于稳定阶段的请求，无法计算稳态指标。
### 解决办法
可以检查一下推理请求的并发图（参考文档：https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/base_tutorials/results_intro/performance_visualization.html），确认并发阶梯图中`Request Concurrency Count`是否达到模型配置文件中设置的并发数（`batch_size`参数）**且至少存在两个请求达到最大并发数**。
若未满足上述条件，可以尝试以下方式达到稳定状态：
#### 并发阶梯图中`Request Concurrency Count`持续增长之后直接持续下降
1. 降低推理请求的并发数（模型配置文件中的`batch_size`参数）。
2. 增加推理的总请求数。
#### 并发阶梯图中`Request Concurrency Count`持续增长之后波动一段时间后持续下降
1. 降低推理请求的并发数（模型配置文件中的`batch_size`参数）。
2. 提高发送推理请求的频率（模型配置文件中的`request_tate`参数）

## SUMM-TYPE-001
### 错误描述
所有数据集任务的`abbr`参数配置存在混用的情况
### 解决办法
例如报错日志为：
```bash
mixed dataset_abbr type is not supported, dataset_abbr type only support (list, tuple) or str.
```
这表明在`datasets`配置中，所有数据集任务的`abbr`参数配置为不同的类型（例如`list`和`str`），需要将所有数据集任务的`abbr`参数配置统一为一个类型的值（例如`list`或`str`）。

## SUMM-FILE-001
### 错误描述
在输出的工作路径下没有任何性能数据文件（`*_details.jsonl`）
### 解决办法
1. 确认是否在执行评测时，通过`--mode perf_viz`误指定了性能结果重计算，如果是希望完整地跑一遍性能测试，请指定`--mode perf`
2. 确认基础输出路径是否正确，例如`outputs/default/20250628_151326`（查找打屏中`Current exp folder: `）。
3. 确认该路径下`performances/`文件夹内是否存在`*_details.jsonl`文件，若不存在，请排查之前的打屏日志中的其他报错信息，确认是否有其他错误导致性能数据文件未生成，依据其他错误日志的指引进一步定位。

## SUMM-MTRC-001
### 错误描述
### 解决办法

## RUNNER-TASK-001
### 错误描述
### 解决办法

## TASK-PARAM-001
### 错误描述
### 解决办法

## TINFER-PARAM-001
### 错误描述
### 解决办法

## TINFER-PARAM-002
### 错误描述
### 解决办法

## TINFER-PARAM-003
### 错误描述
### 解决办法

## TINFER-PARAM-004
### 错误描述
### 解决办法

## TINFER-PARAM-005
### 错误描述
### 解决办法

## TINFER-IMPL-001
### 错误描述
### 解决办法

## TEVAL-PARAM-001
### 错误描述
### 解决办法

## TEVAL-PARAM-002
### 错误描述
### 解决办法

## ICLI-PARAM-001
### 错误描述
### 解决办法

## ICLI-PARAM-002
### 错误描述
### 解决办法

## ICLI-PARAM-003
### 错误描述
### 解决办法

## ICLI-PARAM-004
### 错误描述
### 解决办法

## ICLI-PARAM-005
### 错误描述
### 解决办法

## ICLI-RUNTIME-001
### 错误描述
### 解决办法

## ICLI-RUNTIME-002
### 错误描述
### 解决办法

## ICLI-RUNTIME-003
### 错误描述
### 解决办法

## ICLI-IMPL-001
### 错误描述
### 解决办法

## ICLI-IMPL-002
### 错误描述
### 解决办法

## ICLI-IMPL-003
### 错误描述
### 解决办法

## ICLI-FILE-001
### 错误描述
### 解决办法

## ICLI-FILE-002
### 错误描述
### 解决办法

## ICLE-DATA-001
### 错误描述
### 解决办法

## ICLE-DATA-002
### 错误描述
### 解决办法

## ICLE-IMPL-001
### 错误描述
### 解决办法

## ICLR-TYPE-001
### 错误描述
### 解决办法

## ICLR-TYPE-002
### 错误描述
### 解决办法

## ICLR-PARAM-001
### 错误描述
### 解决办法

## ICLR-PARAM-002
### 错误描述
### 解决办法

## ICLR-PARAM-003
### 错误描述
### 解决办法

## ICLR-PARAM-004
### 错误描述
### 解决办法

## ICLR-IMPL-001
### 错误描述
### 解决办法

## ICLR-IMPL-002
### 错误描述
### 解决办法

## ICLR-IMPL-003
### 错误描述
### 解决办法

## MODEL-IMPL-001
### 错误描述
### 解决办法

## MODEL-IMPL-002
### 错误描述
### 解决办法

## MODEL-PARAM-001
### 错误描述
### 解决办法

## MODEL-PARAM-002
### 错误描述
### 解决办法

## MODEL-PARAM-003
### 错误描述
### 解决办法

## MODEL-PARAM-004
### 错误描述
### 解决办法

## MODEL-PARAM-005
### 错误描述
### 解决办法

## MODEL-TYPE-001
### 错误描述
### 解决办法

## MODEL-TYPE-002
### 错误描述
### 解决办法

## MODEL-TYPE-003
### 错误描述
### 解决办法

## MODEL-TYPE-004
### 错误描述
### 解决办法

## MODEL-DATA-001
### 错误描述
### 解决办法

## MODEL-DATA-002
### 错误描述
### 解决办法

## MODEL-DATA-003
### 错误描述
### 解决办法

## MODEL-CFG-001
### 错误描述
### 解决办法

## MODEL-MOD-001
### 错误描述
### 解决办法