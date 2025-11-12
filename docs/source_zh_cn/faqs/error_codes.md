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
### 解决办法

## UTILS-CFG-003
### 错误描述
### 解决办法

## UTILS-CFG-004
### 错误描述
### 解决办法

## UTILS-CFG-005
### 错误描述
### 解决办法

## UTILS-CFG-006
### 错误描述
### 解决办法

## UTILS-CFG-007
### 错误描述
### 解决办法

## PARTI-FILE-001
### 错误描述
### 解决办法

## CALC-MTRC-001
### 错误描述
### 解决办法

## CALC-FILE-001
### 错误描述
### 解决办法

## CALC-DATA-001
### 错误描述
### 解决办法

## CALC-DATA-002
### 错误描述
### 解决办法

## SUMM-TYPE-001
### 错误描述
### 解决办法

## SUMM-FILE-001
### 错误描述
### 解决办法

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