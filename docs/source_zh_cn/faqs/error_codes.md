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
    dict(
        attr="service1",
        # ......
    )
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
    dict(
        attr="service",
        # ......
    ),
    dict(
        attr="local"
    )
]
```
详细报错日志为：
```bash
Cannot run local and service model together! Please check 'attr' parameter of models
```
因为`models`配置中包含了`'service'`和`'local'`两种参数取值，而工具只支持统一配置一种，因此需要将`models`配置中`attr`参数设置为`'service'`或`'local'`中的一种。

## UTILS-CFG-008
### 错误描述
加载的多模态数据集内容非法。
### 解决办法
1. 若报错日志为`Invalid dataset: /path/to/non-mm-dataset , please check whether the dataset is a MM-style dataset!`，说明指定的数据集`/path/to/non-mm-dataset`不是一个合法的多模态风格的数据集，合法的数据集中每条数据至少包含`type`或`path`字段，如果包含`type`字段，那么`type`字段取值必须为`["image", "video", "audio"]`之一。
2. 若报错日志为`Param 'mm_type' does not match the data type of dataset: /path/to/mm-dataset , please check it!`，说明指定的数据集`/path/to/mm-dataset`是一个合法的多模态风格的数据集，但是数据集配置文件中的提示词工程配置的`mm_type`字段取值非法，而合法的`mm_type`字段取值必须为`["image", "video", "audio"]`之一。

## UTILS-DEPENDENCY-001
### 错误描述
执行过程中缺少必要依赖模块。
### 解决办法
若详细报错日志为`Failed to import required modules. Please install the necessary packages: pip install math_verify`，那么就参照详细日志的指引，执行`pip install math_verify`安装依赖库。

## UTILS-TYPE-001
### 错误描述
暂未有直接解决方法。
### 解决办法
如有解决此问题的诉求，[请提issue](https://github.com/AISBench/benchmark/issues)，请在issue描述中附上此错误码。

## UTILS-TYPE-002
### 错误描述
暂未有直接解决方法。
### 解决办法
如有解决此问题的诉求，[请提issue](https://github.com/AISBench/benchmark/issues)，请在issue描述中附上此错误码。

## UTILS-TYPE-003
### 错误描述
暂未有直接解决方法。
### 解决办法
如有解决此问题的诉求，[请提issue](https://github.com/AISBench/benchmark/issues)，请在issue描述中附上此错误码。

## UTILS-TYPE-004
### 错误描述
暂未有直接解决方法。
### 解决办法
如有解决此问题的诉求，[请提issue](https://github.com/AISBench/benchmark/issues)，请在issue描述中附上此错误码。

## UTILS-TYPE-005
### 错误描述
暂未有直接解决方法。
### 解决办法
如有解决此问题的诉求，[请提issue](https://github.com/AISBench/benchmark/issues)，请在issue描述中附上此错误码。

## UTILS-TYPE-006
### 错误描述
暂未有直接解决方法。
### 解决办法
如有解决此问题的诉求，[请提issue](https://github.com/AISBench/benchmark/issues)，请在issue描述中附上此错误码。

## UTILS-TYPE-007
### 错误描述
暂未有直接解决方法。
### 解决办法
如有解决此问题的诉求，[请提issue](https://github.com/AISBench/benchmark/issues)，请在issue描述中附上此错误码。

## UTILS-TYPE-008
### 错误描述
命令行参数值过大。
### 解决办法
若报错日志为`'--max-num-workers' must be <= 8, but got 9 ......`，这说明命令行参数`--max-num-workers`的值为9，而工具限定范围内只支持8个最大并发数，因此需要将`--max-num-workers`参数值修正为 <= 8。

## UTILS-TYPE-009
### 错误描述
命令行参数值类型不是整数。
### 解决办法
若报错日志为`'--max-num-workers' must be an integer, but got '9' ......`，这说明命令行参数`--max-num-workers`的值为字符串'9'，而工具限定范围内只支持整数类型，因此需要将`--max-num-workers`参数值修正为整数类型。

## UTILS-TYPE-010
### 错误描述
命令行参数值过小。
### 解决办法
若报错日志为`'--max-num-workers' must be >= 1, but got 0 ......`，这说明命令行参数`--max-num-workers`的值为0，而工具限定范围内只支持1个以上并发数，因此需要将`--max-num-workers`参数值修正为 >= 1。

## UTILS-PARAM-001
### 错误描述
暂未有直接解决方法。
### 解决办法
如有解决此问题的诉求，[请提issue](https://github.com/AISBench/benchmark/issues)，请在issue描述中附上此错误码。

## UTILS-PARAM-002
### 错误描述
自定义数据集场景下，配置文件`*.meta.json`中`request_count`参数不在合法范围。
### 解决办法
若报错为`Please make sure that the value of parameter 'request_count' can be converted to int(greater than 0).`，说明`*.meta.json`中`request_count`参数需要设置为 > 0。

## UTILS-PARAM-003
### 错误描述
自定义数据集场景下，配置文件`*.meta.json`中`min_value`参数大于`max_value`参数。
### 解决办法
若报错为`When the uniform distribution is set, parameter 'min_value' must be less than or equal to parameter 'max_value'.`，说明`*.meta.json`中`min_value`参数需要设置为 <= `max_value`参数，需要修正为`min_value` <= `max_value`。

## UTILS-PARAM-004
### 错误描述
自定义数据集场景下，配置文件`*.meta.json`中`min_value`和`max_value`参数不在合法范围
### 解决办法
如有解决此问题的诉求，[请提issue](https://github.com/AISBench/benchmark/issues)，请在issue描述中附上此错误码。

## UTILS-PARAM-005
### 错误描述
自定义数据集场景下，配置文件`*.meta.json`中缺少必要参数。
### 解决办法
例如报错为`When the uniform distribution is set, parameter 'min_value' and 'max_value' must be provided.`，说明在uniform场景下，`*.meta.json`中需要同时设置`min_value`和`max_value`参数。

## UTILS-PARAM-006
### 错误描述
自定义数据集场景下，配置文件`*.meta.json`中`percentage_distribute`参数不合法
### 解决办法
`percentage_distribute`的参数合法取值范围在详细日志中有介绍：
```
 Ensure the configuration data follows the format [max_tokens, percentage], where:
    - 'max_tokens' must be a positive number (greater than 0).
    - 'percentage' must be a float between 0 and 1 (greater than 0 and inclusive 1).
    - The sum of all 'percentage' values must equal exactly 1.
    Example valid format: [[1000, 0.5],[500,0.5]] or [[2000, 1.0]]
    Example invalid formats: [[0, 0.5]] (max_tokens <= 0), [[1000, 1.5]] (percentage > 1), [[1000, 0.3], [500,0.2]] (sum not 1)
```

## UTILS-PARAM-007
### 错误描述
自定义数据集场景下，配置文件`*.meta.json`中，定义数据分发方式的`method`参数值不在合法范围内。
### 解决办法
若报错为`Type of data distribution(method): uniform1 not supported, legal methods chosen from ['uniform', 'percentage'].`，说明`*.meta.json`中`method`参数值`uniform1`不在合法范围内，需要修正为`method`参数值为`uniform`或`percentage`。


## UTILS-PARAM-008
### 错误描述
自定义数据集场景下，配置文件`*.meta.json`中包含非法字段。
### 解决办法
若具体报错为`There are illegal keys: xxxxxx,yyyyyy`，说明`*.meta.json`中包含了`xxxxxx`和`yyyyyy`这两个非法字段，需要将这两个字段从`*.meta.json`中删除。

## UTILS-FILE-002
### 错误描述
模型配置文件中`path`参数表示的tokenizer路径不存在。
### 解决办法
若模型配置文件中内容为：
```python
# vllm_stream_api_chat.py中
models = [
    dict(
        # ......
        path="/path/to/invalid",
        # ......
    ),
]
```
具体报错日志为`Tokenizer path '/path/to/invalid' does not exist`，说明模型配置文件中`path`参数指定的tokenizer路径`/path/to/invalid`不存在（取空也是属于不存在），需要修正为存在的tokenizer路径。

## UTILS-FILE-003
### 错误描述
加载tokenizer文件失败。
### 解决办法
若报错为`Failed to load tokenizer from /path/to/tokenizer: ExceptionName: XXXXXX`，请确认`/path/to/tokenizer`路径下的tokenizer文件与当前运行环境的`transformers`版本是否匹配，若匹配，请依据`XXXXXX`表示的具体报错信息做进一步的问题定位。

## UTILS-FILE-004
### 错误描述
暂未有直接解决方法。
### 解决办法
如有解决此问题的诉求，[请提issue](https://github.com/AISBench/benchmark/issues)，请在issue描述中附上此错误码。

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
详细性能数据中每条请求的有效字段个数不同
### 解决办法
检查基础输出路径例如`outputs/default/20250628_151326`（查找打屏中`Current exp folder: `）下的`*_details.jsonl`中每条请求的有效字段个数是否一致，若不一致，则需要检查打屏日志历史中是否有其他错误导致性能数据文件未生成，依据其他错误日志的指引进一步定位。

## RUNNER-TASK-001
### 错误描述
测评任务执行失败
### 解决办法
例如具体报错为：`[ERROR] [RUNNER-TASK-001]task failed. OpenICLApiInfervllm-api-stream-chat/synthetic failed with code 1, see outputs/default/20251125_160128/logs/infer/vllm-api-stream-chat/synthetic.out`，请查看`outputs/default/20251125_160128/logs/infer/vllm-api-stream-chat/synthetic.out`中具体的报错信息，定位失败的原因。

## TINFER-PARAM-001
### 错误描述
模型配置文件中的最大并发数`batch_size`不在合法范围内
### 解决办法
若报错日志为`Concurrency must be greater than 0 and <= 100000, but got -1`，则表示模型的最大并发数配置为-1，需要在模型配置文件中将`batch_size`参数配置为一个大于0且小于等于100000的整数。
例如：
```python
# vllm_stream_api_chat.py中
models = [
    dict(
        attr="service",
        # ......
        batch_size=100,
        # ......
    ),
]
```

## TINFER-PARAM-002
### 错误描述
模型配置文件中的`generation_kwargs`参数的返回序列数`num_return_sequences`参数不在合法范围内
### 解决办法
若报错日志为`num_return sequences must be a positive integer, but got {0`，则表示模型的返回序列个数配置为0，需要在模型配置文件中将`num_return_sequences`参数配置为一个大于0的整数。
例如：
```python
# vllm_stream_api_chat.py中
models = [
    dict(
        attr="service",
        # ......
        generation_kwargs=dict(
            num_return_sequences=1,
        ),
        # ......
    ),
]
```

## TINFER-PARAM-004
### 错误描述
模型配置文件中`traffic_cfg`参数的爬升策略`ramp_up_strategy`参数不在合法范围内
### 解决办法
若报错日志为`Invalid ramp_up_strategy: {constant} only support 'linear' and 'exponential'`，则表示模型的请求发送策略配置为一个不在`['exponential', 'linear']`中的值，需要在模型配置文件中将`ramp_up_strategy`参数配置为`'exponential'`或`'linear'`。
例如：
```python
# vllm_stream_api_chat.py中
models = [
    dict(
        attr="service",
        # ......
        traffic_cfg=dict(
            ramp_up_strategy="linear",
        ),
        # ......
    ),
]
```

## TINFER-PARAM-005
### 错误描述
工具运行推理时虚拟内存占用过高
### 解决办法
若具体报错日志为：
```bash
Virtual memory usage too high: 90% > 80% (Total memory: 50 GB "Used: 45 GB, Available: 5 GB, Dataset needed memory size: 3000 MB)
```
说明当前系统内存为50GB，已使用45GB，可用5GB，而数据集需要3000MB内存，因此会触发该错误。解决方法分两种情况：
1. 若系统总内存不够，需要增加系统内存。
2. 若系统总内存足够，但是数据集需要的内存大于可用内存，需要清理当前服务器上被占用的内存或者缓存。

## TINFER-IMPL-001
### 错误描述
执行服务化推理任务时，推理任务内拉起多个进程时某个进程启动失败。
### 解决办法
若报错日志：
```bash
Failed to start worker x: XXXXXX, total workers to launch: 4
```
其中`x`为失败的进程编号，`XXXXXX`为具体失败的原因，`4`为总进程数。
解决方法：
1. 若该报错日志的出现次数与总进程数一致，则说明所有进程都启动失败，需要检查具体失败的原因并做相应处理后重试。
2. 若该报错日志的出现次数小于总进程数，则说明有进程启动失败，部分进程启动失败不影响评测任务的执行，但是会影响实际的的最大并发数`batch_size`，请根据实际情况自行决定是否需要手动中断先定位具体失败的原因。

## TINFER-RUNTIME-001
### 错误描述
测评推理服务化时，在预热阶段所有请求失败。
### 解决办法
若报错日志为`Exit task because all warmup requests failed, failed reasons: XXXXXX`，请根据具体的失败原因`XXXXXX`(**来自服务的错误信息**)定位问题并做相应处理后重试。

## TEVAL-PARAM-001
### 错误描述
推理生成候选解个数`n`和从其中采集的样本数`k`的取值非法
### 解决办法
若报错日志为
```bash
k and n must be greater than 0 and k <= n, but got k: 16, n: 8
```
则表示`k`大于`n`，需要将`k`配置为一个小于等于`n`的整数。
例如：
1. 在数据集配置文件中配置了`n`和`k`两个参数，则在配置文件中将两个参数的取值设置为合法范围的值：
```python
# 在aime2024_gen_0_shot_str.py中 k参数对应`k`
aime2024_datasets = [
    dict(
        abbr='aime2024',
        type=Aime2024Dataset,
        # ......
        k=4,
        n=8,
    )
]
```
2. 若数据集配置文件中未配置`n`这个参数，模型配置文件中的`num_return_sequences`参数值将作为`n`的取值，需要将数据集配置文件中的`k`配置为一个小于等于模型配置文件中`num_return_sequences`的整数。

```python
# 在vllm_stream_api_chat.py中 num_return_sequences参数对应`n`
models = [
    dict(
        attr="service",
        # ......
        generation_kwargs=dict(
            num_return_sequences=8,
        ),
        # ......
    ),
]

# 在aime2024_gen_0_shot_str.py中 k参数对应`k`
aime2024_datasets = [
    dict(
        abbr='aime2024',
        type=Aime2024Dataset,
        # ......
        k=4,
    )
]
```

## ICLI-PARAM-001
### 错误描述
数据集配置文件中构造提示词工程的retriever参数的type参数不是BaseRetriever的子类或者不是BaseRetriever的子类的list
### 解决办法
1. 如果想使用自定义的retriever类`CustomedRetriever`，请确保`CustomedRetriever`是`BaseRetriever`的子类。
2. 如果想使用多个自定义的retriever类`CustomedRetriever1, CustomedRetriever2`，则需要在数据集配置文件中配置`retriever`参数为`[CustomedRetriever1, CustomedRetriever2]`，且list中的每个类都需要继承自`BaseRetriever`。

## ICLI-PARAM-002
### 错误描述
多轮对话数据集配置文件中inferencer参数的infer_mode参数取值不在合法范围内
### 解决办法
以mtbench的配置文件为例，若mtbench_gen.py的配置如下:
```python
mtbench_infer_cfg = dict(
    # ......
    inferencer=dict(type=MultiTurnGenInferencer, infer_mode="every1")
)
```
日志报错为：
```bash
Multiturn dialogue infer model only supports every、last or every_with_gt, but got every1
```
正确的配置应当将infer_mode参数配置为`every`、`last`或`every_with_gt`中的一个。

## ICLI-PARAM-003
### 错误描述
命令行指定`--mode perf --pressure`进行性能压力测试时，模型配置文件中未指定batch_size参数
### 解决办法
以`vllm_stream_api_chat.py`配置文件为例：
```python
# 在vllm_stream_api_chat.py中
models = [
    dict(
        attr="service",
        # ......
        batch_size=16,
        # ......
    ),
]
```

## ICLI-PARAM-004
### 错误描述
模型配置文件中的最大并发数`batch_size`不在合法范围内
### 解决办法
若报错日志为`The range of batch_size is [1, 100000], but got -1. Please set it in datasets config`，则表示模型的最大并发数配置为-1，需要在模型配置文件中将`batch_size`参数配置为一个大于0且小于等于100000的整数。
例如：
```python
# vllm_stream_api_chat.py中
models = [
    dict(
        attr="service",
        # ......
        batch_size=100,
        # ......
    ),
]
```
## ICLI-PARAM-006
### 错误描述
PPL类的数据集不支持性能测试
### 解决办法
查看使用的数据集配置文件，例如：
```python
# ARC_c_ppl_0_shot_str.py中
ARC_c_infer_cfg = dict(
    # ......
    inferencer=dict(type=PPLInferencer))
```
`inferencer`的type为`PPLInferencer`，这种数据集配置文件不支持性能测试，因此需要换成其他数据集配置文件或者指定`--mode all`执行精度测评

## ICLI-PARAM-007
### 错误描述
PPL类的数据集不支持使用流式的模型配置进行推理
### 解决办法
查看使用的数据集配置文件，例如：
```python
# ARC_c_ppl_0_shot_str.py中
ARC_c_infer_cfg = dict(
    # ......
    inferencer=dict(type=PPLInferencer))
```
`inferencer`的type为`PPLInferencer`，这种数据集配置文件不支持使用流式的模型配置进行推理，因此需要换成其他数据集配置文件，或者`--models`指定非流式的模型配置文件，例如`--models vllm_api_general_chat`

## ICLI-IMPL-004
### 错误描述
BFCL数据集不支持性能测试
### 解决办法
1. 若希望使用BFCL数据集任务进行精度测试，单命令行中误指定`--mode perf`，则会进行性能测试，命令行中改为`--mode all`指定为精度测试。
2. 若希望使用BFCL数据集任务进行性能测试，则当前不支持。

## ICLI-IMPL-004
### 错误描述
接口类型为流式接口的模型任务不支持使用BFCL数据集进行精度测评
### 解决办法
参考[模型配置说明](../base_tutorials/all_params/models.md)，选取接口类型为文本接口（例如`vllm_api_general_chat`）的模型任务进行推理。

## ICLI-IMPL-008
### 错误描述
当前模型配置文件对应的模型后端没有实现PPL推理所需的方法
### 解决办法
参考文档（暂时还没有）查看哪些模型配置支持PPL推理，例如`vllm_api_general_chat`

## ICLI-IMPL-010
### 错误描述
PPL推理场景下某次推理结果中没有任何tokenid导致无法计算loss
### 解决办法
确认被测推理对象（推理服务）是否支持PPL推理，能否正常返回PPL推理所需的合法的`prompt_logprobs`

## ICLI-RUNTIME-001
### 错误描述
预热访问推理服务时获取推理结果失败了
### 解决办法
若日志为`Get result from cache queue failed: XXXXXX`其中`XXXXXX`为获取推理结果失败的具体原因，请依据具体原因做相应的处理（例如如果是超时相关的异常，请确认推理服务的超时时间是否设置合理或者检查当前配置能否正常访问推理服务）。

## ICLI-FILE-001
### 错误描述
落盘推理结果文件失败。
### 解决办法
1. 若日志为`Failed to write results to /path/to/outputs/default/20250628_151326/*/*/*.json: XXXXXX`，则表示精度场景推理结果落盘失败，请依据`XXXXXX`表示的具体保存原因（例如权限问题、磁盘空间不足等）进行排查和解决。
2. 若日志为`Failed to write results to /path/to/outputs/default/20250628_151326/*/*/*.jsonl: XXXXXX`，则表示性能场景推理结果落盘失败，请依据`XXXXXX`表示的具体保存原因（例如权限问题、磁盘空间不足等）进行排查和解决。

## ICLI-FILE-002
### 错误描述
将numpy格式的数据（例如每条请求的ITL数据）保存到数据库中失败
### 解决办法
若日志为`Failed to save numpy array to database: XXXXXX`，则表示将numpy格式的数据保存到数据库中失败，请依据`XXXXXX`表示的具体保存原因（例如数据库连接问题、数据库表不存在等）进行排查和解决。

## ICLE-DATA-002
### 错误描述
配置的推理生成候选解个数`n`与实际返回的候选解个数不一致。
### 解决办法
1. 若命令行中指定`--mode all`或者不指定`--mode`，则表示执行infer + evaluate，这种场景触发此异常说明工具本身存在bug，可以在[issue](https://github.com/AISBench/benchmark/issues)中反馈。
2. 若命令行中指定`--mode eval`则基于之前的推理结果进行evaluate，若异常报错为;
`Replication length mismatch, len of replications: 4 != n: 8`，那么需要在数据集任务对应的配置文件中将参数`n`设置为replication的数量`4`:
```python
# 在aime2024_gen_0_shot_str.py中 k参数对应`k`
aime2024_datasets = [
    dict(
        abbr='aime2024',
        type=Aime2024Dataset,
        # ......
        n=4,
    )
]
```

## ICLR-TYPE-001
### 错误描述
数据集配置文件中，提示词模板的类型不正确，当前仅支持`str`或`dict`类型。
### 解决办法
确认数据集配置文件中，推理配置中的提示词模板的类型为`str`或`dict`，例如：
```python
# 在aime2024_gen_0_shot_str.py中
aime2024_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{question}\nPlease reason step by step, and put your final answer within \\boxed{}.' # str类型
    ),
    # ......
)

# 在aime2024_gen_0_shot_chat_prompt.py中
aime2024_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict( # dict类型
            round=[
                dict(
                    role="HUMAN",
                    prompt="{question}\nPlease reason step by step, and put your final answer within \\boxed{}.",
                ),
            ],
        ),
    ),
    # ......
)
```
如果`template`参数的取值的类型不对，请更正为`str`或`dict`类型。

## ICLR-TYPE-002
### 错误描述
数据集配置文件中，提示词模板的类型为`dict`时，其中所有键值对value的取值类型错误，取值类型当前仅支持`str`，`list`和`dict`。
### 解决办法
确认数据集配置文件中，推理配置中的提示词模板中所有键值对value的取值类型为`str`，`list`或`dict`，例如：
```python
# 在aime2024_gen_0_shot_chat_prompt.py中
aime2024_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict( # dict类型
            round=[
                dict(
                    role="HUMAN", # str 类型
                    prompt="{question}\nPlease reason step by step, and put your final answer within \\boxed{}.", # str类型
                ),
            ],
        ),
    ),
    # ......
)
```

## ICLR-PARAM-001
### 错误描述
数据集配置文件中，提示词模板配置了`ice_token`参数时，`template`参数的取值中未包含`ice_token`参数的取值
### 解决办法
1. 当`template`参数类型为`str`时，确认`template`取值的字符串中包含`ice_token`参数的取值。例如：
```python
# 在ceval_gen_5_shot_str.py中
ceval_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=f'以下是中国关于{_ch_name}考试的单项选择题，请选出其中的正确答案。\n</E>{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n答案: {{answer}}', # 字符串中包含ice_token的取值'</E>'
        ice_token='</E>',
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
    inferencer=dict(type=GenInferencer),
)

```


2. 当`template`参数类型为`dict`时，确认`template`取值的字典中所有键值对value的取值中存在`ice_token`参数的取值。例如：
```python
# 在aime2024_gen_0_shot_chat_prompt.py中
cmmlu_infer_cfg = dict(
    # ......
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='</E>', # 与ice_token取值相同
            round=[
                dict(role='HUMAN', prompt=prompt_prefix+QUERY_TEMPLATE),
            ],
        ),
        ice_token='</E>',
    ),
    # ......
)
```

## ICLR-PARAM-002
### 错误描述
数据集配置文件需要基于训练集构造few-shots时未指定`ice_template`参数。
### 解决办法
以cmmlu_gen_5_shot_cot_chat_prompt.py为例，该配置中指定`retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),`来构造few-shots，因此必须指定`ice_template`参数，可以参考其内容做修改：
```python
cmmlu_infer_cfg = dict(
    ice_template=dict( # 必须配置ice_template
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=prompt_prefix+QUERY_TEMPLATE,
            ),
            dict(role='BOT', prompt="{answer}\n",)
        ]),
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='</E>',
            round=[
                dict(role='HUMAN', prompt=prompt_prefix+QUERY_TEMPLATE),
            ],
        ),
        ice_token='</E>',
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]), # 指定了5-shots
    inferencer=dict(type=GenInferencer),
)

```

## ICLR-PARAM-003
### 错误描述
多模态类的数据集配置文件中，提示词模板中的`prompt_mm`参数的key取值不是["text", "image", "video", "audio"]之一。
### 解决办法
以`textvqa_gen_base64.py`为例，该配置中提示词模板中的`prompt_mm`参数的key取值为"text"，"image"，"video"，"audio"之一，可以参考其内容做修改：
```python
textvqa_infer_cfg = dict(
    prompt_template=dict(
        type=MMPromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt_mm={ # prompt_mm参数的key取值为"text"，"image"，"video"，"audio"之一
                    "text": {"type": "text", "text": "{question} Answer the question using a single word or phrase."},
                    "image": {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}},
                    "video": {"type": "video_url", "video_url": {"url": "data:video/jpeg;base64,{video}"}},
                    "audio": {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,{audio}"}},
                })
            ]
            )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)
```

## ICLR-PARAM-004
### 错误描述
数据集配置文件中构造few-shots的`fix_id_list`中的id取值超出了训练集可选取id的范围。
### 解决办法
若数据集配置文件中的构造few-shots的配置如下:
```python
retriever=dict(type=FixKRetriever, fix_id_list=[1,2,5,8]),
```
详细报错日志为`Fix-K retriever index 8 is out of range of [0, 8)`，说明`fix_id_list`中的id取值超出了训练集可选取id的范围[0, 8)，需要修正在此范围内。

## ICLR-IMPL-002
### 错误描述
在数据集配置文件中的提示词模板中，未配置`ice_token`参数。
### 解决办法
1. 若同时存在`prompt_template`参数和`ice_template`参数，日志报错为`ice_token of prompt_template is not provided`，则`prompt_template`参数中必须存在`ice_token`参数，例如
```python
cmmlu_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=prompt_prefix+QUERY_TEMPLATE,
            ),
            dict(role='BOT', prompt="{answer}\n",)
        ]),
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='</E>',
            round=[
                dict(role='HUMAN', prompt=prompt_prefix+QUERY_TEMPLATE),
            ],
        ),
        ice_token='</E>', # 必须设置
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]), # 指定了5-shots
    inferencer=dict(type=GenInferencer),
)
```
2. 若仅存在`ice_template`参数，日志报错为`ice_token of ice_template is not provided`，则`ice_template`参数中必须存在`ice_token`参数，例如
```python
ceval_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=f'以下是中国关于{_ch_name}考试的单项选择题，请选出其中的正确答案。\n</E>{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n答案: {{answer}}',
        ice_token='</E>', # 必须存在
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
    inferencer=dict(type=GenInferencer),
)
```

## ICLR-IMPL-003
### 错误描述
数据集配置文件中缺失必要的模板字段
### 解决办法
若报错日志为`Leaving prompt as empty is not supported`，说明数据集配置文件中至少需要存在`prompt_template`参数和`ice_template`参数的其中一个。
例如
```python
cmmlu_infer_cfg = dict( # ice_template和prompt_template至少存在一个
    ice_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=prompt_prefix+QUERY_TEMPLATE,
            ),
            dict(role='BOT', prompt="{answer}\n",)
        ]),
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='</E>',
            round=[
                dict(role='HUMAN', prompt=prompt_prefix+QUERY_TEMPLATE),
            ],
        ),
        ice_token='</E>', # 必须设置
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]), # 指定了5-shots
    inferencer=dict(type=GenInferencer),
)
```

## MODEL-IMPL-001
### 错误描述
当基于`BaseAPIModel`类实现一个新的类时，未实现`parse_text_response`方法，无法通过文本接口测试推理服务。
### 解决办法
（面向开发者）实现基于`BaseAPIModel`类的子类时，若希望通过文本接口测试推理服务，需要实现
`parse_text_response`方法，用于解析模型返回的文本响应，将其转换为模型推理服务的输出格式。

## MODEL-IMPL-002
### 错误描述
当基于`BaseAPIModel`类实现一个新的类时，未实现`parse_stream_response`方法，无法通过流式接口测试推理服务。
### 解决办法
（面向开发者）实现基于`BaseAPIModel`类的子类时，若希望通过流式接口测试推理服务，需要实现
`parse_stream_response`方法，用于解析模型返回的流式响应，将其转换为模型推理服务的输出格式。

## MODEL-PARAM-002
### 错误描述
数据集配置文件中，chat类型的prompt template中没有包含`role`或`fallback_role`字段
### 解决办法
参考以下配置文件内容：
```python
cmmlu_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN', # 包含'role字段'
                prompt=prompt_prefix+QUERY_TEMPLATE,
            ),
            dict(role='BOT', prompt="{answer}\n",)
        ]),
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='</E>',
            round=[
                dict(role='HUMAN', prompt=prompt_prefix+QUERY_TEMPLATE),
            ],
        ),
        ice_token='</E>',
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]), # 指定了5-shots
    inferencer=dict(type=GenInferencer),
)
```

## MODEL-PARAM-003
### 错误描述
数据集配置文件中提示词工程chat模板的`role`参数的取值不在合法范围。
### 解决办法
若数据集配置文件中chat模板相关的配置如下：
```python
# 以aime2024_gen_0_shot_chat_prompt.py为例
aime2024_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN1",
                    prompt="{question}\nPlease reason step by step, and put your final answer within \\boxed{}.",
                ),
            ],
        ),
    ),
    # ......
)
```
报错日志为`Unknown role HUMAN1 in chat template, legal role chosen from ['HUMAN', 'BOT', 'SYSTEM'].`说明chat模板中role参数的值为HUMAN1，而合法的role值为HUMAN、BOT、SYSTEM，因此需要将role参数值修正为HUMAN、BOT、SYSTEM中的一个。

## MODEL-PARAM-004
### 错误描述
暂未有直接解决方法。
### 解决办法
如有解决此问题的诉求，[请提issue](https://github.com/AISBench/benchmark/issues)，请在issue描述中附上此错误码。

## MODEL-PARAM-005
### 错误描述
暂未有直接解决方法。
### 解决办法
如有解决此问题的诉求，[请提issue](https://github.com/AISBench/benchmark/issues)，请在issue描述中附上此错误码。

## MODEL-TYPE-001
### 错误描述
数据集配置文件中，提示词工程模板中不支持一组字符串。
### 解决办法
若数据集配置文件中提示词模板相关的配置如下：
```python
# 以aime2024_gen_0_shot_chat_prompt.py为例
aime2024_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[ # list中包含多个字符串
                "{question}\nPlease reason step by step, and put your final answer within \\boxed{}.",
                "{question}\nPlease reason step by step, and put your final answer within \\boxed{}."
            ],
        ),
    ),
    # ......
)
```
则会出现报错：`Mixing str without explicit role is not allowed in API models!`，请将`round`改为合法的chat模板，例如:
```python
round=[
    dict(
        role="HUMAN1",
        prompt="{question}\nPlease reason step by step, and put your final answer within \\boxed{}.",
    ),
],
```

## MODEL-TYPE-002
### 错误描述
暂未有直接解决方法。
### 解决办法
如有解决此问题的诉求，[请提issue](https://github.com/AISBench/benchmark/issues)，请在issue描述中附上此错误码。

## MODEL-TYPE-003
### 错误描述
暂未有直接解决方法。
### 解决办法
如有解决此问题的诉求，[请提issue](https://github.com/AISBench/benchmark/issues)，请在issue描述中附上此错误码。

## MODEL-TYPE-004
### 错误描述
暂未有直接解决方法。
### 解决办法
如有解决此问题的诉求，[请提issue](https://github.com/AISBench/benchmark/issues)，请在issue描述中附上此错误码。

## MODEL-DATA-001
### 错误描述
模型任务向被测推理服务获取模型名称信息失败。
### 解决办法
若报错为`Failed to get service model path from http://url-to-infer-service. Error: XXXXXX`，说明访问`http://url-to-infer-service/v1/models`失败，需要检查被测推理服务是否正常运行，以及是否开启了`/v1/models`子服务，也可依据具体报错`XXXXXX`，定位访问`http://url-to-infer-service/v1/models`失败的原因。若url`http://url-to-infer-service/`不支持`v1/models`子服务，可以在模型配置文件中在`model`参数中配置模型名称，例如：
```python
# vllm_api_stream_chat.py中
models = [
    dict(
        # ......
        model="name_of_model",
        # ......
    )
]
```
## MODEL-DATA-002
### 错误描述
数据集配置文件中未包含必要参数。
### 解决办法
若报错为`Invalid prompt content: without 'prompt' or 'prompt_mm' param!`，说明数据集配置文件中未包含`prompt`或`prompt_mm`参数之一，需要在数据集配置文件中添加这两个参数之一，例如：
```python
# 以aime2024_gen_0_shot_chat_prompt.py为例
aime2024_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict( # 必须包含prompt或prompt_mm字段之一
                    role="HUMAN",
                    prompt="{question}\nPlease reason step by step, and put your final answer within \\boxed{}.",
                ),
            ],
        ),
    ),
    # ......
)



```

## MODEL-DATA-003
### 错误描述
以JSON格式解析请求的返回结果时失败。
### 解决办法
若报错为`Unexpected response format. Please check 'error_info' in {dataset_abbr}_failed.jsonl for more information.`，那么需要在当前推理任务落盘路径（例如`outputs/default/20250628_151326/performances/vllm-api-stream-chat/`）下的`{dataset_abbr}_failed.jsonl`文件中查看具体的错误信息(`error_info`字段的内容)，并进一步寻求解决方法。

## MODEL-CFG-001
### 错误描述
本地模型配置文件中未配置`max_seq_len`参数。
### 解决办法
若报错为`max_seq_len is not provided and cannot be inferred from the model config.`说明需要在本地模型配置文件中添加`max_seq_len`参数，例如：
```python
# hf_chat_model.py中
models = [
    dict(
        attr="local",
        # ......
        max_seq_len=2048,
        # ......
    )
]
```

## MODEL-MOD-001
### 错误描述
模型执行需要的特殊依赖未安装。
### 解决办法
若报错为`fastchat module not found. Please install with\npip install "fschat[model_worker,webui]"`，说明缺少`fastchat`这个依赖，可以执行`pip install "fschat[model_worker,webui]"`安装。

## DSET-CFG-001
### 错误描述
数据集配置文件中缺少`path`字段指定数据集路径。
### 解决办法
若报错为`The 'path' argument is required to load the dataset.`，说明数据集配置文件中未包含`path`字段，需要在数据集配置文件中添加`path`字段，例如：
```python
# aime2024_gen_0_shot_chat_prompt.py中
aime2024_datasets = [
    dict(
        abbr='aime2024',
        type=Aime2024Dataset,
        path='ais_bench/datasets/aime/aime.jsonl', # 必填字段，配置数据集路径
        # ......
    )
]
```

## DSET-FILE-001
### 错误描述
数据集文件不存在。
### 解决办法
1. 若报错为`Path is not a directory or Parquet file: /path/to/dataset.jsonl`，说明`/path/to/dataset.jsonl`不是所需的`.parquet`格式的数据集，请确认数据集格式符合预期。
2. 若报错为`No Parquet file found in /path/to/dataset/.`，说明在`/path/to/dataset/`路径下找不到`.parquet`格式的数据集，请确认数据集格式符合预期。
3. 若报错为`"Dataset file not found: /path/to/dataset/`，说明数据集路径`/path/to/dataset/`本身不存在，请确认数据集路径是否与预期传入的一致。

## DSET-DATA-002
### 错误描述
数据集内容结构不合法。
### 解决办法
请依据详细报错的信息检查数据集内容中存在的格式问题。

## DSET-DATA-005
### 错误描述
暂未有直接解决方法。
### 解决办法
如有解决此问题的诉求，[请提issue](https://github.com/AISBench/benchmark/issues)，请在issue描述中附上此错误码。

## DSET-DATA-006
### 错误描述
暂未有直接解决方法。
### 解决办法
如有解决此问题的诉求，[请提issue](https://github.com/AISBench/benchmark/issues)，请在issue描述中附上此错误码。

## DSET-PARAM-002
### 错误描述
推理生成候选解个数`n`和从其中采集的样本数`k`的取值非法
### 解决办法
若报错日志为
```bash
Maximum value of `k` 4 must be less than or equal to `n` 8
```
则表示`k`大于`n`，需要将`k`配置为一个小于等于`n`的整数。
例如：
1. 在数据集配置文件中配置了`n`和`k`两个参数，则在配置文件中将两个参数的取值设置为合法范围的值：
```python
# 在aime2024_gen_0_shot_str.py中 k参数对应`k`
aime2024_datasets = [
    dict(
        abbr='aime2024',
        type=Aime2024Dataset,
        # ......
        k=4,
        n=8,
    )
]
```

## DSET-PARAM-004
### 错误描述
数据集配置文件中参数非法。
### 解决办法
请依据详细报错的信息检查数据集配置文件中内容中存在的参数取值非法问题。

## DSET-DEPENDENCY-002
### 错误描述
缺少数据集任务evaluate所需的依赖。
### 解决办法
若报错为:
```bash
Please install human_eval use following steps:
git clone git@github.com:open-compass/human-eval.git
cd human-eval && pip install -e .
```
参加该报错日志内容依次执行`git clone git@github.com:open-compass/human-eval.git`和`cd human-eval && pip install -e .`安装`human-eval`库

## DSET-MTRC-001
### 错误描述
暂未有直接解决方法。
### 解决办法
如有解决此问题的诉求，[请提issue](https://github.com/AISBench/benchmark/issues)，请在issue描述中附上此错误码。

## DSET-MTRC-003
### 错误描述
暂未有直接解决方法。
### 解决办法
如有解决此问题的诉求，[请提issue](https://github.com/AISBench/benchmark/issues)，请在issue描述中附上此错误码。