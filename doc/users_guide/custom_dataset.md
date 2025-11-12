# 自定义数据集使用说明

本教程仅供临时性的、非正式的数据集使用。

在本教程中，我们将会介绍如何在不实现 config，不修改 ais_bench 源码的情况下，对一新增数据集进行测试的方法。我们支持的任务类型包括选择 (`mcq`) 和问答 (`qa`) 两种，目前 `mcq` 和 `qa` 均仅支持 `gen` 推理。

## 数据集格式

我们支持 `.jsonl` 和 `.csv` 两种格式的数据集。

### 选择题 (`mcq`)

对于选择 (`mcq`) 类型的数据，默认的字段（其他字段请参考[特殊字段](#特殊字段)部分内容）如下：

- `question`: 表示选择题的题干
- `A`, `B`, `C`, ... : 使用单个大写字母表示选项，个数不限定。默认只会从 `A` 开始，解析连续的字母作为选项。
- `answer`: 表示选择题的正确答案，其值必须是上述所选用的选项之一，如 `A`, `B`, `C` 等。

`.jsonl` 格式样例如下：

```jsonl
{"question": "165+833+650+615=", "A": "2258", "B": "2263", "C": "2281", "answer": "B"}
{"question": "368+959+918+653+978=", "A": "3876", "B": "3878", "C": "3880", "answer": "A"}
{"question": "776+208+589+882+571+996+515+726=", "A": "5213", "B": "5263", "C": "5383", "answer": "B"}
{"question": "803+862+815+100+409+758+262+169=", "A": "4098", "B": "4128", "C": "4178", "answer": "C"}
```

`.csv` 格式样例如下:

```csv
question,A,B,C,answer
127+545+588+620+556+199=,2632,2635,2645,B
735+603+102+335+605=,2376,2380,2410,B
506+346+920+451+910+142+659+850=,4766,4774,4784,C
504+811+870+445=,2615,2630,2750,B
```

### 问答题 (`qa`)

对于问答 (`qa`) 类型的数据，默认的字段（其他字段请参考[特殊字段](#特殊字段)部分内容）如下：

- `question`: 表示问答题的题干
- `answer`: 表示问答题的正确答案。可缺失，表示该数据集无正确答案。


`.jsonl` 格式样例如下：

```jsonl
{"question": "752+361+181+933+235+986=", "answer": "3448"}
{"question": "712+165+223+711=", "answer": "1811"}
{"question": "921+975+888+539=", "answer": "3323"}
{"question": "752+321+388+643+568+982+468+397=", "answer": "4519"}
```

`.csv` 格式样例如下：

```csv
question,answer
123+147+874+850+915+163+291+604=,3967
149+646+241+898+822+386=,3142
332+424+582+962+735+798+653+214=,4700
649+215+412+495+220+738+989+452=,4170
```

## 通过命令行指定模型和自定义数据集

自定义数据集可直接通过命令行来调用开始评测。

|参数|说明|样例|
| ----- | ----- | ---- |
|--models|和普通数据集使用方式一致，指定模型推理后端任务名称（对应ais_bench/benchmark/configs/models路径下一个已经实现的默认模型配置文件），支持传入多个任务名称，支持的任务范围请参考父级目录的README中使用普通数据集为例的方式，"命令行指定模型和数据集" ("[AISBench benchmark评测工具](../README.md)"中的"命令行指定模型和数据集"章节) 方式的必选配置，与"配置文件指定模型和数据集" ("[AISBench benchmark评测工具](../README.md)"中的"配置文件指定模型和数据集"章节) 方式的"config"参数二选一|--models vllm_api_general|
|--custom-dataset-path|指定自定义数据集路径（支持绝对/相对路径），当`--datasets`配置时，该参数无效，默认未配置|--custom-dataset-path xxx/test_mcq.csv|
|--custom-dataset-meta-path|指定数据集补充信息.meta.json文件路径（支持绝对/相对路径）|--custom-dataset-meta-path xxx/test_mcq.csv.meta.json|
|--custom-dataset-data-type|指定自定义数据集的任务类型，目前支持选项[`mcq`,`qa`]，表示选择题类型(`mcq`) 和问答题类型 (`qa`) 两种，未配置时会根据数据集格式自动识别类型，配置时则根据填入参数进行对应格式的解析|--custom-dataset-data-type mcq|
|--custom-dataset-infer-method|指定自定义数据集的推理类型，目前仅支持选项[`gen`]，未配置时默认为`gen`|--custom-dataset-infer-method gen|

其余参数均和普通数据集一致，同样支持通过vllm和mindie两种api访问对应的推理服务化

```shell
ais_bench \
    --models vllm_api_general \
    --custom-dataset-path xxx/test_mcq.csv \
    --custom-dataset-data-type mcq \
    --mode all
```

```shell
ais_bench \
    --models mindie_stream_api_general \
    --custom-dataset-path xxx/test_qa.jsonl \
    --custom-dataset-data-type qa \
    --custom-dataset-infer-method gen
```

在绝大多数情况下，`--custom-dataset-data-type` 和 `--custom-dataset-infer-method` 可以省略，ais_bench 会根据以下逻辑进行设置：

- 如果从数据集文件中可以解析出选项，如 `A`, `B`, `C` 等，则认定该数据集为 `mcq`，否则认定为 `qa`。
- 默认 `--custom-dataset-infer-method` 为 `gen`。

## 通过配置文件指定模型和自定义数据集

该方式目前仅支持精度测评场景。其余参数均和普通数据集一致，同样支持通过vllm和mindie两种api访问对应的推理服务化

命令行参考：

```shell
ais_bench ais_bench/configs/api_examples/infer_api_vllm_general.py
```

```shell
ais_bench ais_bench/configs/api_examples/infer_api_mindie_stream_general.py
```

在原配置文件中，直接向 `datasets` 变量中添加新的项即可。同普通数据集一致，该方式下支持自定义数据集与普通数据集混用。

```python
datasets = [
    ..., # 普通数据集
    {"path": "xxx/test_qa.jsonl", "data_type": "qa", "infer_method": "gen"},
    ..., # 普通数据集
]
```

## 数据集补充信息`.meta.json`使用指南
目前仅支持性能测评场景。ais_bench 会默认尝试对输入的数据集文件进行解析，因此在绝大多数情况下，`.meta.json` 文件都是 **不需要** 的。但是，如果原生数据集中没有指定max_tokens，或者需要通过配置进行数据采样等，则需要在 `.meta.json` 文件中进行指定。

我们会在数据集同级目录下，以文件名+`.meta.json` 的形式放置一个表征数据集使用方法的文件，样例文件结构如下：
```tree
.
├── test_mcq.csv
├── test_mcq.csv.meta.json
├── test_qa.jsonl
└── test_qa.jsonl.meta.json
```
当前支持字段如下：
- `request_count` (str or int): 最终数据集生成request_count条case，数量不足则循环填充，数量超过则截取前request_count条，不设置默认原始数据集的长度。
- `sampling_mode` (str): 采样数据集的模式，可选值为 `shuffle`、`random`、`default`.
- `output_config` : 控制每条请求中模型输出等相关选项。
- `method` (str): 数据分布的类型，可选值为 `uniform`(均匀分布)、`percentage`（百分比分布）。
- `params` (str): 数据分布设置的参数。
- `min_value` (str or int): 生成数据最小长度，当method: uniform有效。
- `max_value` (str or int): 生成数据最大长度，当method: uniform有效。
- `percentage_distribute` (list): 生成输出长度的百分比分布，当method: percentage有效，格式为二维数组，其中第一个元素表示输出长度，第二个元素表示百分比。

下面分别提供百分比分布和均匀分布的配置示例：
```json
{
    "output_config": {
        "method": "percentage",
        "params": {
            "percentage_distribute": [
                [100, 0.5],
                [200, 0.3],
                [400, 0.2]
            ]
        }
    },
    "request_count": "10",
    "sampling_mode": "shuffle"
}
```
```json
{
    "output_config": {
        "method": "uniform",
        "params": {
            "min_value": 100,
            "max_value": 200
        }
    }
}
```
## 特殊字段

### 最大输出长度：`max_tokens`

在`csv`与`jsonl`两种格式的数据集文件中，均支持以「每条请求」为粒度设置最大输出长度——只需在`jsonl`文件的每条对象或`csv`文件的每一行中新增`max_tokens`字段并赋予相应数值即可。该字段目前尚不适用于性能压测场景。

示例如下:

- `.jsonl` 格式：

  - mcq类型：

    ```jsonl
    {"question": "165+833+650+615=", "A": "2258", "B": "2263", "C": "2281", "answer": "B", "max_tokens": 512}
    {"question": "368+959+918+653+978=", "A": "3876", "B": "3878", "C": "3880", "answer": "A", "max_tokens": 1024}
    {"question": "776+208+589+882+571+996+515+726=", "A": "5213", "B": "5263", "C": "5383", "answer": "B", "max_tokens": 2048}
    {"question": "803+862+815+100+409+758+262+169=", "A": "4098", "B": "4128", "C": "4178", "answer": "C", "max_tokens": 256}
    ```

  - qa类型：

    ```jsonl
    {"question": "165+833+650+615=", "answer": "2263", "max_tokens": 512}
    {"question": "368+959+918+653+978=", "answer": "3876", "max_tokens": 1024}
    {"question": "776+208+589+882+571+996+515+726=", "answer": "5263", "max_tokens": 2048}
    {"question": "803+862+815+100+409+758+262+169=", "answer": "4178", "max_tokens": 256}
    ```

- `.csv` 格式：

  - mcq类型：

    ```csv
    question,A,B,C,answer,max_tokens
    127+545+588+620+556+199=,2632,2635,2645,B,512
    735+603+102+335+605=,2376,2380,2410,B,1024
    506+346+920+451+910+142+659+850=,4766,4774,4784,C,2048
    504+811+870+445=,2615,2630,2750,B,256
    ```

  - qa类型：

    ```csv
    question,answer,max_tokens
    127+545+588+620+556+199=,2635,512
    735+603+102+335+605=,2380,1024
    506+346+920+451+910+142+659+850=,4784,2048
    504+811+870+445=,2630,256
    ```

- 数据集样例及对应性能测评结果示例图：
  - 数据集样例：
    ![custom_dataset_example.img](../img/自定义数据集/custom_dataset_example.png)

  - 性能测评结果示例图：
    ![custom_results_example.img](../img/自定义数据集/custom_results_example.png)
