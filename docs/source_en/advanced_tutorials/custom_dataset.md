# Guide to Using Custom Datasets

This tutorial is only for temporary and informal use of datasets.

In this tutorial, we will explain how to test a new dataset without implementing a config or modifying the source code of `ais_bench`. The supported task types include **multiple-choice questions (mcq)** and **question-answering (qa)**. Currently, both `mcq` and `qa` only support `gen` (generation-based) inference.


## Dataset Formats

We support two dataset formats: `.jsonl` and `.csv`.


### Multiple-Choice Questions (mcq)

For `mcq`-type data, the default fields are as follows (for other fields, refer to the [Special Fields](#special-fields) section):

- `question`: The main stem of the multiple-choice question.
- `A`, `B`, `C`, ... : Options represented by single uppercase letters (unlimited in number). By default, the system only parses consecutive letters starting from `A` as valid options.
- `answer`: The correct answer to the multiple-choice question, which must be one of the above options (e.g., `A`, `B`, `C`).
- If an exact match cannot be found during accuracy calculation, the **Levenshtein Distance Algorithm** will be used to select the closest answer. This may cause misjudgment and result in an artificially high accuracy score.


#### Example of `.jsonl` Format
```json
{"question": "165+833+650+615=", "A": "2258", "B": "2263", "C": "2281", "answer": "B"}
{"question": "368+959+918+653+978=", "A": "3876", "B": "3878", "C": "3880", "answer": "A"}
{"question": "776+208+589+882+571+996+515+726=", "A": "5213", "B": "5263", "C": "5383", "answer": "B"}
{"question": "803+862+815+100+409+758+262+169=", "A": "4098", "B": "4128", "C": "4178", "answer": "C"}
```


#### Example of `.csv` Format
```bash
question,A,B,C,answer
127+545+588+620+556+199=,2632,2635,2645,B
735+603+102+335+605=,2376,2380,2410,B
506+346+920+451+910+142+659+850=,4766,4774,4784,C
504+811+870+445=,2615,2630,2750,B
```


### Question-Answering (qa)

For `qa`-type data, the default fields are as follows (for other fields, refer to the [Special Fields](#special-fields) section):

- `question`: The main stem of the question.
- `answer`: The correct answer to the question (optional; if missing, the dataset is considered to have no correct answers).


#### Example of `.jsonl` Format
```json
{"question": "752+361+181+933+235+986=", "answer": "3448"}
{"question": "712+165+223+711=", "answer": "1811"}
{"question": "921+975+888+539=", "answer": "3323"}
{"question": "752+321+388+643+568+982+468+397=", "answer": "4519"}
```


#### Example of `.csv` Format
```bash
question,answer
123+147+874+850+915+163+291+604=,3967
149+646+241+898+822+386=,3142
332+424+582+962+735+798+653+214=,4700
649+215+412+495+220+738+989+452=,4170
```


## Specify Models and Custom Datasets via Command Line

Custom datasets can be directly invoked for evaluation via the command line.

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--models` | Same as when using standard datasets. Specifies the name of the model inference backend task (corresponding to a pre-implemented default model config file under `ais_bench/benchmark/configs/models`). Multiple task names can be passed. For supported tasks, refer to the README in the parent directory (using standard datasets as examples). | `--models vllm_api_general` |
| `--custom-dataset-path` | Specifies the path to the custom dataset (absolute/relative paths supported). Invalid if `--datasets` is configured. Not configured by default. | `--custom-dataset-path xxx/test_mcq.csv` |
| `--custom-dataset-meta-path` | Specifies the path to the dataset supplementary info file (`.meta.json`; absolute/relative paths supported). | `--custom-dataset-meta-path xxx/test_mcq.csv.meta.json` |
| `--custom-dataset-data-type` | Specifies the task type of the custom dataset. Currently supports `mcq` (multiple-choice) and `qa` (question-answering). If not configured, the system automatically identifies the type based on the dataset format; if configured, parses the dataset according to the specified type. | `--custom-dataset-data-type mcq` |
| `--custom-dataset-infer-method` | Specifies the inference type for the custom dataset. Currently only supports `gen`. Defaults to `gen` if not configured. | `--custom-dataset-infer-method gen` |


Other parameters are the same as those for standard datasets. The system also supports accessing inference services via two APIs: `vllm` and `mindie`.


#### Example Command 1 (mcq Dataset with vllm API)
```shell
ais_bench \
    --models vllm_api_general \
    --custom-dataset-path xxx/test_mcq.csv \
    --custom-dataset-data-type mcq \
    --mode all
```


#### Example Command 2 (qa Dataset with mindie API)
```shell
ais_bench \
    --models mindie_stream_api_general \
    --custom-dataset-path xxx/test_qa.jsonl \
    --custom-dataset-data-type qa \
    --custom-dataset-infer-method gen
```


In most cases, `--custom-dataset-data-type` and `--custom-dataset-infer-method` can be omitted. The `ais_bench` system will set them automatically based on the following logic:
- If options (e.g., `A`, `B`, `C`) can be parsed from the dataset, the dataset is identified as `mcq`; otherwise, it is identified as `qa`.
- The default value of `--custom-dataset-infer-method` is `gen`.


## Specify Models and Custom Datasets via Config Files

This method currently only supports **accuracy evaluation scenarios**. Other parameters are the same as those for standard datasets, and the system also supports accessing inference services via `vllm` and `mindie` APIs.


### Example Command
```shell
# Use vllm API
ais_bench ais_bench/configs/api_examples/infer_api_vllm_general.py

# Use mindie API
ais_bench ais_bench/configs/api_examples/infer_api_mindie_stream_general.py
```


### Config File Modification
In the original config file, simply add a new entry to the `datasets` variable. Similar to standard datasets, this method supports mixing custom datasets with standard datasets.

```python
datasets = [
    ...,  # Standard datasets
    {"path": "xxx/test_qa.jsonl", "data_type": "qa", "infer_method": "gen"},
    ...,  # Standard datasets
]
```


## Guide to Using Dataset Supplementary Info (`.meta.json`)

This feature currently only supports **performance evaluation scenarios**. The `ais_bench` system will automatically attempt to parse the input dataset file, so in most cases, a `.meta.json` file is **not required**. However, if the original dataset does not specify `max_tokens`, or if you need to configure data sampling, you must define these settings in a `.meta.json` file.


### File Structure
The `.meta.json` file is placed in the same directory as the dataset, named in the format `[dataset_filename].meta.json`. Example structure:
```bash
.
├── test_mcq.csv
├── test_mcq.csv.meta.json
├── test_qa.jsonl
└── test_qa.jsonl.meta.json
```


### Supported Fields
- `request_count` (str/int): The final number of test cases generated from the dataset. If the original dataset has fewer cases, it will be cyclically padded; if more, only the first `request_count` cases will be used. Defaults to the length of the original dataset if not set.
- `sampling_mode` (str): Dataset sampling mode. Optional values: `shuffle` (shuffle and sample), `random` (random sample), `default` (no sampling).
- `output_config`: Controls model output settings for each request.
  - `method` (str): Type of data distribution. Optional values: `uniform` (uniform distribution), `percentage` (percentage distribution).
  - `params` (str): Parameters for data distribution settings.
    - `min_value` (str/int): Minimum length of generated data (valid only if `method: uniform`).
    - `max_value` (str/int): Maximum length of generated data (valid only if `method: uniform`).
    - `percentage_distribute` (list): Percentage distribution of output lengths (valid only if `method: percentage`). Format: 2D array, where the first element is the output length and the second is the percentage.


### Example Configurations

#### 1. Percentage Distribution
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

#### 2. Uniform Distribution
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


## Special Fields

### Maximum Output Length: `max_tokens`

In both `.csv` and `.jsonl` datasets, you can set the maximum output length **per request** by adding a `max_tokens` field (with a corresponding numeric value) to each object (in `.jsonl`) or each row (in `.csv`). This field is not yet applicable to performance stress testing scenarios.


#### Example 1: `.jsonl` Format
- For `mcq` type:
  ```json
  {"question": "165+833+650+615=", "A": "2258", "B": "2263", "C": "2281", "answer": "B", "max_tokens": 512}
  {"question": "368+959+918+653+978=", "A": "3876", "B": "3878", "C": "3880", "answer": "A", "max_tokens": 1024}
  {"question": "776+208+589+882+571+996+515+726=", "A": "5213", "B": "5263", "C": "5383", "answer": "B", "max_tokens": 2048}
  {"question": "803+862+815+100+409+758+262+169=", "A": "4098", "B": "4128", "C": "4178", "answer": "C", "max_tokens": 256}
  ```

- For `qa` type:
  ```json
  {"question": "165+833+650+615=", "answer": "2263", "max_tokens": 512}
  {"question": "368+959+918+653+978=", "answer": "3876", "max_tokens": 1024}
  {"question": "776+208+589+882+571+996+515+726=", "answer": "5263", "max_tokens": 2048}
  {"question": "803+862+815+100+409+758+262+169=", "answer": "4178", "max_tokens": 256}
  ```


#### Example 2: `.csv` Format
- For `mcq` type:
  ```bash
  question,A,B,C,answer,max_tokens
  127+545+588+620+556+199=,2632,2635,2645,B,512
  735+603+102+335+605=,2376,2380,2410,B,1024
  506+346+920+451+910+142+659+850=,4766,4774,4784,C,2048
  504+811+870+445=,2615,2630,2750,B,256
  ```

- For `qa` type:
  ```bash
  question,answer,max_tokens
  127+545+588+620+556+199=,2635,512
  735+603+102+335+605=,2380,1024
  506+346+920+451+910+142+659+850=,4784,2048
  504+811+870+445=,2630,256
  ```


### Example of Dataset and Corresponding Performance Evaluation Results
- Dataset Example:
  ![custom_dataset_example.img](../img/custom_datasets/custom_dataset_example.png)

- Performance Evaluation Result Example:
  ![custom_results_example.img](../img/custom_datasets/custom_results_example.png)