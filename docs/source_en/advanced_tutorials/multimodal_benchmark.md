# Guide to MultiModal Evaluation

## Introduction to MultiModal
Multimodal models can handle data of two or more modalities such as text, images, audio, and video simultaneously. They encode data of different modalities into a unified semantic space, enabling cross-modal retrieval, understanding, reasoning, and generation, allowing machines to complete tasks by comprehensively processing "seeing, hearing, and speaking" information like humans. Currently, only the evaluation of multimodal understanding models is supported, while the evaluation of multimodal generation is not supported for the time being.


## Introduction to Evaluation Capabilities
The current performance and accuracy evaluation of multi-modal data is supported. The support levels for different model backends and datasets are as follows:

Supported Model backend
+ ✅Service-oriented online inference such as vLLM/vLLM Ascend/MindIE Service
+ ✅vLLM/vLLM Ascend Offline inference
+ ✅Pure model inference of transformers such as QwenVL

### Supported Datasets
+ ✅TextVQA（image+text）
+ ✅MMMU（image+text）
+ ✅MMMU_Pro（image+text）
+ ✅InfoVQA（image+text）
+ ✅DocVQA（image+text）
+ ✅MMStar（image+text）
+ ✅OmniDocBench（image+text）
+ ✅OcrBench-v2（image+text）
+ ✅VideoBench（video+text）
+ ✅Video-MME（video+text）
+ ✅VocalSound（audio+text）
+ ✅MM_Custom（image text audio text）


## Quick Start
### Multimodal input format
There are various formats for service-oriented multimodal data input. Taking image + text input as an example, it is as follows:
- Method 1: Local file format, default method
```
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
        "model": "qwen2_vl",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "What is the text in the illustrate?"},
                {"type": "image_url", "image_url": {"url": "file:///data/demo.jpg"}}
            ]}
        ]
    }'
```
- Method 2: Simplify the path format
```
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
        "model": "qwen2_vl",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "What is the text in the illustrate?"},
                {"type": "image_url", "image_url": "/data/demo.jpg"}
            ]}
        ]
    }'
```
- Method 3: url object format
```
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
        "model": "qwen2_vl",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "What is the text in the illustrate?"},
                {"type": "image_url", "image_url": {"url": "/data/demo.jpg"}}
            ]}
        ]
    }'
```
- Method 4: base64 input format
```
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
        "model": "qwen2_vl",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "What is the text in the illustrate?"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]}
        ]
    }'
```
### Usage Notes
The multimodal dataset configuration file defines the prompt format for input. Users can adjust it according to the format they need. Taking the textvqa image-text dataset as an example, the `{question}` will be automatically filled by the content of the dataset, and the input of images will default to the local file format
```
template=dict(
    round=[
        dict(role="HUMAN", prompt_mm={
            "text": {"type": "text", "text": "{question} Answer the question using a single word or phrase."},
            "image": {"type": "image_url", "image_url": {"url": "file://{image}"}},
        })
    ]
    )
```
In the base64 input scenario, the image input format needs to be converted to the base64 input format. Additionally, since the filled content is no longer the image path but the converted base64 value, it is also necessary to modify `image_type` to `image_base64`.
```
template=dict(
    round=[
        dict(role="HUMAN", prompt_mm={
            "text": {"type": "text", "text": "{question} Answer the question using a single word or phrase."},
            "image": {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}},
        })
    ]
    )
...
image_type="image_base64",
```
⚠️In addition, in the scenarios of vLLM offline inference and transformers pure model inference, the multimodal data input format needs to be configured as the simplified path format of Method 2.

### Command Explanation
Take the `textvqa` multimodal `vLLM` service-oriented performance evaluation scenario as an example:
```
ais_bench --models vllm_api_stream_chat --datasets textvqa_gen --debug -m perf
```
Where:
- `--models`: Specifies the model task, i.e., the `vllm_api_stream_chat` model task.
- `--datasets`: Specifies the dataset task, i.e., the `textvqa_gen` dataset task.
### Preparations Before Running the Command
- `--models`: To use the `vllm_api_stream_chat` model task, you need to prepare an inference service that supports the `/v1/chat/completions` sub-service. Refer to 🔗 [Start an OpenAI-Compatible Server with vLLM](https://docs.vllm.com/en/latest/getting_started/quickstart.html#openai-compatible-server) to launch the inference service.
- `--datasets`: To use the `textvqa_gen` dataset task, you need to prepare the TextVQA dataset by following the instructions in 🔗 [TextVQA Dataset](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/textvqa/README_en.md).
### Modifying Configuration Files for Corresponding Tasks
Each model task, dataset task, and result presentation task corresponds to a configuration file. These files must be modified before running the command. To find the paths of these configuration files, add `--search` to the original AISBench command. For example:
```shell
# Note: Adding "--mode perf" to the search command does not affect the search results
ais_bench --models vllm_api_stream_chat --datasets textvqa_gen --mode perf --search
```
> ⚠️ **Note**: Executing the command with `--search` will print the **absolute paths** of the configuration files corresponding to the tasks.

The query result will look like this:
```shell
06/28 11:52:25 - AISBench - INFO - Searching configs...
╒══════════════╤═══════════════════════════════════════╤════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╕
│ Task Type    │ Task Name                             │ Config File Path                                                                                                               │
╞══════════════╪═══════════════════════════════════════╪════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╡
│ --models     │ vllm_api_stream_chat                  │ /your_workspace/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py                                  │
├──────────────┼───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ --datasets   │ textvqa_gen                           │ /your_workspace/benchmark/ais_bench/benchmark/configs/datasets/textvqa/textvqa_gen.py                                          │
╘══════════════╧═══════════════════════════════════════╧════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╛

```

- In the Quick Start section, the dataset task configuration file `textvqa_gen.py` does not require any additional modifications. The content introduction of the dataset task configuration file can be referred to📚 [Configuring Open-Source Datasets](../base_tutorials/all_params/datasets.md#configuring-open-source-datasets)

- The model configuration file `vllm_api_stream_chat.py` contains the configuration content related to model operation and needs to be modified according to the actual situation. The contents that need to be modified in the quick start are marked with comments.
```python
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-chat-stream',
        path="",                       # Specify the absolute path to the model's serialized vocabulary file (usually the model weight folder path)
        model="",                      # Specify the name of the model loaded on the server (configure based on the actual model pulled by the vLLM inference service)
        stream=True,                   # stream infer mode
        request_rate=0,                # Request sending frequency: 1 request is sent to the server every 1/request_rate seconds. If < 0.1, all requests are sent at once.
        retry=2,
        api_key="",                    # Customize api_key, which is empty by default
        host_ip="localhost",           # Specify the IP address of the inference service
        host_port=8080,                # Specify the port of the inference service
        url="",                        # Customize url, which is empty by default
        max_out_len=512,               # Maximum number of tokens output by the inference service
        batch_size=1,                  # Maximum concurrency for sending requests
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.01,
            ignore_eos=False
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
```
### Executing the Command
After modifying the configuration files, run the following command to start the service-based performance evaluation (⚠️ It is recommended to add `--debug` for the first execution to print detailed logs, which helps troubleshoot errors during inference service requests):
```bash
# Add --debug to the command line
ais_bench --models vllm_api_stream_chat --datasets textvqa_gen -m perf --debug
```
### Viewing Performance Results
#### Example of Printed Performance Results

```bash
06/05 20:22:24 - AISBench - INFO - Performance Results of task: vllm-api-chat-stream/textvqadataset:

╒══════════════════════════╤═════════╤══════════════════╤══════════════════╤══════════════════╤══════════════════╤══════════════════╤══════════════════╤══════════════════╤══════╕
│ Performance Parameters   │ Stage   │ Average          │ Min              │ Max              │ Median           │ P75              │ P90              │ P99              │  N   │
╞══════════════════════════╪═════════╪══════════════════╪══════════════════╪══════════════════╪══════════════════╪══════════════════╪══════════════════╪══════════════════╪══════╡
│ E2EL                     │ total   │ 2048.2945  ms    │ 1729.7498 ms     │ 3450.96 ms       │ 2491.8789 ms     │ 2750.85 ms       │ 3184.9186 ms     │ 3424.4354 ms     │ 8    │
├──────────────────────────┼─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────┤
│ TTFT                     │ total   │ 50.332 ms        │ 50.6244 ms       │ 52.0585 ms       │ 50.3237 ms       │ 50.5872 ms       │ 50.7566 ms       │ 50.0551 ms       │ 8    │
├──────────────────────────┼─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────┤
│ TPOT                     │ total   │ 10.6965 ms       │ 10.061 ms        │ 10.8805 ms       │ 10.7495 ms       │ 10.7818 ms       │ 10.808 ms        │ 10.8582 ms       │ 8    │
├──────────────────────────┼─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────┤
│ ITL                      │ total   │ 10.6965 ms       │ 7.3583 ms        │ 13.7707 ms       │ 10.7513 ms       │ 10.8009 ms       │ 10.8358 ms       │ 10.9322 ms       │ 8    │
├──────────────────────────┼─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────┤
│ InputTokens              │ total   │ 1512.5           │ 1481.0           │ 1566.0           │ 1511.5           │ 1520.25          │ 1536.6           │ 1563.06          │ 8    │
├──────────────────────────┼─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────┤
│ OutputTokens             │ total   │ 287.375          │ 200.0            │ 407.0            │ 280.0            │ 322.75           │ 374.8            │ 403.78           │ 8    │
├──────────────────────────┼─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────┤
│ OutputTokenThroughput    │ total   │ 115.9216 token/s │ 107.6555 token/s │ 116.5352 token/s │ 117.6448 token/s │ 118.2426 token/s │ 118.3765 token/s │ 118.6388 token/s │ 8    │
╘══════════════════════════╧═════════╧══════════════════╧══════════════════╧══════════════════╧══════════════════╧══════════════════╧══════════════════╧══════════════════╧══════╛
╒══════════════════════════╤═════════╤════════════════════╕
│ Common Metric            │ Stage   │ Value              │
╞══════════════════════════╪═════════╪════════════════════╡
│ Benchmark Duration       │ total   │ 19897.8505 ms      │
├──────────────────────────┼─────────┼────────────────────┤
│ Total Requests           │ total   │ 8                  │
├──────────────────────────┼─────────┼────────────────────┤
│ Failed Requests          │ total   │ 0                  │
├──────────────────────────┼─────────┼────────────────────┤
│ Success Requests         │ total   │ 8                  │
├──────────────────────────┼─────────┼────────────────────┤
│ Concurrency              │ total   │ 0.9972             │
├──────────────────────────┼─────────┼────────────────────┤
│ Max Concurrency          │ total   │ 1                  │
├──────────────────────────┼─────────┼────────────────────┤
│ Request Throughput       │ total   │ 0.4021 req/s       │
├──────────────────────────┼─────────┼────────────────────┤
│ Total Input Tokens       │ total   │ 12100              │
├──────────────────────────┼─────────┼────────────────────┤
│ Prefill Token Throughput │ total   │ 17014.3123 token/s │
├──────────────────────────┼─────────┼────────────────────┤
│ Total generated tokens   │ total   │ 2299               │
├──────────────────────────┼─────────┼────────────────────┤
│ Input Token Throughput   │ total   │ 608.7438 token/s   │
├──────────────────────────┼─────────┼────────────────────┤
│ Output Token Throughput  │ total   │ 115.7835 token/s   │
├──────────────────────────┼─────────┼────────────────────┤
│ Total Token Throughput   │ total   │ 723.5273 token/s   │
╘══════════════════════════╧═════════╧════════════════════╛

06/05 20:22:24 - AISBench - INFO - Performance Result files locate in outputs/default/20250605_202220/performances/vllm-api-chat-stream.

```
💡 For the meaning of specific performance parameters, refer to 📚 [Explanation of Performance Evaluation Results](../base_tutorials/results_intro/performance_metric.md).

### Viewing Detailed Performance Data
After executing the AISBench command, detailed task execution data is saved to a default output path. The output path is indicated in the printed logs during runtime. For example:
```shell
06/28 15:13:26 - AISBench - INFO - Current exp folder: outputs/default/20250628_151326
```
This log indicates that detailed task data is stored in `outputs/default/20250628_151326` (relative to the directory where the command was executed).
```shell
20250628_151326           # Unique directory generated for each experiment based on timestamp
├── configs               # Auto-saved dump of all configuration files
├── logs                  # Runtime logs (no log files are saved if --debug is added to the command, as logs are printed directly to the terminal)
│   └── performance/      # Logs from the inference phase
└── performance           # Performance evaluation results
     └── vllm-api-chat-stream/          # Name of the "service-based model configuration" (corresponds to the `abbr` parameter in the model task configuration file)
          ├── textvqadataset.csv          # Per-request performance output (CSV), matching the "Performance Parameters" table in the printed results
          ├── textvqadataset.json         # End-to-end performance output (JSON), matching the "Common Metric" table in the printed results
          ├── textvqadataset_details.h5   # Full-granularity ITL data (Inter-Token Latency)
          ├── textvqadataset_details.json # Full detailed metrics
          └── textvqadataset_plot.html    # Request concurrency visualization report (HTML)
```

💡 The `textvqadataset_plot.html` report (a request concurrency visualization) is recommended to be opened in browsers such as Chrome or Edge. It shows the latency of each request and the number of concurrent service requests perceived by the client at each moment.

> ⚠️ **Note**: In multi-turn dialogue scenarios, the upper chart connects multiple requests in each dialogue group into a single line. Therefore, the **vertical axis represents the index of multi-turn dialogue data groups** (not concurrency).

  ![full_plot_example.img](../img/request_concurrency/full_plot_example.png)

For instructions on how to view the charts in the specific HTML file, please refer to 📚 [Guide to Using Performance Test Visualization Concurrent Charts](../base_tutorials/results_intro/performance_visualization.md)