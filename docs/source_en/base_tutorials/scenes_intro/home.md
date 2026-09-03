# Introduction to Evaluation Scenarios
### Accuracy Evaluation
#### Service-Oriented Accuracy Evaluation
- **Function Description**: Evaluate the prediction accuracy of a model deployed as a service on specific datasets. Currently supports accuracy evaluation based on generative and PPL (Perplexity-based) modes.

- **Requirements**: The model has been deployed, and its actual service capabilities need to be tested.

- **Model Tasks and Dataset Tasks Supported by This Scenario**:
  - **Model Tasks**: 📚 [Service-Oriented Inference Backend](../all_params/models.md#service-oriented-inference-backend)
  - **Dataset Tasks**: 📚 [Open-Source Datasets](../../get_started/datasets.md#open-source-datasets) and 📚 [Custom Datasets](../../get_started/datasets.md#custom-datasets)

- **Constraint**: Currently, PPL mode accuracy evaluation tasks only support `vllm_api_general` and `vllm_api_general_chat` model configurations; other configurations are not supported.

After selecting the **model task** and **dataset task** according to your usage needs, refer to the document for detailed usage of this scenario: 📚 [Service-Oriented Accuracy Evaluation Guide](accuracy_benchmark.md)

#### Pure Model Accuracy Evaluation
- **Function Description**: Evaluate the accuracy of locally loaded models (non-service-oriented) on different datasets.

- **Requirements**: Offline model weights and a deployment environment.

- **Supported Items**:
  - **Model Tasks**: 📚 [Local Model Backend](../all_params/models.md#local-model-backend)
  - **Dataset Tasks**: 📚 [Open-Source Datasets](../../get_started/datasets.md#open-source-datasets) and 📚 [Custom Datasets](../../get_started/datasets.md#custom-datasets)

- **Constraint**: PPL mode evaluation tasks are not supported.

After selecting the **model task** and **dataset task** according to your usage needs, refer to the document for detailed usage of this scenario: 📚 [Pure Model Accuracy Evaluation Guide](accuracy_benchmark_local.md)

#### Harbor-Based Agent Evaluation
- **Function Description**: Run agent evaluation by launching [Harbor](https://github.com/harbor-framework/harbor) via `--mode agent`, executing and monitoring each case in real time, and outputting a single table + CSV summary. AISBench supports all Harbor-defined Agents; parameters with the same meaning across different Agents are adapted automatically by `AgentParamAdapter` (unified `--api-base` / `--agent-api-key`), and custom `module.path:ClassName` Agents are also supported.

- **Requirements**: A model inference service that follows the **OpenAI chat/completions API** and supports **tool call**; a Python 3.12 environment; Docker / execution environment prepared as required by Harbor; install the standalone dependency set `requirements/agent.txt` (version-conflict/compile warnings during install do not affect usage and can be ignored).

- **Supported Items**:
  - **Agent Tasks**: All built-in Harbor `AgentName` Agents or custom `module.path:ClassName` (`-a/--agent`)
  - **Dataset Tasks**: The three sources Harbor can resolve — a local dataset directory / a single task directory (`-p/--agent-dataset-path`), Registry `name@version`, and Package `org/name@ref` (`-d/--dataset`)
  - Unified semantic parameters (`--api-base` / `--agent-api-key` / `--model`), parameter override (`--ak` / `--ae`), real-time monitoring HTTP service (`--monitor-port`), resume (`--reuse`), and automatic retry of exception cases (`--reuse` + `--purge-exception-cases`)

- **Constraint**: `--purge-exception-cases` takes effect only when `--reuse` is set; agent evaluation runs on a standalone, minimal dependency set and does not depend on AISBench's native inference/accuracy chain.

After selecting the **agent task** and **dataset task** according to your usage needs, refer to the document for detailed usage of this scenario: 📚 [Harbor-Based Agent Evaluation Guide](agent_benchmark.md)

### Performance Evaluation
#### Service-Oriented Performance Evaluation
- **Function Description**: Evaluate the operational efficiency (throughput, latency) of a service model in a real deployment environment.

- **Requirements**: The model inference service must support access via a **streaming interface**.

- **Supported Items**:
  - **Model Tasks**: Streaming interface types in 📚 [Service-Oriented Inference Backend](../all_params/models.md#service-oriented-inference-backend)
  - **Dataset Tasks**: All data types in 📚 [Supported Dataset Types](../../get_started/datasets.md#supported-dataset-types)

- **Note**: The cache size occupied by performance evaluation is proportional to the context length of requests and the number of requests, so it usually increases positively with the evaluation duration.

- **Constraint**: PPL mode evaluation tasks are not supported.

After selecting the **model task** and **dataset task** according to your usage needs, refer to the document for detailed usage of this scenario: 📚 [Service-Oriented Performance Evaluation Guide](performance_benchmark.md)