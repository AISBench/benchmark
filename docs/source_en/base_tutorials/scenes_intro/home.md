# Introduction to Evaluation Scenarios
### Accuracy Evaluation
#### Service-Oriented Accuracy Evaluation
- **Function Description**: Evaluate the prediction accuracy of a model deployed as a service on specific datasets. Currently supports accuracy evaluation based on generative and PPL (Perplexity-based) modes.

- **Requirements**: The model has been deployed, and its actual service capabilities need to be tested.

- **Model Tasks and Dataset Tasks Supported by This Scenario**:
  - **Model Tasks**: 📚 [Service-Oriented Inference Backend](../all_params/models.md#service-oriented-inference-backend)
  - **Dataset Tasks**: 📚 [Open-Source Datasets](../all_params/datasets.md#open-source-datasets) and 📚 [Custom Datasets](../all_params/datasets.md#custom-datasets)

After selecting the **model task** and **dataset task** according to your usage needs, refer to the document for detailed usage of this scenario: 📚 [Service-Oriented Accuracy Evaluation Guide](accuracy_benchmark.md)

#### Pure Model Accuracy Evaluation
- **Function Description**: Evaluate the accuracy of locally loaded models (non-service-oriented) on different datasets.

- **Requirements**: Offline model weights and a deployment environment.

- **Supported Items**:
  - **Model Tasks**: 📚 [Local Model Backend](../all_params/models.md#local-model-backend)
  - **Dataset Tasks**: 📚 [Open-Source Datasets](../all_params/datasets.md#open-source-datasets) and 📚 [Custom Datasets](../all_params/datasets.md#custom-datasets)

After selecting the **model task** and **dataset task** according to your usage needs, refer to the document for detailed usage of this scenario: 📚 [Pure Model Accuracy Evaluation Guide](accuracy_benchmark_local.md)

### Performance Evaluation
#### Service-Oriented Performance Evaluation
- **Function Description**: Evaluate the operational efficiency (throughput, latency) of a service model in a real deployment environment.

- **Requirements**: The model inference service must support access via a **streaming interface**.

- **Supported Items**:
  - **Model Tasks**: Streaming interface types in 📚 [Service-Oriented Inference Backend](../all_params/models.md#service-oriented-inference-backend)
  - **Dataset Tasks**: All data types in 📚 [Supported Dataset Types](../all_params/datasets.md#supported-dataset-types)

- **Note**: The cache size occupied by performance evaluation is proportional to the context length of requests and the number of requests, so it usually increases positively with the evaluation duration.

After selecting the **model task** and **dataset task** according to your usage needs, refer to the document for detailed usage of this scenario: 📚 [Service-Oriented Performance Evaluation Guide](performance_benchmark.md)