# Pure Model Accuracy Evaluation
Load models and datasets in a local environment, compare outputs with reference answers through a unified inference process, and evaluate the inherent accuracy of the model. Customize parameters such as batch size and sequence length, applicable to the **Huggingface Transformers** inference framework.

## Test Preparation
Before performing local model inference, the following conditions must be met:

- Available model weights: Ensure that the model weight files to be tested are already available locally. Open-source weights can be obtained from 🔗 [Hugging Face Community](https://huggingface.co/models).
- Dataset task preparation: Select a dataset from 📚 [Open-Source Datasets](../all_params/datasets.md#open-source-datasets), and choose the dataset task to execute in the "detailed introduction" document corresponding to the dataset. Prepare the dataset files according to the "detailed introduction" document of the selected dataset task. It is recommended to manually place the open-source dataset in the default directory `ais_bench/datasets/`, and the program will automatically load the dataset files during task execution.
- Model task preparation: Select the model task to execute from 📚 [Local Model Backend](../all_params/models.md#local-model-backend).

## Main Functions
The main functions in the pure model accuracy evaluation scenario are similar to those in the service-oriented accuracy evaluation scenario.

### Pure Model Multi-Task Evaluation
Refer to [Usage of Service-Oriented Accuracy Multi-Task Evaluation](accuracy_benchmark.md#multi-task-evaluation).

### Pure Model Multi-Task Parallel Evaluation
Refer to [Usage of Service-Oriented Accuracy Multi-Task Parallel Evaluation](accuracy_benchmark.md#multi-task-parallel-evaluation).
> ⚠️ Note: Multi-task parallel evaluation in pure model accuracy evaluation will occupy different GPU units. The number of GPU units required for parallel tasks should be less than or equal to the total number of available GPUs.

### Pure Model Resumption After Interruption
During the pure model accuracy evaluation, if the task is interrupted, you can use the `--reuse` parameter to specify the task timestamp directory to continue the unfinished inference task, realizing breakpoint resumption. This function does not require re-running all tasks, but only performs supplementary inference on the unfinished parts. For details on usage, refer to [Usage of Service-Oriented Accuracy Resumption After Interruption](accuracy_benchmark.md#resumption-after-interruption-&-retesting-of-failed-cases).
> ⚠️ Note: Currently, pure model accuracy evaluation does not support automatic retesting of failed cases.

### Pure Model Merged Sub-Dataset Inference
Refer to [Usage of Service-Oriented Accuracy Merged Sub-Dataset Inference](accuracy_benchmark.md#merging-sub-dataset-inference).

## Other Functions
### Re-Evaluation of Pure Model Inference Results
Refer to [Usage of Service-Oriented Accuracy Re-Evaluation of Inference Results](accuracy_benchmark.md#re-evaluation-of-inference-results).