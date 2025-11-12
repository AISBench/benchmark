# AISBench FAQ (Frequently Asked Questions)

## 1. Common Installation Errors

### 1.1 ‘torch.library’ has no attribute ‘register_fake’
![Image Description](https://foruda.gitee.com/images/1752115913457665110/c75f79c4_15797231.png "Screenshot")

**Root Cause:**
Mismatched versions of PyTorch (torch) and TorchVision (torchvision) in the AISBench environment.

**Recommended Solutions:**
Refer to [Previous PyTorch Versions](https://pytorch.org/get-started/previous-versions/) to install compatible versions of torch and torchvision.


### 1.2 ImportError: Failed to import required modules
![Image Description](https://foruda.gitee.com/images/1752115955182360975/cec5312d_15797231.png "Screenshot")

**Root Cause:**
Missing dependency packages, which may prevent some mathematical computation or multimodal accuracy calculation functions from working properly.

**Recommended Solutions:**
Execute the following commands to install the extended dependency packages for AISBench:
```shell
pip3 install -r requirements/api.txt
pip3 install -r requirements/extra.txt
```
---

## 2. Parameter Configuration Issues

### 2.1 Given the evaluation command, which configuration files do I need to modify?

**Root Cause:**
AISBench provides a quick configuration file search function.

**Recommended Solutions:**
Simply add the `--search` parameter to the original evaluation command. The terminal will print the paths of the configuration files involved in the task, and you can modify them directly.
![Image Description](https://foruda.gitee.com/images/1752204051848035094/9987953a_15797231.png "Screenshot")


### 2.2 Which parameters in the configuration files should be modified?

**Root Cause:**
AISBench provides default configuration files for each dataset, and direct modification is not recommended. If you need to customize a template or specify a dataset path, it is advisable to back up the original configuration file first, then modify the `template` field and `path` field in the dataset configuration:
![Image Description](https://foruda.gitee.com/images/1752204060095585759/8e45aa17_15797231.png "Screenshot")

The following parameters in the model configuration **must** be modified:
![Image Description](https://foruda.gitee.com/images/1752204068024744757/2b26c904_15797231.png "Screenshot")

**Recommended Solutions:**
If you need to configure parameters such as request rate, concurrency speed, and maximum model output length, refer to:
[Model Configuration Instructions](../base_tutorials/all_params/models.md)


### 2.3 Error: Please pass the argument ‘trust_remote_code=True’ to allow custom code to run
![Image Description](https://foruda.gitee.com/images/1752204075220061499/7c5c2c79_15797231.png "Screenshot")

**Root Cause:**
New models or community models have not yet been integrated into the official Transformers library. If the model repository contains Python scripts for custom architectures or tokenizers, `trust_remote_code=True` must be set; otherwise, model loading will fail.

**Recommended Solutions:**
Configure `'trust_remote_code=True'` in the model configuration file to inform Transformers that it is safe to execute the downloaded .py files.
![Image Description](https://foruda.gitee.com/images/1752204081509799033/3e2b778c_15797231.png "Screenshot")


### 2.4 Error: FileExistsError: Dataset path: **** is not exist!
![Image Description](https://foruda.gitee.com/images/1752204088304680700/6777b51c_15797231.png "Screenshot")

**Root Cause:**
Incorrect dataset configuration, causing AISBench to fail to locate the dataset path.

**Recommended Solutions:**
1. Refer to the [Dataset Preparation Guide](../base_tutorials/all_params/datasets.md) to complete dataset preparation.
2. It is recommended to place open-source datasets in `{tool_root_path}/ais_bench/datasets`.
3. For custom dataset paths, modify the configuration file specified by `--datasets`, and update the `path` field to the actual dataset path (ensure the data format in the actual path matches the requirements).
![Image Description](https://foruda.gitee.com/images/1752204096428670468/faa01659_15797231.png "Screenshot")

---

## 3. Service-Side Return Errors

### 3.1 Request failed: HTTP status 500. Server response: **timeout**

**Root Cause:**
Inference timeout on the server side.

**Recommended Solutions:**
- Increase the timeout duration.
- Reduce the model’s maximum output length via the `max_out_len` parameter in the model configuration to lower inference time.
- Decrease the model’s maximum concurrency via the `batch_size` parameter in the model configuration to reduce parallel resource usage on the server and improve request inference efficiency.
- Upgrade hardware configurations.


### 3.2 Exceeded maximum retry attempts (2) or Connection refused

**Root Cause:**
The client failed to establish a connection with the server. This is most likely an issue with the service port configuration or URL configuration.

**Recommended Solutions:**
1. Verify if the server is accessible normally using the command `curl http:{url}:{port}/v1/models`. For example, the `id` field in the 'data' dictionary below indicates the model name loaded on the server, confirming that the service backend can respond normally:
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
      "owned_by": "organization-owner"
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
2. Check if the server is started and listening.
3. Review firewall or security group configurations.


### 3.3 HTTP error during stream response processing: Response ended prematurely

**Root Cause:**
The streaming response was interrupted unexpectedly.

**Recommended Solutions:**
Check the server logs to identify the cause of the interruption.


### 3.4 [AisBenchClientException] Error processing stream response: [StreamResponseError] Expecting value: line 1 column 65533 (char 65532)!

**Root Cause:**
The content of the streaming response exceeds the default chunk cache size, leading to parsing failure. This usually occurs when non-standard endpoints include additional server-side return data.

**Recommended Solutions:**
Increase `MAX_CHUNK_SIZE` in `ais_bench/benchmark/global_consts.py` to expand the chunk cache space.

---

## 4. Common Accuracy Evaluation Issues

### 4.1 How to select model configurations for accuracy evaluation?

**Root Cause:**
Unfamiliarity with the characteristics and applicable scenarios of different model configurations.

**Recommended Solutions:**
1. Read the official documentation of the service framework to confirm the supported service backends. For example, vLLM and SGLang only support OpenAI endpoints, so you can only select model configurations corresponding to `v1/chat/completions` and `v1/completions` for evaluation.
2. Refer to [Model Configuration Instructions](../base_tutorials/all_params/models.md) to select model configurations compatible with the service backend.
3. For chat-type APIs, structured information containing a `system` role in the `chat_template` will be added to the original prompt based on the model configuration. For example:
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
Other model configurations directly use the prompt as input content. Therefore, accuracy results may vary—select the appropriate model configuration based on actual needs during testing.


### 4.2 How to view the model's output content during accuracy evaluation?

**Root Cause:**
AISBench saves the model’s accuracy evaluation results but does not print them directly.

**Recommended Solutions:**
During inference, the model’s inference results are cached in the `predictions` subdirectory of the working directory. For example: `outputs/default/20250710_164659/predictions/vllm-api-stream-chat/gsm8k.json`.

The data format is as follows:
```json
{
    "0": { # Data ID
        "origin_prompt": "What is 2 + 2?", # Original input of the data
        "prediction": "4", # Model output
        "gold": "4" # Correct answer of the data
    },
    ...
}
```


### 4.3 How to view the correct/incorrect results for each question?

**Root Cause:**
By default, AISBench does not save correct/incorrect information for each case during accuracy calculation. However, it provides the `--dump-eval-details` parameter to enable recording of detailed evaluation results.

**Recommended Solutions:**
1. Add the `--dump-eval-details` parameter to the evaluation command.
2. If model inference results are already saved in the working directory, use it with the `--reuse` parameter to regenerate files containing detailed evaluation results. For details, refer to: [Eval Mode](../base_tutorials/all_params/mode.md#eval-mode).
3. During inference, detailed evaluation results are cached in the `results` subdirectory of the working directory. For example: `outputs/default/20250710_164659/results/vllm-api-stream-chat/gsm8k.json`.

The data format is as follows:
```json
"details": {
    "0": { # Data ID
        "prompt": "What is 2 + 2?", # Original input of the data
        "origin_prediction": "The answer is 4.", # Raw model output
        "predictions": "4", # Answer extracted from the model output
        "references": "4", # Correct answer of the data
        "correct": true # Whether the answer is correct
    },
    ...
}
```


### 4.4 The model output contains the correct answer, but the accuracy calculation result is abnormal

**Root Cause:**
AISBench extracts correct answers from model responses based on specific matching rules. In some cases, humans can identify the correct answer from the model output, but the output may not meet AISBench’s matching rules, resulting in empty or abnormal accuracy calculation results.

**Recommended Solutions:**
Refer to [How to view the correct/incorrect results for each question](#43-how-to-view-the-correctincorrect-results-for-each-question) to enable the `--dump-eval-details` parameter. Check the detailed evaluation results of the model output to confirm whether the abnormal accuracy calculation is caused by mismatched rules.


### 4.5 The difference between naive_average and weighted_average accuracy results in CEval and MMLU

**Root Cause:**
Accuracy results for datasets with multiple subcategories (such as CEval and MMLU) include accuracy for each subcategory, as well as overall average accuracy. The overall average accuracy is calculated in two ways: `naive_average` (simple average of accuracy for each subcategory) and `weighted_average` (weighted average of accuracy for each subcategory based on the data volume of each subcategory).

**Recommended Solutions:**
1. Accuracy results in papers are usually `weighted_average` results. If you need to reproduce the accuracy results in papers, refer to the `weighted_average` results.
2. If you need to analyze the accuracy results of each subcategory, refer to the `naive_average` results.


### 4.6 Failed to reproduce paper accuracy on DS (refer to AISBench Wiki)

**Root Cause:**
1. The actual environment configuration used is inconsistent with that in the AISBench Wiki.
2. The dataset has undergone additional processing and differs from the standard original dataset.

**Recommended Solutions:**
1. Compare the configuration parameters in `--models` with those in the AISBench Wiki, including the maximum output length `max_out_len` (to prevent truncation) and post-processing parameters `generation_kwargs`.
2. Refer to the AISBench community’s [Dataset Preparation Guide](../base_tutorials/all_params/datasets.md) to complete dataset preparation.
3. Submit the discrepancy results as an Issue to seek assistance.


### 4.7 FileNotFoundError: [Errno 2] No such file or directory

**Root Cause:**
The dataset was not configured according to the documentation instructions.

**Recommended Solutions:**
Refer to the AISBench community’s [Dataset Preparation Guide](../base_tutorials/all_params/datasets.md#) to complete dataset preparation.

---

## 5. Common Performance Evaluation Issues

### 5.1 How to select model configurations for performance evaluation?

**Root Cause:**
Unfamiliarity with the characteristics and applicable scenarios of different model configurations.

**Recommended Solutions:**
1. Read the official documentation of the service framework to confirm the supported service backends. For example, vLLM and SGLang only support OpenAI endpoints, so you can only select model configurations corresponding to `v1/chat/completions` and `v1/completions` for evaluation.
2. Refer to [Model Configuration Instructions](../base_tutorials/all_params/models.md) to select model configurations compatible with the service backend. Note that only streaming interfaces support performance evaluation—model configurations must contain the `stream` keyword.
3. For chat-type APIs, structured information containing a `system` role in the `chat_template` will be added to the original prompt based on the model configuration. For example:
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
Other model configurations directly use the prompt as input content. Therefore, the actual input token length may vary, leading to differences in performance results.
4. Model configurations corresponding to `v1/chat/completions` and `v1/completions` extract the number of generated tokens from the server’s return information, which is consistent with the total number of tokens actually generated by the server. Other model configurations encode the actual output string into token IDs to obtain the output token length, which may differ from the total number of tokens actually generated by the server. If you need to ensure the complete accuracy of the output token length, it is recommended to select model configurations corresponding to `v1/chat/completions` and `v1/completions` for performance evaluation.


### 5.2 What is Steady State?

**Root Cause:**
Steady-state performance testing (hereinafter referred to as "steady-state testing") is designed to simulate real-world business scenarios of inference services and test the performance of the inference service when it is in a stable state. A "steady state" refers to a state where the inference service can handle concurrent requests simultaneously and remain stable when the number of concurrent requests reaches its maximum.

**Recommended Solutions:**
Refer to [Service-Side Steady-State Performance Testing](../advanced_tutorials/stable_stage.md)


### 5.3 What is Stress Testing?

**Root Cause:**
AISBench’s service-side performance stress testing is intended to simulate real-world business scenarios of inference services and test the performance of the service under maximum concurrent load within a specific time period.

**Recommended Solutions:**
Refer to [Service-Side Stress Testing](../advanced_tutorials/stable_stage.md#quick-start-for-stress-testing)


### 5.4 Mismatch Between InputTokens and Synthetic Dataset Input

**Root Cause:**
1. A chat-type model configuration (e.g., `v1/chat/completions`) is used. This configuration concatenates the input prompt with the `chat_template`, resulting in a mismatch between the actual number of input tokens and the input of the synthetic dataset.
2. Synthetic datasets fall into two categories: `string` and `tokenid`, and their principles for constructing synthetic datasets differ:

- **`string`**: Treats `A ` (letter "A" followed by a space) as one token. Input strings are constructed by concatenating multiple `A ` segments, so the final number of input tokens matches the length of the actual input string. Example code is as follows:
```python
from transformers import AutoTokenizer

# Load the specified tokenizer model (Qwen2-7B-Instruct)
tokenizer_model = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    trust_remote_code=True,
    use_fast=True,
    local_files_only=False
)

token_num = 10  # Set the number of tokens to generate
text = "A " * token_num  # Construct a string containing 10 "A " segments
text = text.strip()  # Remove the trailing space

# Encode the text into a list of token IDs
new_token_ids = tokenizer_model.encode(text)

# Decode the list of token IDs back to text
new_text = tokenizer_model.decode(new_token_ids)

print(text == new_text)                      # Output: True
print(f"Token Count: {len(new_token_ids)}")  # Output: Token Count: 10
```

- **`tokenid`**: Generates a list of `token_num` token IDs first, then generates the corresponding input string based on this list of token IDs. The process of converting token IDs to strings and then back to token IDs is irreversible, which may cause the actual number of tokens to increase or decrease, leading to a mismatch with the configured number of input tokens. Example code is as follows:
```python
from transformers import AutoTokenizer

# Load the specified tokenizer model (Qwen2-7B-Instruct)
tokenizer_model = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    trust_remote_code=True,
    use_fast=True,
    local_files_only=False
)

original_token_ids = [i for i in range(100)]  # Original list of token IDs, length = 100

# Decode the list of token IDs into text
text = tokenizer_model.decode(original_token_ids)

# Re-encode the decoded text into a list of token IDs
new_token_ids = tokenizer_model.encode(text)

print(f"New token ids nums: {len(new_token_ids)}")  # Output: New token ids nums: 35
```

**Recommended Solutions:**
1. Confirm whether the model configuration is a chat-type configuration. If yes, note that the actual number of input tokens may not match the input of the synthetic dataset.
2. Confirm the type of the synthetic dataset and select an appropriate model configuration.


### 5.5 Mismatch Between OutputTokens and Maximum Output Length

**Root Cause:**
1. The service-side model has a maximum context limit, which may cause the actual number of generated tokens to be less than the maximum output length.
2. The `ignore_eos` parameter is not configured in the model configuration’s `generation_kwargs`.
3. The model configuration does not support the `ignore_eos` parameter, causing the model to stop generation early when it encounters the `eos_token` (end-of-sequence token).

**Recommended Solutions:**
1. Confirm whether the service-side model has a maximum context limit. For service backends that support OpenAI endpoints, you can construct a request based on the input length and send it using the command below to check if the `completion_tokens` in the response reaches `max_tokens`:
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
2. Use the `--search` parameter to locate the task’s configuration file, and check if the `ignore_eos` parameter is correctly configured in `generation_kwargs`.
3. Refer to the official documentation of the service backend to confirm whether the `ignore_eos` parameter is supported.


### 5.6 Both TTFT and TPOT Decrease—Why Does Total Throughput Drop Instead?

**Root Cause:**
Single-request latency and concurrency effects interact with each other. With fixed hardware resources, a reduction in actual concurrency reduces resource competition between parallel tasks, lowering the latency of individual requests. However, due to the decreased parallel efficiency, the total throughput may drop.

**Recommended Solutions:**
After AISBench completes performance testing, it generates real-time concurrent visualization charts of the testing process. Refer to [Guide to Using Performance Test Visualization Concurrent Charts](../base_tutorials/results_intro/performance_visualization.md) to analyze the concurrency efficiency and request execution of the test task.


### 5.7 Definitions and Formulas of Various Metrics

**Root Cause:**
The definition of AISBench’s evaluation metrics is consistent with industry standards. The definitions of common metrics are shown in the figure below:
![Image Description](https://foruda.gitee.com/images/1752204138375003502/8c9de2a8_15797231.png "Screenshot")

**Recommended Solutions:**
For more detailed explanations, refer to: [Performance Evaluation Result Description](../base_tutorials/results_intro/performance_metric.md)


### 5.8 ValueError: Tokenizer path '' does not exist

**Root Cause:**
The tokenizer path is not specified in the model configuration, or the directory pointed to by the path does not exist.

**Recommended Solutions:**
1. Use the `--search` parameter to query the configuration files involved in the task.
2. In the relevant configuration file, set the `path` field to the local vocabulary file path (usually the directory where the model weights are stored).
3. Confirm that the path is filled in correctly and that the file actually exists.
![Image Description](https://foruda.gitee.com/images/1752204145754826215/b7fe93f0_15797231.png "Screenshot")


### 5.9 Performance Differences Between AISBench and vllm_benchmark

**Root Cause:**
1. **Input Misalignment**: Both AISBench and vllm_benchmark support synthetic dataset testing. AISBench uses the `tokenid` method to generate synthetic datasets, which is similar to the "random" dataset construction method in vllm_benchmark. However, due to the randomness of converting token IDs to strings, inputs cannot be completely aligned.
2. **Mismatched Endpoints for Testing**: vllm_benchmark specifies endpoints via the `--backend` parameter, while AISBench specifies endpoints via model configuration files. The corresponding relationship between the two is shown in the table below:

| vllm_benchmark `--backend` Parameter | AISBench `--models` Configuration File Example | Description |
|:--------------------------------------|:-----------------------------------------------|:-------------|
| vllm/lmdeploy/openai/scalellm/sglang  | vllm_api_general_stream                        | `v1/completions` |
| openai-chat                           | vllm_api_stream_chat                           | `v1/chat/completions` |
| tgi                                   | tgi_stream_api_general                         | `generate_stream` |
| tensorrt-llm                          | triton_stream_api_general                      | v2/models/{model_name}/generate_stream |

> The actual names of the model configuration files depend on the AISBench version and are usually located in the `ais_bench/benchmark/configs/models/` directory.

**Recommended Solutions:**
1. Both tools support custom dataset testing. It is recommended to use the same dataset construction method to ensure input alignment.
2. Confirm that the endpoints accessed are consistent to ensure alignment of the test environment.
3. For performance testing, it is recommended to disable post-processing by setting `temperature = 0` or `temperature = 0.01` (some endpoints do not support `temperature = 0`) to reduce the impact of post-processing on performance results.


### 5.10 Significant Gap in First-Token Latency Between AISBench and Benchmark When Request Rate Is Enabled

**Root Cause:**
This is caused by differences in the request capture mechanisms of the two tools. AISBench captures multiple requests at the start of the sending phase, leading to a backlog of requests waiting for processing. Subsequent requests in each phase are thus delayed by a certain period.


### 5.11 Actual Concurrency in Steady-State Testing Exceeds Maximum Concurrency


## 6. Others

### 6.1 No Logs or Progress Bar Printed During Task Execution

**Root Cause:**
When AISBench executes parallel tasks, logs printed by different processes may become garbled. Therefore, by default, logs are saved in the `logs` directory of the working directory (e.g., `outputs/default/20250710_164010/logs`).

**Recommended Solutions:**
1. Use the `tail -f` command to view log content. Example:
```shell
tail -f ./outputs/default/20250710_164010/logs/infer/vllm-api-stream-chat/gsm8k.out
```
2. Add the `--debug` parameter to enable debug mode, which executes tasks sequentially and prints logs directly to the terminal.


### 6.2 Task OpenICLInfer [***] Fail, See ***.out

**Root Cause:**
The inference task failed. Since debug mode is not enabled, the system prompts you to view the subtask log file for details.

**Recommended Solutions:**
1. Use the `vim` or `cat` command to view the subtask log file. Example:
```shell
vim outputs/default/20250711_104313/logs/infer/vllm-api-stream-chat/gsm8k.out
```


### 6.3 Visualization Files Cannot Be Opened Normally in Browsers

When a visualization file remains blank for a long time after being opened in a browser, press **F12** to open the browser’s Developer Tools, and check the "Console" tab for red error messages.

- **Error 1: "File protocol not supported"**
  - **Cause 1**: Blocked by the browser’s CORS (Cross-Origin Resource Sharing) security policy, which treats the request as a cross-origin request.
    - **Solution 1**: Enable local file access: Search for the "Allow access to local files" switch in the browser settings and enable it.
    - **Solution 2**: Modify shortcut properties: Add `--allow-file-access-from-files --user-data-dir="Path to the visualization HTML file" --disable-web-security` to the "Target" field of the browser shortcut. Then click the shortcut to open a new window, which will allow access to local files (depending on the browser, you may need to restart your computer or close other browser windows first).
    - **Solution 3**: Check extensions and cache: Disable browser extensions that may interfere with local file access, and clear the browser cache.
    - **Solution 4**: Start a simple local server via Python and access the file through `http://localhost`.

- **Error 2: "plotly.min.js not found"**
  - **Cause 1**: The server’s security settings are too strict, preventing the required `plotly.min.js` file from being loaded via CDN.
    - **Solution 1**: Remove network firewall-related settings.
    - **Solution 2**: Load locally:
      - Step 1: Go to the [Download section of the Plotly official website](https://www.plotly.com/javascript/getting-started/) or the [bootcdn.plotly website (select versions above 3.0)](https://www.bootcdn.cn/plotly.js/) to download the file, and save it in the same directory as the visualization HTML file.
      - Step 2: Open the HTML file in a text editor, find the following code at the beginning of the file:
          ```html
          <script src="https://cdn.plot.ly/plotly-3.1.0.min.js" charset="utf-8"></script>
          ```
          Modify it to:
          ```html
          <script src="Path to the downloaded file" charset="utf-8"></script>
          ```
          Save and close the file.
      - Step 3: Reopen the file in the browser. (If the page is still blank, refer to the solutions for "Error 1".)
  - **Cause 2**: No network connection, preventing the required `plotly.min.js` file from being loaded.
    - **Solution 1**: Refer to "Solution 2" under "Cause 1"—download the file in an environment with network access and transfer it to the environment where the visualization file is located.