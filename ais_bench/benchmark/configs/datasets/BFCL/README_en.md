# BFCL (Berkeley Function Calling Leaderboard) V3
[中文](README.md) | English
## Dataset Introduction
The **Berkeley Function Calling Leaderboard (BFCL)** is a comprehensive, executable evaluation benchmark specifically designed to assess the function-calling capabilities of Large Language Models (LLMs).

### Key Features
- **First Comprehensive Evaluation Benchmark**: BFCL is the first full-fledged evaluation platform dedicated to testing the function-calling capabilities of LLMs.
- **Executability Verification**: Unlike previous evaluation methods, BFCL not only assesses the model’s ability to generate function calls but also executes these calls to verify their correctness.
- **Diverse Function-Calling Formats**: The dataset covers various forms of function calls, enabling comprehensive testing of model performance across different scenarios.
- **Rich Application Scenarios**: BFCL includes multiple usage scenarios to ensure the comprehensiveness and practicality of the evaluation.

> 🔗 **Official Homepage**: [https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html)

## Dataset Deployment
The BFCL dataset is integrated as a Python dependency package. Data files are included in the `bfcl-eval` dependency package—once the dependency is installed, the dataset can be used directly without additional downloads or network connections.

### Environment Requirements
- **bfcl-eval** dependency package (contains the complete dataset)

### Installation Steps
```bash
pip3 install -r requirements/datasets/bfcl_dependencies.txt --no-deps
```

✅ **After installation, the BFCL dataset is installed locally along with the dependency package and can be used normally in an offline environment.**

### ⚠️ Usage Notes

> - **Model Restriction**: The BFCL dataset only supports evaluation using the API model [`vllm_api_function_call_chat`](../../models/vllm_api/vllm_api_function_call_chat.py).
> - **Dataset Merging**: Each sub-category dataset of BFCL uses a different evaluation method. Therefore, the `--merge-ds` parameter cannot be used to merge sub-datasets for testing.

### Usage Examples

#### 1. Model Configuration
Configure the [`vllm_api_function_call_chat`](../../models/vllm_api/vllm_api_function_call_chat.py) model:

```python
from ais_bench.benchmark.models import VLLMFunctionCallAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMFunctionCallAPIChat,
        abbr="vllm-api-function-call-chat",
        path="",
        model="",
        request_rate=0,
        retry=2,
        host_ip="localhost",        # Inference service IP address
        host_port=8080,             # Inference service port number
        max_out_len=10240,          # Maximum output token length
        batch_size=100,             # Concurrent request batch size
        returns_tool_calls=True,    # Function call information extraction method (set to True if tool_calls field is supported)
        trust_remote_code=False,
        generation_kwargs=dict(     # Generation parameter configuration
            temperature=0.01,       # Low temperature is recommended to reduce accuracy fluctuations
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
```

#### 2. Run Evaluation
Execute the AISBench evaluation command:

```bash
# Basic command format
ais_bench --models vllm_api_function_call_chat --datasets {BFCL dataset configuration}

# Specific example: Test simple function calls
ais_bench --models vllm_api_function_call_chat --datasets BFCL_gen_simple
```

#### 3. Result Example
Accuracy results displayed after evaluation completion:

```bash
dataset          version    metric    mode      vllm-api-function-call-chat
---------       ---------  --------  ------  --------------------------------
BFCL-v3-simple   542b40    accuracy   gen            0.96 (385/400)
```

## Dataset Classification
The `--datasets` parameter allows flexible selection of test scope, supporting three granularities of test configurations as follows:

### Individual Test Categories

- [`BFCL_gen_simple`](./BFCL_gen_simple.py) - Simple Python function calls
- [`BFCL_gen_java`](./BFCL_gen_java.py) - Simple Java function calls
- [`BFCL_gen_javascript`](./BFCL_gen_javascript.py) - Simple JavaScript function calls
- [`BFCL_gen_parallel`](./BFCL_gen_parallel.py) - Parallel function calls
- [`BFCL_gen_multiple`](./BFCL_gen_multiple.py) - Sequential calls of multiple functions
- [`BFCL_gen_parallel_multiple`](./BFCL_gen_parallel_multiple.py) - Mixed parallel and sequential calls
- [`BFCL_gen_irrelevance`](./BFCL_gen_irrelevance.py) - Function calls with irrelevant documents
- [`BFCL_gen_live_simple`](./BFCL_gen_live_simple.py) - Live: Simple function calls
- [`BFCL_gen_live_multiple`](./BFCL_gen_live_multiple.py) - Live: Sequential calls of multiple functions
- [`BFCL_gen_live_parallel`](./BFCL_gen_live_parallel.py) - Live: Parallel function calls
- [`BFCL_gen_live_parallel_multiple`](./BFCL_gen_live_parallel_multiple.py) - Live: Mixed parallel and sequential calls
- [`BFCL_gen_live_irrelevance`](./BFCL_gen_live_irrelevance.py) - Live: Function calls with irrelevant documents
- [`BFCL_gen_live_relevance`](./BFCL_gen_live_relevance.py) - Live: Function calls with relevant documents
- [`BFCL_gen_multi_turn_base`](./BFCL_gen_multi_turn_base.py) - Multi-turn: Basic scenario
- [`BFCL_gen_multi_turn_miss_func`](./BFCL_gen_multi_turn_miss_func.py) - Multi-turn: Missing function
- [`BFCL_gen_multi_turn_miss_param`](./BFCL_gen_multi_turn_miss_param.py) - Multi-turn: Missing parameter
- [`BFCL_gen_multi_turn_long_context`](./BFCL_gen_multi_turn_long_context.py) - Multi-turn: Long context

### Test Groups
Suitable for batch testing, running multiple related test categories at once:

- [`BFCL_gen_all`](./BFCL_gen_all.py) - Full test: Includes all test categories
- [`BFCL_gen_single_turn`](./BFCL_gen_single_turn.py) - Single-turn dialogue test:
  - [`BFCL_gen_simple`](./BFCL_gen_simple.py): Simple function calls (Single-turn: Basic Python function calls)
  - [`BFCL_gen_irrelevance`](./BFCL_gen_irrelevance.py): With irrelevant documents (Single-turn: Function calls with irrelevant documents)
  - [`BFCL_gen_parallel`](./BFCL_gen_parallel.py): Parallel calls (Single-turn: Parallel function calls)
  - [`BFCL_gen_multiple`](./BFCL_gen_multiple.py): Sequential calls of multiple functions (Single-turn: Sequential calls of multiple functions)
  - [`BFCL_gen_parallel_multiple`](./BFCL_gen_parallel_multiple.py): Mixed parallel and sequential (Single-turn: Mixed parallel and sequential calls)
  - [`BFCL_gen_java`](./BFCL_gen_java.py): Java function calls (Single-turn: Basic Java function calls)
  - [`BFCL_gen_javascript`](./BFCL_gen_javascript.py): JavaScript function calls (Single-turn: Basic JavaScript function calls)
- [`BFCL_gen_multi_turn`](./BFCL_gen_multi_turn.py) - Multi-turn dialogue test:
  - [`BFCL_gen_multi_turn_base`](./BFCL_gen_multi_turn_base.py): Multi-turn: Basic scenario
  - [`BFCL_gen_multi_turn_miss_func`](./BFCL_gen_multi_turn_miss_func.py): Multi-turn: Missing function
  - [`BFCL_gen_multi_turn_miss_param`](./BFCL_gen_multi_turn_miss_param.py): Multi-turn: Missing parameter
  - [`BFCL_gen_multi_turn_long_context`](./BFCL_gen_multi_turn_long_context.py): Multi-turn: Long context
- [`BFCL_gen_live`](./BFCL_gen_live.py) - Live test:
  - [`BFCL_gen_live_simple`](./BFCL_gen_live_simple.py): Live - Simple calls
  - [`BFCL_gen_live_multiple`](./BFCL_gen_live_multiple.py): Live - Sequential multiple functions
  - [`BFCL_gen_live_parallel`](./BFCL_gen_live_parallel.py): Live - Parallel calls
  - [`BFCL_gen_live_parallel_multiple`](./BFCL_gen_live_parallel_multiple.py): Live - Mixed parallel and sequential
  - [`BFCL_gen_live_irrelevance`](./BFCL_gen_live_irrelevance.py): Live - With irrelevant documents
  - [`BFCL_gen_live_relevance`](./BFCL_gen_live_relevance.py): Live - With relevant documents
- [`BFCL_gen_non_live`](./BFCL_gen_non_live.py) - Standard test:
  - [`BFCL_gen_simple`](./BFCL_gen_simple.py): Simple function calls
  - [`BFCL_gen_irrelevance`](./BFCL_gen_irrelevance.py): With irrelevant documents
  - [`BFCL_gen_parallel`](./BFCL_gen_parallel.py): Parallel calls
  - [`BFCL_gen_multiple`](./BFCL_gen_multiple.py): Sequential calls of multiple functions
  - [`BFCL_gen_parallel_multiple`](./BFCL_gen_parallel_multiple.py): Mixed parallel and sequential
  - [`BFCL_gen_java`](./BFCL_gen_java.py): Java function calls
  - [`BFCL_gen_javascript`](./BFCL_gen_javascript.py): JavaScript function calls

### Precise Test Configuration
Suitable for debugging and precise verification of specific test cases:

Use `--datasets BFCL_gen_ids` to specify specific test case IDs for precise testing.

#### Configuration Method
Edit the `test_ids_to_generate` dictionary in the [`BFCL_gen_ids.py`](./BFCL_gen_ids.py) file:

```python
test_ids_to_generate = {
    "simple": ["simple_0"], # Specify the case named simple_0 in BFCL_gen_simple
    "irrelevance": [],      # An empty list means no cases are specified for this category
    "parallel": ["parallel_0"],
    "multiple": ["multiple_0"],
    "parallel_multiple": ["parallel_multiple_0"],
    "java": [],
    "javascript": ["javascript_0"],
    "live_simple": ["live_simple_0-0-0"],
    "live_multiple": ["live_multiple_0-0-0"],
    "live_parallel": ["live_parallel_0-0-0"],
    "live_parallel_multiple": ["live_parallel_multiple_0-0-0"],
    "live_irrelevance": ["live_irrelevance_0-0-0"],
    "live_relevance": ["live_relevance_0-0-0"],
    "multi_turn_base": ["multi_turn_base_0"],
    "multi_turn_miss_func": ["multi_turn_miss_func_0"],
    "multi_turn_miss_param": ["multi_turn_miss_param_0"],
    "multi_turn_long_context": ["multi_turn_long_context_0"],
}
```

## 💡 Usage Recommendations
1. **First Evaluation**: It is recommended to use [Test Groups](#test-groups) for a relatively comprehensive assessment.
2. **Performance Tuning**: Select specific categories for in-depth testing based on initial results.
3. **Issue Localization**: Use precise test configurations to locate specific problematic test cases.

---