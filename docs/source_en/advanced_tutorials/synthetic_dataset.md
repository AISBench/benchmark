# Guide to Using Random Synthetic Datasets

## I. Application Introduction

> This feature only supports performance evaluation scenarios and does not support accuracy evaluation scenarios.

This feature is designed for scenarios where real datasets are unavailable, and randomly constructed synthetic datasets are used for large language model inference performance benchmarking.


## II. Usage Guide

### 2.1 Modifying the Configuration File

Configure the `synthetic_config` parameter in the configuration template files under the `ais_bench/benchmark/configs/datasets/synthetic` directory.

Currently, **two different types** of random dataset configuration template files are provided. Users can select and modify the appropriate template file according to their needs:

- **`synthetic_gen_string.py`**: Generates random-length strings (simulating real input)

  ```python
  synthetic_config = {
      "Type": "string",
      ... # Other parameters
  }
  ```

- **`synthetic_gen_tokenid.py`**: Generates random token ID sequences (directly inputting encoded tokens)

  ```python
  synthetic_config = {
      "Type": "tokenid",
      ... # Other parameters
  }
  ```

**Note:** The configuration used by the [`ais_bench/benchmark/configs/datasets/synthetic/synthetic_gen.py`](../../../ais_bench/benchmark/configs/datasets/synthetic/synthetic_gen.py) template needs to be modified in [`ais_bench/benchmark/configs/datasets/synthetic`](../../../ais_bench/datasets/synthetic/synthetic_config.py). To unify the dataset configuration mode, this method will be deprecated in future versions.

### 2.2 Command Execution

Run the following command in the command line to start the evaluation:

```bash
ais_bench --models {model_api_file} --datasets synthetic_gen_{string/tokenid} {other_option_args}
```


## III. Parameter Description

> The following is a general description of parameters in the configuration files. For detailed value requirements, refer to the comments in the configuration files and specific usage scenarios.

### 3.1 Public Parameters

| Parameter Name | Type   | Description                          | Value Range       |
|----------------|--------|--------------------------------------|-------------------|
| Type           | string | Dataset type (required)              | string/tokenid    |
| RequestCount   | int    | Total number of generated requests (required) | [1, 1,048,576] |


### 3.2 String Type Configuration (Required when Type="string")

```python
"StringConfig" : {
    "Input" : {          # Input sequence configuration
        "Method": str,    # Distribution type: uniform/gaussian/zipf
        "Params": {}      # Parameters for the corresponding distribution
    },
    "Output" : {         # Output sequence configuration (parameters same as above)
        "Method": str,
        "Params": {}
    }
}
```

#### Description of Input/Output Distribution Parameters

> **Key Rules**
>
> - The maximum value of all numerical parameters should not exceed 2^20 (i.e., 1,048,576) by default.
> - The maximum input/output length of requests is also limited by service configuration. Refer to the comments in the configuration file for details.

| Distribution Type | Parameter   | Type  | Description                                                                 | Value Range          |
|-------------------|-------------|-------|-----------------------------------------------------------------------------|---------------------|
| **uniform**       | `MinValue`  | int   | Minimum length of input/output sequences                                    | [1, 1,048,576]      |
|                   | `MaxValue`  | int   | Maximum length of input/output sequences (can equal MinValue)               | [≥MinValue]         |
| **gaussian**      | `Mean`      | float | Central value of the distribution (mean)                                    | [-3.0e38, 3.0e38]   |
|                   | `Var`       | float | Variance (controls data dispersion)                                         | [0, 3.0e38]         |
|                   | `MinValue`  | int   | Hard lower limit for input/output sequence length                           | [1, 1,048,576]      |
|                   | `MaxValue`  | int   | Hard upper limit for input/output sequence length                           | [≥MinValue]         |
| **zipf**          | `Alpha`     | float | Shape parameter (larger values make the distribution more uniform)          | (1.0, 10.0]         |
|                   | `MinValue`  | int   | Minimum length of input/output sequences                                    | [1, 1,048,576]      |
|                   | `MaxValue`  | int   | Maximum length of input/output sequences (**must** be greater than MinValue) | [>MinValue]         |


### 3.3 TokenId Type Configuration (Required when Type="tokenid")

```python
"TokenIdConfig" : {
    "RequestSize": int,   # Number of tokens per request
    "PrefixLen": int     # Number of common prefix tokens for all requests
}
```


## IV. Configuration Examples

### 4.1 String Type Examples

#### 1. Uniform Distribution

```python
synthetic_config = {
    "Type": "string",
    "RequestCount": 1000,
    "StringConfig": {
        "Input": {
            "Method": "uniform",
            "Params": {"MinValue": 50, "MaxValue": 500}  # Input length: 50-500
        },
        "Output": {
            "Method": "uniform",
            "Params": {"MinValue": 20, "MaxValue": 200}  # Output length: 20-200
        }
    }
}
```

**Features**: Input/output lengths are evenly distributed within the range, suitable for baseline performance testing.


#### 2. Gaussian Distribution

```python
synthetic_config = {
    "Type": "string",
    "RequestCount": 800,
    "StringConfig": {
        "Input": {
            "Method": "gaussian",
            "Params": {
                "Mean": 256,       # Central value: 256
                "Var": 10,         # Variance: 10
                "MinValue": 64,    # Actual range: 64-512
                "MaxValue": 512
            }
        },
        "Output": {
            "Method": "gaussian",
            "Params": {
                "Mean": 128,
                "Var": 50,
                "MinValue": 32,
                "MaxValue": 256
            }
        }
    }
}
```

**Distribution Characteristics**: Approximately 95% of input lengths fall within [236, 276] (μ±2σ).


#### 3. Zipf Distribution

```python
synthetic_config = {
    "Type": "string",
    "RequestCount": 1200,
    "StringConfig": {
        "Input": {
            "Method": "zipf",
            "Params": {
                "Alpha": 1.5,      # Strong long-tail effect
                "MinValue": 10,    # Input length range: 10-1000
                "MaxValue": 1000
            }
        },
        "Output": {
            "Method": "zipf",
            "Params": {
                "Alpha": 2.0,     # Flatter distribution
                "MinValue": 5,
                "MaxValue": 500
            }
        }
    }
}
```

**Typical Scenario**: Simulates long-tail distribution of requests in real scenarios. When Alpha=1.5, approximately 20% of requests account for 60% of the computation.


#### 4. Mixed Distribution Configuration

```python
synthetic_config = {
    "Type": "string",
    "RequestCount": 1500,
    "StringConfig": {
        "Input": {
            "Method": "zipf",    # Long-tail distribution for input
            "Params": {
                "Alpha": 1.2,
                "MinValue": 10,
                "MaxValue": 2000
            }
        },
        "Output": {
            "Method": "uniform",  # Uniform distribution for output
            "Params": {
                "MinValue": 50,
                "MaxValue": 300
            }
        }
    }
}
```


### 4.2 TokenId Type Examples

#### Long Text Stress Testing

```python
synthetic_config = {
    "Type": "tokenid",
    "RequestCount": 1000,
    "TokenIdConfig": {
        "RequestSize": 2048   # 2048 tokens per request
    }
}
```

#### Short Text Performance Testing

```python
synthetic_config = {
    "Type": "tokenid",
    "RequestCount": 5000,
    "TokenIdConfig": {
        "RequestSize": 128    # Short text processing scenario
    }
}
```

#### prefix Cache Performance Testing

```python
synthetic_config = {
    "Type": "tokenid",
    "RequestCount": 5000,
    "TokenIdConfig": {
        "RequestSize": 512,    # 512 tokens per request
        "PrefixLen": 256       # All requests have the first 256 tokens equal
    }
}
```

## V. Frequently Asked Questions

### Q1: How to choose a distribution type?

- **Uniform distribution**: Suitable for baseline scenarios in stress testing.
- **Gaussian distribution**: Simulates average request lengths in real scenarios.
- **Zipf distribution**: Generates long-tail distributed data (e.g., 1% of requests account for 50% of computation).
- **Suggested distribution combinations**:
  - **Stress testing**: Use zipf distribution for Input and uniform distribution for Output.
  - **Stability testing**: Use gaussian distribution for both Input and Output.


### Q2: Why does the performance evaluation result matrix show unexpected values even after specifying the *input length*?

- **`tokenid` mode**: When sending requests, prompts of specified length (composed of randomly generated tokens within the model's vocabulary range) are re-decoded into strings before being sent to the service. Fluctuations may occur due to possible many-to-one or one-to-many vocabulary mappings in different models.
- **`string` mode**: The input length here refers to the length of the input string, not the number of tokens.
- **Preprocessing stage**: Additional string concatenation may be performed before/after using chat-related APIs.


### Q3: Why does the performance evaluation result matrix show unexpected values even after specifying the *output length* in String mode?

- **Significant discrepancy**: Check if the `ignore_eos` parameter in `generation_kwargs` of the model API configuration file is correctly set to `True` (this ensures the service ignores the end-of-sequence token until the preset output length is reached).


## VI. Notes

1. **`tokenid` mode**: The value range of `tokenid` depends on the vocabulary range of the model specified in the model configuration file.
2. **`string` mode**: A fixed-length sequence is generated when MinValue=MaxValue.