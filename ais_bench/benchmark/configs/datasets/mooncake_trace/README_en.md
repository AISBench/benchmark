# Mooncake Trace

[中文](README.md) | English

## Dataset Introduction

Mooncake Trace is a trace dataset for performance evaluation. It generates prompts from JSONL files containing hash_id and trace data. The dataset implements an AIPerf-compatible prompt generation mechanism, including reproducible random number generation, corpus sampling, hash_id caching, and precise request timing via the timestamp field.

**Reference & Implementation**: The corpus is sourced from [AIPerf](https://github.com/ai-dynamo/aiperf). The prompt generation mechanism (reproducible RNG, corpus sampling, hash_id caching, etc.) references AIPerf's implementation.

Main features:

- **Hash ID caching**: Same hash_id yields the same prompt content for reproducibility
- **Timestamp control**: Use the timestamp field to control request send times and simulate real traffic
- **input_text support**: Specify prompts via the input_text field; mixed mode (some input_text, some generated) is supported
- **Fixed schedule**: Auto-offset, start offset, and end offset for flexible time windows
- **Caching**: Generated prompts are cached to avoid regeneration and speed up loading

## Dataset Format

The dataset file is in JSONL format, one JSON object per line. Supported fields:

### Required Fields (one of two)

- `input_length` (int): Number of input tokens used to generate the prompt
- `input_text` (str, optional): Literal prompt text. If present, `input_length` and `hash_ids` are ignored

### Optional Fields

- `output_length` (int): Maximum output tokens; default 1
- `hash_ids` (list[int]): Hash ID list for cache reuse; each hash_id maps to one token block (default 512 tokens)
- `timestamp` (int or float, optional): Request timestamp in milliseconds for send-time control; must be >= 0. If omitted, the record has no timestamp and does not participate in timestamp-based scheduling or filtering

### Hash IDs and Input Length Rules

When using `hash_ids`, `input_length` must satisfy:

- **One token block per hash_id**: Default 512 tokens per hash_id (block_size = 512)
- **Formula**:
  - For n hash_ids, the first n-1 each use 512 tokens; the last uses `final_block_size` tokens
  - `final_block_size = input_length - (n-1) * 512`
  - Must hold: `0 < final_block_size <= 512`

- **Valid range**:
  - For n hash_ids: `input_length` in `[(n-1) * 512 + 1, n * 512]`
  - Min: `(n-1) * 512 + 1`, Max: `n * 512`

- **Examples**:
  - 1 hash_id: `input_length` in `[1, 512]`
  - 2 hash_ids: `input_length` in `[513, 1024]`
  - 3 hash_ids: `input_length` in `[1025, 1536]`
  - 4 hash_ids: `input_length` in `[1537, 2048]`

- **Validation**:

  ```
  final_block_size = input_length - (len(hash_ids) - 1) * 512
  Must satisfy: 0 < final_block_size <= 512
  ```

### Data Format Examples

```jsonl
# Prompt from hash_ids (1 hash_id, input_length 1–512)
{"timestamp": 0, "input_length": 256, "output_length": 10, "hash_ids": [1]}

# Prompt from hash_ids (2 hash_ids, input_length 513–1024)
{"timestamp": 200, "input_length": 1024, "output_length": 100, "hash_ids": [2, 3]}

# Prompt from hash_ids (3 hash_ids, input_length 1025–1536)
{"timestamp": 500, "input_length": 1536, "output_length": 200, "hash_ids": [4, 5, 6]}

# Direct input_text (no hash_ids)
{"timestamp": 1000, "input_text": "What is the capital of France?", "output_length": 10}
```

**Note**: Example `input_length` values lie within the valid range for the given number of hash_ids.

## Dataset Deployment

1. Prepare a JSONL trace data file
2. Place it in a suitable path (relative or absolute)
3. Set the `path` parameter in the config to point to the file

## Configuration Parameters

### Basic Parameters

| Parameter | Type | Description | Default |
| ---- | ---- | ---- | ---- |
| `path` | str | Path to the JSONL file with hash_id and trace data. Relative paths are relative to the project root; absolute paths are supported | Required |
| `model_path` | str | Model path for loading the tokenizer | Required |
| `random_seed` | int | Random seed for reproducibility | None |
| `generated_prompts_path` | str | Path for generated prompt cache. If empty, a default cache path is used. Relative paths are relative to the project root | "" |

### Fixed Schedule Parameters

| Parameter | Type | Description | Default |
| ---- | ---- | ---- | ---- |
| `fixed_schedule_auto_offset` | bool | Whether to auto-offset timestamps so the first is 0 | False |
| `fixed_schedule_start_offset` | int | Start offset in ms; timestamps below this are filtered out | 0 |
| `fixed_schedule_end_offset` | int | End offset in ms; -1 means no limit; if >= 0, timestamps above it are filtered out | -1 |

### Parameter Usage

1. **fixed_schedule_auto_offset**:
   - When True, the minimum timestamp is computed and subtracted from all timestamps so the first is 0
   - Use when timestamps do not start at 0

2. **fixed_schedule_start_offset** and **fixed_schedule_end_offset**:
   - Define a time window for the trace
   - `start_offset`: keep only requests with timestamp >= this value
   - `end_offset`: keep only requests with timestamp <= this value (-1 = no limit)
   - Example: `start_offset=1000, end_offset=5000` processes only timestamps in [1000, 5000]

3. **Caching**:
   - When fixed_schedule parameters are used, the cache filename includes them so different settings use different caches
   - If the cache file exists, it is loaded and prompt generation is skipped
   - Delete the cache file to regenerate

## Available Dataset Tasks

| Task Name | Description | Metrics | few-shot | Prompt Format | Config Path |
| --- | --- | --- | --- | --- | --- |
| mooncake-trace | Mooncake trace generative task | Performance | 0-shot | String | [mooncake_trace_gen.py](mooncake_trace_gen.py) |

## Usage Examples

### Basic Configuration

```python
mooncake_trace_datasets = [
    dict(
        abbr='mooncake-trace',
        type=MooncakeTraceDataset,
        path='path/to/trace.jsonl',
        model_path='path/to/model',
        random_seed=1234,
        generated_prompts_path='',  # auto cache path
        fixed_schedule_auto_offset=False,
        fixed_schedule_start_offset=0,
        fixed_schedule_end_offset=-1,
        reader_cfg=mooncake_trace_reader_cfg,
        infer_cfg=mooncake_trace_infer_cfg,
        eval_cfg=mooncake_trace_eval_cfg
    )
]
```

### Timestamp Auto-Offset

```python
mooncake_trace_datasets = [
    dict(
        abbr='mooncake-trace',
        type=MooncakeTraceDataset,
        path='path/to/trace.jsonl',
        model_path='path/to/model',
        random_seed=1234,
        fixed_schedule_auto_offset=True,  # enable auto-offset
        fixed_schedule_start_offset=0,
        fixed_schedule_end_offset=-1,
        # ... other config
    )
]
```

### Time Window

```python
mooncake_trace_datasets = [
    dict(
        abbr='mooncake-trace',
        type=MooncakeTraceDataset,
        path='path/to/trace.jsonl',
        model_path='path/to/model',
        random_seed=1234,
        fixed_schedule_auto_offset=False,
        fixed_schedule_start_offset=1000,  # from 1s
        fixed_schedule_end_offset=5000,   # to 5s
        # ... other config
    )
]
```

### Using input_text

Specify prompts directly in the JSONL file:

```jsonl
{"timestamp": 0, "input_text": "What is the capital of France?", "output_length": 10}
{"timestamp": 200, "input_length": 655, "output_length": 2, "hash_ids": [46, 47]}
```

Mixed mode is supported: some rows use `input_text`, others use `input_length` and `hash_ids` for generation.

## Feature Description

### Hash ID Caching

- Each `hash_id` maps to one token block (default 512 tokens)
- Same `hash_id` produces the same token sequence for reproducibility
- Caching avoids duplicate generation and improves efficiency

### Timestamp Control

- The `timestamp` field controls when requests are sent
- In performance mode, the system uses timestamps to compute delays and schedule sends
- Whether timestamp-based scheduling is used is determined by **use_timestamp** in the model config: when **use_timestamp** is True and the dataset has timestamps, requests follow timestamps and **request_rate** is ignored; when **use_timestamp** is False, **request_rate** is used and dataset timestamps are not used for scheduling
- Set `use_timestamp=True` in the **model configuration file** to enable timestamp-based scheduling; see the [Model Configuration documentation](../../../../../docs/source_en/base_tutorials/all_params/models.md) for the parameter description.

### input_text Support

- When a trace row has `input_text`, it is used as the prompt
- `input_length` and `hash_ids` are ignored in that case
- Mixed mode: some rows use `input_text`, others use generated prompts

## Notes

1. **Corpus file**: The corpus is sourced from [AIPerf](https://github.com/ai-dynamo/aiperf). The system uses Shakespeare text as the corpus (`assets/shakespeare.txt`) for prompt generation. Ensure the file is at **`ais_bench/third_party/aiperf/assets/shakespeare.txt`** (relative to the ais_bench package root). On error, the message lists the paths that were checked.

2. **Cache file**: Generated prompts are cached. If the data file or config changes, delete the cache to regenerate.

3. **Timestamp unit**: All timestamp-related parameters are in **milliseconds**.

4. **Parameter validation**: When `fixed_schedule_end_offset >= 0`, `fixed_schedule_start_offset` must be <= `fixed_schedule_end_offset`.

5. **Data format validation**: When using `hash_ids`, ensure `input_length` matches the number of hash_ids:
   - **Common error**: `input_length` outside the valid range
     - Bad: `{"input_length": 512, "hash_ids": [1, 2]}`
     - Reason: For 2 hash_ids, `input_length` must be 513–1024
     - Good: `{"input_length": 1024, "hash_ids": [1, 2]}`

   - **Validation**:

     ```
     final_block_size = input_length - (len(hash_ids) - 1) * 512
     If final_block_size <= 0 or final_block_size > 512, the format is invalid
     ```

   - **Quick check**: For n hash_ids, ensure `(n-1) * 512 + 1 <= input_length <= n * 512`

   - **Error**: Invalid format raises **ParameterValueError** (code **DSET-PARAM-004**), e.g.:

     ```
     Input length: XXX, Hash IDs: [...], Block size: 512 are not compatible.
     Final block size: XXX must be > 0 and <= 512.
     ```

     See error code **DSET-PARAM-004** in the [error codes documentation](../../../../../docs/source_en/faqs/error_codes.md).
