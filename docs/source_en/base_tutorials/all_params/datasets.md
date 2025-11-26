# Dataset Preparation Guide
## Supported Dataset Types
The dataset types currently supported by AISBench Benchmark are as follows:
1. [Open-Source Datasets](#open-source-datasets)：Cover multiple domains including general language understanding (e.g., ARC, SuperGLUE_BoolQ, MMLU), mathematical reasoning (e.g., GSM8K, AIME2024, Math), code generation (e.g., HumanEval, MBPP, LiveCodeBench), text summarization (e.g., XSum, LCSTS), and multimodal tasks (e.g., TextVQA, VideoBench, VocalSound). They meet the needs of comprehensive evaluation of language models in terms of multi-task, multimodal, and multilingual capabilities.
2. [Randomly Synthesized Datasets](#randomly-synthesized-datasets)：Support specifying the length of input/output sequences and the number of requests. They are suitable for performance testing scenarios that have requirements for sequence distribution and data scale.
3. [Custom Datasets](#custom-datasets)：Support converting user-defined data content into data in a fixed format for evaluation. They are applicable to customized accuracy and performance testing scenarios.


## Open-Source Datasets
Open-source datasets refer to widely used, publicly accessible datasets in the community. They are typically used for model training, validation, and comparing the performance of different algorithms. AISBench Benchmark supports multiple mainstream open-source datasets, enabling users to quickly conduct standardized tests. Detailed introductions and acquisition methods are as follows:

### LLM Datasets
| Dataset Name | Category | Detailed Introduction & Acquisition Method |
|---------------|----------------------|-------------------|
| DEMO | Mathematical Reasoning | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/demo/README_en.md) |
| ARC_c | Reasoning (Common Sense + Science) | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/ARC_c/README_en.md) |
| ARC_e | Reasoning (Common Sense + Science) | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/ARC_e/README_en.md) |
| SuperGLUE_BoolQ | Natural Language Understanding (Q&A) | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/SuperGLUE_BoolQ/README_en.md) |
| agieval | Comprehensive Exams / Reasoning | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/agieval/README_en.md) |
| aime2024 | Mathematical Reasoning | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/aime2024/README_en.md) |
| aime2025 | Mathematical Reasoning | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/aime2025/README_en.md) |
| bbh | Multi-Task (Big-Bench Hard) | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/bbh/README_en.md) |
| cmmlu | Chinese Understanding / Knowledge Q&A | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/cmmlu/README_en.md) |
| ceval | Chinese Professional Exams | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/ceval/README_en.md) |
| drop | Reading Comprehension + Reasoning | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/drop/README_en.md) |
| gsm8k | Mathematical Reasoning | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/gsm8k/README_en.md) |
| gpqa | Knowledge Q&A | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/gpqa/README_en.md) |
| hellaswag | Common Sense Reasoning | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/hellaswag/README_en.md) |
| humaneval | Programming (Code Generation + Testing) | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/humaneval/README_en.md) |
| humanevalx | Programming (Multilingual) | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/humanevalx/README_en.md) |
| ifeval | Programming (Function Generation) | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/ifeval/README_en.md) |
| lambada | Long Text Cloze | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/lambada/README_en.md) |
| lcsts | Chinese Text Summarization | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/lcsts/README_en.md) |
| livecodebench | Programming (Real-Time Code) | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/livecodebench/README_en.md) |
| longbench | Long Sequences | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/longbench/README_en.md) |
| longbenchv2 | Long Sequences | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/longbenchv2/README_en.md) |
| math | Advanced Mathematical Reasoning | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/math/README_en.md) |
| mbpp | Programming (Python) | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/mbpp/README_en.md) |
| mgsm | Multilingual Mathematical Reasoning | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/mgsm/README_en.md) |
| mmlu | Multidisciplinary Understanding (English) | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/mmlu/README_en.md) |
| mmlu_pro | Multidisciplinary Understanding (Professional Version) | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/mmlu_pro/README_en.md) |
| needlebench_v2 | Long Sequences | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/needlebench_v2/README_en.md) |
| piqa | Physical Common Sense Reasoning | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/piqa/README_en.md) |
| siqa | Social Common Sense Reasoning | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/siqa/README_en.md) |
| triviaqa | Knowledge Q&A | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/triviaqa/README_en.md) |
| winogrande | Common Sense Reasoning (Pronoun Resolution) | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/winogrande/README_en.md) |
| Xsum | Text Generation (Summarization) | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/Xsum/README_en.md) |
| BFCL | Function Calling Capability Evaluation | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/BFCL/README_en.md) |

### Multimodal Datasets
| Dataset Name | Category | Detailed Introduction & Acquisition Method |
|---------------|----------------------|-------------------|
| textvqa | Multimodal Understanding (Image + Text) | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/textvqa/README_en.md) |
| videobench | Multimodal Understanding (Video) | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/videobench/README_en.md) |
| vocalsound | Multimodal Understanding (Audio) | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/vocalsound/README_en.md) |
| Omnidocbench          | Image OCR (Image + Text)    | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/omnidocbench/README.md) |
| MMMU       | Multimodal Understanding (Image + Text)      | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/mmmu/README.md) |
| MMMU_Pro       | Multimodal Understanding (Image + Text)      | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/mmmu_pro/README.md) |
| InfoVQA       | Multimodal Understanding (Image + Text)      | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/infovqa/README.md) |
| DocVQA       | Multimodal Understanding (Image + Text)     | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/docvqa/README.md) |

### Multi-Turn Dialogue Datasets
| Dataset Name | Category | Detailed Introduction & Acquisition Method |
|---------------|----------------------|-------------------|
| sharegpt | Multi-Turn Dialogue | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/sharegpt/README_en.md) |
| mtbench | Multi-Turn Dialogue | [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/mtbench/README_en.md) |


**Tip**: Users can uniformly place the acquired dataset folders in the `ais_bench/datasets/` directory. AISBench Benchmark will automatically retrieve the dataset files in this directory based on the dataset configuration file for testing.

### Configuring Open-Source Datasets
The configurations of AISBench Benchmark's open-source datasets are stored in the [`configs/datasets`](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets) directory by dataset name. Each dataset's corresponding folder contains multiple dataset configurations, with the file structure as shown below:

```text
ais_bench/benchmark/configs/datasets
├── agieval
├── aime2024
├── ARC_c
├── ...
├── gsm8k  # Dataset
│   ├── gsm8k_gen.py  # Configuration files for different versions of the dataset
│   ├── gsm8k_gen_0_shot_cot_str_perf.py
│   ├── gsm8k_gen_0_shot_cot_chat_prompt.py
│   ├── gsm8k_gen_0_shot_cot_str.py
│   ├── gsm8k_gen_4_shot_cot_str.py
│   ├── gsm8k_gen_4_shot_cot_chat_prompt.py
│   └── README.md
├── ...
├── vocalsound
├── winogrande
└── Xsum
```

The name of an open-source dataset configuration follows the format: `{dataset_name}_{evaluation_method}_{number_of_shots}_shot_{chain_of_thought_rule}_{request_type}_{task_category}.py`. Taking `gsm8k/gsm8k_gen_0_shot_cot_chat_prompt.py` as an example, this configuration file corresponds to the `gsm8k` dataset. The evaluation method is `gen` (generative evaluation, currently only generative evaluation is supported), the number of shot prompts is 0, the chain-of-thought rule is `cot` (indicating that the request includes chain-of-thought prompts; if not specified, there are no chain-of-thought prompts), `chat_prompt` indicates the request type is dialogue, and the task category is not specified (defaulting to accuracy testing). Similarly, `gsm8k_gen_0_shot_cot_str_perf.py` specifies the request type as `str` (string), and the request type `perf` indicates the template is used for performance evaluation tasks.

> 💡 **Tip**: When specifying the dataset configuration name, the `.py` suffix can be omitted.

The configuration parameters of open-source datasets are also described using Python syntax. Taking gsm8k as an example, the parameter content is as follows:
```python
gsm8k_datasets = [
    dict(
        abbr='gsm8k',                       # Unique identifier of the dataset in the evaluation task
        type=GSM8KDataset,                  # Dataset class member, bound to the dataset; modification is not supported temporarily
        path='ais_bench/datasets/gsm8k',    # Dataset path; relative paths are relative to the source code root directory, and absolute paths are supported
        reader_cfg=gsm8k_reader_cfg,    # Data reading configuration; modification is not supported temporarily
        infer_cfg=gsm8k_infer_cfg,      # Inference evaluation configuration; modification is not supported temporarily
        eval_cfg=gsm8k_eval_cfg)        # Accuracy evaluation configuration; modification is not supported temporarily
]
```


## Randomly Synthesized Datasets
Synthesized datasets are automatically generated by programs and are suitable for testing the generalization ability of models under different input lengths, distributions, and modes. AISBench Benchmark provides two types of synthesized datasets: random character sequences and random token sequences. No additional download is required—users only need to set parameters through the configuration file to use them. For details, see: 📚 [Guide to Using Synthesized Random Dataset Configuration Files](../../advanced_tutorials/synthetic_dataset.md)

### Usage Method
The usage method is the same as that of open-source datasets. Simply select the required configuration file in the `ais_bench/benchmark/configs/datasets/synthetic/` directory. Currently, [synthetic_gen.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/synthetic/synthetic_gen.py) is available. An example command is as follows:

```bash
ais_bench --models vllm_api_stream_chat --datasets synthetic_gen
```


## Custom Datasets
AISBench Benchmark supports users in integrating custom datasets to meet specific business needs. Users can organize private data into a standard format and seamlessly integrate it into the evaluation process through built-in interfaces. For details, see: 📚 [Guide to Using Custom Datasets](../../advanced_tutorials/custom_dataset.md)