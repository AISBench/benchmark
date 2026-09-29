# MME

中文 | [English](README_en.md)

## Dataset Overview

MME is a comprehensive benchmark for evaluating image understanding in multimodal large language models. It contains 14 perception and cognition tasks. Each image is associated with two Yes/No questions. ACC measures the accuracy of individual questions, while ACC+ gives credit for an image only when both of its questions are answered correctly.

> 🔗 Dataset home: [https://huggingface.co/datasets/darkyarding/MME](https://huggingface.co/datasets/darkyarding/MME)


## Dataset Setup

- Download the dataset from the Hugging Face dataset page: [https://huggingface.co/datasets/darkyarding/MME/tree/main](https://huggingface.co/datasets/darkyarding/MME/tree/main).
- MME is provided in Parquet format and contains two test shards. The bundled config reads all `.parquet` files from `C:\需求\MME\MME\data`.
- After downloading, run `tree MME /F` from `C:\需求\MME` to check the directory structure. The dataset is ready when it looks like this:

    ```text
    MME/
    ├── .gitattributes
    ├── MME_Benchmark_release_version.zip
    ├── README.md
    └── data/
        ├── test-00000-of-00002.parquet
        └── test-00001-of-00002.parquet
    ```

  Evaluation uses the two Parquet shards under `data/`. The zip archive in the repository root is not used by this adapter.

- Each Parquet record must contain the following fields:

    | Field | Description |
    | --- | --- |
    | `question_id` | Unique question identifier, used to associate the two questions for an image |
    | `image` | Image bytes, converted to a base64 data URL when loaded |
    | `question` | Original Yes/No question, used directly to construct the prompt |
    | `answer` | Ground-truth answer |
    | `category` | Evaluation task/category |


## Available Dataset Tasks

| Task | Description | Metrics | Few-Shot | Prompt Format | Source Config |
| --- | --- | --- | --- | --- | --- |
| mme_gen_base64 | MME multimodal image understanding benchmark | 14 task scores | 0-shot | Multimodal base64 | mme_gen_base64.py |

Data loading and prompt message construction follow the InfoVQA style, with the image before the text. The prompt uses the `question` from Parquet verbatim and does not append an additional answer instruction. Images are encoded as base64 data URLs with the matching JPEG or PNG MIME type.


## Evaluation Output

Evaluation reports:

- The CLI, `mme.json`, and summary files expose only the 14 official task scores in official order, where `score = ACC + ACC+`;
- `<work_dir>/results/<model>/mme_results/mme_metrics.json` groups tasks under `Perception` and `Cognition` and stores each task's `ACC`, `ACC+`, and `score`, together with each group's `total_score`;
- 14 task-specific txt result files under `<work_dir>/results/<model>/mme_results/`.

`mme_metrics.json` uses the following structure:

```json
{
    "Perception": {
        "total_score": 1647.3155262104842,
        "tasks": {
            "existence": {
                "ACC": 100.0,
                "ACC+": 100.0,
                "score": 200.0
            }
        }
    },
    "Cognition": {
        "total_score": 751.7857142857142,
        "tasks": {
            "code_reasoning": {
                "ACC": 97.5,
                "ACC+": 95.0,
                "score": 192.5
            }
        }
    }
}
```

Each line in a result file uses the following four-column format:

```text
image_name\tquestion\tground_truth\tmodel_response
```

The two questions associated with the same `question_id` are written next to each other. Newlines and tabs in model responses are replaced with spaces.
