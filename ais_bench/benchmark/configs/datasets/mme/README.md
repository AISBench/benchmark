# MME

中文 | [English](README_en.md)

## 数据集简介

MME 是一个用于评估多模态大语言模型图像理解能力的综合评测集，共包含 14 个感知与认知子任务。每张图像对应两道 Yes/No 问题；ACC 统计单题准确率，ACC+ 仅在同一张图像的两道问题均回答正确时计分。

> 🔗 数据集主页链接：[https://huggingface.co/datasets/darkyarding/MME](https://huggingface.co/datasets/darkyarding/MME)


## 数据集部署

- 可以从 Hugging Face 数据集页面 🔗 [https://huggingface.co/datasets/darkyarding/MME/tree/main](https://huggingface.co/datasets/darkyarding/MME/tree/main) 获取数据集。
- MME 数据集使用 Parquet 格式，共包含两个 test 分片。本适配默认从 benchmark 仓库内的 `ais_bench/datasets/MME/data` 目录读取全部 `.parquet` 文件。
- 下载后，在 benchmark 仓库根目录执行 `tree ais_bench\datasets\MME /F` 检查目录结构。如果目录结构如下所示，则数据集部署成功：

    ```text
    ais_bench/
    └── datasets/
        └── MME/
            └── data/
                ├── test-00000-of-00002.parquet
                └── test-00001-of-00002.parquet
    ```

  评测直接使用 `data/` 目录下的两个 Parquet 分片，目录中不需要保留原始 zip 压缩包。

- 每条 Parquet 数据应包含以下字段：

    | 字段 | 说明 |
    | --- | --- |
    | `question_id` | 题目唯一标识，用于关联同一图像的两道问题 |
    | `image` | 图像二进制数据，加载后转换为 base64 data URL |
    | `question` | 原始 Yes/No 问题，直接用于构造 prompt |
    | `answer` | 标准答案 |
    | `category` | 所属评测子任务 |


## 可用数据集任务

| 任务名称 | 简介 | 评估指标 | Few-Shot | Prompt 格式 | 对应源码配置文件路径 |
| --- | --- | --- | --- | --- | --- |
| mme_gen_base64 | MME 多模态图像理解评测集 | 14 个子任务 score | 0-shot | 多模态 base64 格式 | mme_gen_base64.py |

数据加载与 prompt 消息构造参照 InfoVQA：图片在前、文本在后；prompt 直接使用 Parquet 中的 `question` 原文，不追加额外回答指令。图片会根据真实格式编码为 JPEG 或 PNG 的 base64 data URL。


## 评测输出

评测完成后会输出：

- CLI、`mme.json` 和 summary 文件仅展示官方顺序的 14 个子任务 `score`，其中 `score = ACC + ACC+`；
- `<work_dir>/results/<model>/mme_results/mme_metrics.json` 按 `Perception` 和 `Cognition` 分组，详细保存各子任务的 `ACC`、`ACC+`、`score` 以及分组 `total_score`；
- `<work_dir>/results/<model>/mme_results/` 目录下按子任务生成的 14 个 txt 结果文件。

`mme_metrics.json` 的结构如下：

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

每个 txt 文件中的单行格式如下：

```text
图片名\t问题\t标准答案\t模型回答
```

同一 `question_id` 对应的两道问题会相邻写入；模型回答中的换行符和 Tab 会替换为空格。
