# MME

MME 是面向多模态大模型的图像理解评测集，包含 14 个感知与认知子任务。每张图像对应两道 Yes/No 问题；只有两题全部答对时，该图像才计入 ACC+。

## 数据路径

本适配默认从以下目录读取全部 parquet 分片：

```text
C:\需求\MME\MME\data
```

parquet 必须包含 `question_id`、`image`、`question`、`answer` 和 `category` 字段。数据加载与 prompt 消息构造遵循 InfoVQA 的图片在前、文本在后风格；图片字节会编码为 base64，prompt 直接使用 `question` 原文，不添加 InfoVQA 的额外回答指令。

## 配置

```python
from ais_bench.benchmark.configs.datasets.mme.mme_gen_base64 import mme_datasets as datasets
```

推理采用 `MMPromptTemplate`，并通过与图片真实格式匹配的 base64 data URL 传输 JPEG/PNG 图片。

## 评测输出

评测结果包含：

- 整体及每个子任务的 `ACC` 与 `ACC+`（百分数）；
- `Perception`、`Cognition` 和 `MME Score` 官方总分；
- `<work_dir>/results/<model>/mme_results/*.txt` 下的 14 个官方格式结果文件。

每个 txt 行格式为：

```text
图片名\t问题\t标准答案\t模型回答
```

同一 `question_id` 的两道问题会相邻写出，模型回答中的换行和 Tab 会替换为空格。
