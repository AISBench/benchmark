# DROP
中文 | [English](README_en.md)
## 数据集简介
DROP 是一个通过众包和对抗性创建的、包含 96,000 个问题的基准测试。在该测试中，系统必须解析问题中的引用（可能涉及多个输入位置），并对这些引用执行离散操作（例如加法、计数或排序）。这些操作要求对段落内容的理解比之前的数据集更加全面和深入。

> 🔗 数据集主页[https://huggingface.co/datasets/ucinlp/drop](https://huggingface.co/datasets/ucinlp/drop)

## 数据集部署
- 可以从opencompass提供的链接🔗 [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/drop_simple_eval.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/drop_simple_eval.zip)下载数据集压缩包。
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/drop_simple_eval.zip
unzip drop_simple_eval.zip
rm drop_simple_eval.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree drop_simple_eval/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    drop_simple_eval
    └── dev.jsonl
    ```

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|配套文件导入方式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- | --- |
|drop_gen_0_shot_str|drop数据集生成式任务|accuracy(pass@1)|0-shot|字符串格式|`from ais_bench.benchmark.configs.datasets.drop.drop_gen_0_shot_str import drop_datasets as datasets`|[drop_gen_0_shot_str.py](drop_gen_0_shot_str.py)|
|drop_gen_3_shot_str|drop数据集生成式任务|accuracy(pass@1)|3-shot|字符串格式|`from ais_bench.benchmark.configs.datasets.drop.drop_gen_3_shot_str import drop_datasets as datasets`|[drop_gen_3_shot_str.py](drop_gen_3_shot_str.py)|
