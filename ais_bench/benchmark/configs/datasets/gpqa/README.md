# GPQA
中文 | [English](README_en.md)
## 数据集简介
GPQA 是一个包含选择题的问答数据集，其中的高难度问题由生物学、物理学和化学领域的专家编写并验证。当这些专家尝试回答自己专业领域之外的问题时（例如，一位物理学家回答化学问题），即便他们能无限制地使用谷歌搜索，且花费超过 30 分钟作答，答题准确率也仅有 34%。

> 🔗 数据集主页[https://github.com/idavidrein/gpqa](https://github.com/idavidrein/gpqa)

## 数据集部署
- 可以从opencompass提供的链接🔗 [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gpqa.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gpqa.zip)下载数据集压缩包。
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gpqa.zip
unzip gpqa.zip
rm gpqa.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree gpqa/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    gpqa
    ├── gpqa_diamond.csv
    ├── gpqa_experts.csv
    ├── gpqa_extended.csv
    ├── gpqa_main.csv
    └── license.txt
    ```
## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|配套文件导入方式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- | --- |
|gpqa_gen_0_shot_str|gpqa数据集生成式任务|accuracy(pass@1)|0-shot|字符串格式|`from ais_bench.benchmark.configs.datasets.gpqa.gpqa_gen_0_shot_str import gpqa_datasets as datasets`|[gpqa_gen_0_shot_str.py](gpqa_gen_0_shot_str.py)|
|gpqa_gen_0_shot_cot_chat_prompt|gpqa数据集生成式任务（对齐DeepSeek R1精度测试）|accuracy(pass@1)|0-shot|对话格式|`from ais_bench.benchmark.configs.datasets.gpqa.gpqa_gen_0_shot_cot_chat_prompt import gpqa_datasets as datasets`|[gpqa_gen_0_shot_cot_chat_prompt.py](gpqa_gen_0_shot_cot_chat_prompt.py)|
|gpqa_ppl_0_shot_str|gpqa数据集PPL任务|accuracy(pass@1)|0-shot|字符串格式|`from ais_bench.benchmark.configs.datasets.gpqa.gpqa_ppl_0_shot_str import gpqa_datasets as datasets`|[gpqa_ppl_0_shot_str.py](gpqa_ppl_0_shot_str.py)|