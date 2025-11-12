# TriviaQA
中文 | [English](README_en.md)
## 数据集简介
TriviaQA是一个阅读理解数据集，包含超过65万组"问题-答案-证据"三元组。该数据集包含9.5万道由 trivia 爱好者编写的问题-答案对，以及独立收集的佐证文档（平均每道问题6份），这些文档为问题解答提供了高质量的远程监督。

> 🔗 数据集主页[https://huggingface.co/datasets/mandarjoshi/trivia_qa](https://huggingface.co/datasets/mandarjoshi/trivia_qa)

## 数据集部署
- 可以从opencompass提供的链接🔗 [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/triviaqa.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/triviaqa.zip)下载数据集压缩包。
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/triviaqa.zip
unzip triviaqa.zip
rm triviaqa.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree triviaqa/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    triviaqa
    ├── trivia-dev.qa.csv
    ├── triviaqa-train.jsonl
    ├── triviaqa-validation.jsonl
    └── trivia-test.qa.csv
    ```

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|triviaqa_gen_5_shot_chat_prompt|TriviaQA数据集生成式任务|accuracy|5-shot|对话格式|[triviaqa_gen_5_shot_chat_prompt.py](triviaqa_gen_5_shot_chat_prompt.py)|

