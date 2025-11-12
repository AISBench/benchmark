# HellaSwag
中文 | [English](README_en.md)
## 数据集简介
HellaSwag是一个用于评估自然语言理解能力的基准数据集，主要用于测试模型在常识推理方面的表现。数据集包含多个选择题，要求模型从多个选项中选择最合理的答案。

> 🔗 数据集主页[https://huggingface.co/datasets/Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag)

## 数据集部署
- 可以从opencompass提供的链接🔗 [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/hellaswag.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/hellaswag.zip)下载数据集压缩包。
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/hellaswag.zip
unzip hellaswag.zip
rm hellaswag.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree hellaswag/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    hellaswag
    ├── hellaswag.jsonl
    ├── hellaswag_train_sampled25.jsonl
    └── hellaswag_val_contamination_annotations.json
    ```

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|hellaswag_gen_0_shot_chat_prompt|hellaswag数据集生成式任务|accuracy|0-shot|对话格式|[hellaswag_gen_0_shot_chat_prompt.py](hellaswag_gen_0_shot_chat_prompt.py)|
|hellaswag_gen_10_shot_chat_prompt|hellaswag数据集生成式任务|accuracy|10-shot|对话格式|[hellaswag_gen_10_shot_chat_prompt.py](hellaswag_gen_10_shot_chat_prompt.py)|
