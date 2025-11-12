# BoolQ
中文 | [English](README_en.md)
## 数据集简介
BoolQ 是一个用于回答是非问题的问答数据集，包含 15942 个示例。这些问题是自然产生的 —— 它们是在无提示且不受限制的情况下生成的。每个示例都是由（问题、段落、答案）组成的三元组，页面标题作为可选的额外背景信息。

> 🔗 数据集主页链接[https://huggingface.co/datasets/google/boolq](https://huggingface.co/datasets/google/boolq)

## 数据集部署
- 可以从opencompass提供的链接🔗 [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/SuperGLUE.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/SuperGLUE.zip)下载数据集压缩包。
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/SuperGLUE.zip
unzip SuperGLUE.zip
rm SuperGLUE.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree SuperGLUE/BoolQ/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    BoolQ/
    ├── test.jsonl
    └── val.jsonl
    ```

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|SuperGLUE_BoolQ_gen_883d50_str|BoolQ数据集生成式任务|accuracy(naive_average)|0-shot|string|[SuperGLUE_BoolQ_gen_883d50_str.py](SuperGLUE_BoolQ_gen_883d50_str.py)|
|SuperGLUE_BoolQ_gen_0_shot_cot_str|BoolQ数据集生成式任务，prompt带逻辑链|accuracy(naive_average)|0-shot|string|[SuperGLUE_BoolQ_gen_0_shot_cot_str.py](SuperGLUE_BoolQ_gen_0_shot_cot_str.py)|
|SuperGLUE_BoolQ_gen_5_shot_str|BoolQ数据集生成式任务，few-shot|accuracy(naive_average)|5-shot|string|[SuperGLUE_BoolQ_gen_5_shot_str.py](SuperGLUE_BoolQ_gen_5_shot_str.py)|
|SuperGLUE_BoolQ_gen_0_shot_str|BoolQ数据集生成式任务，few-shot|accuracy(naive_average)|5-shot|string|[SuperGLUE_BoolQ_gen_0_shot_str.py](SuperGLUE_BoolQ_gen_0_shot_str.py)|