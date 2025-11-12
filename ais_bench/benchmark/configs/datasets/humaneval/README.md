# HumanEval
中文 | [English](README_en.md)
## 数据集简介
OpenAI 发布的 HumanEval 数据集包含 164 个编程问题，每个问题都提供了函数签名、文档字符串、函数主体以及多个单元测试。这些问题均为手工编写，以确保它们不会出现在代码生成模型的训练集中。

> 🔗 数据集主页[https://huggingface.co/datasets/openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval)

⏰**注意**：数据集运行前请先安装依赖[extra.txt](../../../../../requirements/extra.txt)
```shell
# 需要处在最外层benchmark文件夹下，运行下列指令：
pip3 install -r requirements/extra.txt
```

## 数据集部署
- 可以从opencompass提供的链接🔗 [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/humaneval.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/humaneval.zip)下载数据集压缩包。
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/humaneval.zip
unzip humaneval.zip
rm humaneval.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree humaneval/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    humaneval
    └── human-eval-v2-20210705.jsonl
    ```

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|humaneval_gen_0_shot|humaneval数据集生成式任务|pass@1|0-shot|字符串格式|[humaneval_gen_0_shot.py](humaneval_gen_0_shot.py)|