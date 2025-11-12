# IFEval
中文 | [English](README_en.md)
## 数据集简介
IFEval是一个用于评估大语言模型（如GPT-4、PaLM 2等）指令遵循能力的数据集。随着大语言模型在自然语言任务中的广泛应用，模型的指令遵循能力成为一个重要的评估指标。

> 🔗 数据集主页[https://huggingface.co/datasets/google/IFEval](https://huggingface.co/datasets/google/IFEval)

⏰**注意**：数据集运行前请先安装依赖[extra.txt](../../../../../requirements/extra.txt)
```shell
# 需要处在最外层benchmark文件夹下，运行下列指令：
pip3 install -r requirements/extra.txt
```

## 数据集部署
- 可以从opencompass提供的链接🔗 [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/ifeval.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/ifeval.zip)下载数据集压缩包。
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/ifeval.zip
unzip ifeval.zip
rm ifeval.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree feval/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    ifeval
    └── input_data.jsonl
    ```

## 可用数据集任务
### ifeval_0_shot_gen_str
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|ifeval_0_shot_gen_str|ifeval数据集生成式任务|accuracy|0-shot|字符串格式|[ifeval_0_shot_gen_str.py](ifeval_0_shot_gen_str.py)|