# HumanEvalx
中文 | [English](README_en.md)
## 数据集简介
HumanEval-X 是由清华大学 KEG 实验室 THUDM 提供的一套多语言代码生成模型的评价标准。它包含 820 个高质量手写样本，覆盖 Python、C++、Java、JavaScript 和 Go 语言。

> 🔗 数据集主页[https://huggingface.co/datasets/THUDM/humaneval-x](https://huggingface.co/datasets/THUDM/humaneval-x)

⏰**注意**：数据集运行前请先安装依赖[extra.txt](../../../../../requirements/extra.txt)
```shell
# 需要处在最外层benchmark文件夹下，运行下列指令：
pip3 install -r requirements/extra.txt
```

## 数据集部署
- 可以从opencompass提供的链接🔗 [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/humanevalx.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/humanevalx.zip)下载数据集压缩包。
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/humanevalx.zip
unzip humanevalx.zip
rm humanevalx.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree humanevalx/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    humanevalx
    └── humanevalx_cpp.jsonl
    └── humanevalx_go.jsonl
    └── humanevalx_java.jsonl
    └── humanevalx_js.jsonl
    └── humanevalx_python.jsonl
    ```

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|配套文件导入方式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- | --- |
|humanevalx_gen_0_shot|humanevalx数据集生成式任务|pass@1|0-shot|字符串格式|`from ais_bench.benchmark.configs.datasets.humanevalx.humanevalx_gen_0_shot import humanevalx_datasets as datasets`|[humanevalx_gen_0_shot.py](humanevalx_gen_0_shot.py)|