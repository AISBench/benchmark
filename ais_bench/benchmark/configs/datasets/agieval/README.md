# AGIEval
中文 | [English](README_en.md)
## 数据集简介
AGIEval—是一个专为评估基础模型而设计的新型基准测试，其特别关注人类中心化的标准化考试场景，包括大学入学考试、法学院入学测试、数学竞赛以及律师资格考试等。

> 🔗 数据集主页链接[https://github.com/ruixiangcui/AGIEval](https://github.com/ruixiangcui/AGIEval)

## 数据集部署
- 可以从opencompass提供的汇总数据集链接🔗 [https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip)将压缩包中`data/AGIEval/data/v1`下的文件复制到`agieval/`中
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip -d OpenCompassData-core-20240207
mkdir agieval/
cp -r OpenCompassData-core-20240207/data/AGIEval/data/v1/* agieval/
rm -r OpenCompassData-core-20240207/
rm -r OpenCompassData-core-20240207.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree agieval/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    agieval/
    ├── aqua-rat.jsonl
    ├── gaokao-biology.jsonl
    ├── gaokao-chemistry.jsonl
    ├── gaokao-chinese.jsonl
    ├── gaokao-english.jsonl
    ├── gaokao-geography.jsonl
    ├── gaokao-history.jsonl
    ├── gaokao-mathcloze.jsonl
    ├── gaokao-mathqa.jsonl
    ├── gaokao-physics.jsonl
    ├── jec-qa-ca.jsonl
    ├── jec-qa-kd.jsonl
    ├── LICENSE
    ├── logiqa-en.jsonl
    ├── logiqa-zh.jsonl
    ├── lsat-ar.jsonl
    ├── lsat-lr.jsonl
    ├── lsat-rc.jsonl
    ├── math.jsonl
    ├── sat-en.jsonl
    ├── sat-en-without-passage.jsonl
    └── sat-math.jsonl
    ```
## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|配套文件导入方式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- | --- |
|agieval_gen_0_shot_chat_prompt|AGIEval数据集生成式任务，共包含21个子任务|accuracy|0-shot|对话格式|`from ais_bench.benchmark.configs.datasets.agieval.agieval_gen_0_shot_chat_prompt import agieval_datasets as datasets`|[agieval_gen_0_shot_chat_prompt.py](agieval_gen_0_shot_chat_prompt.py)|