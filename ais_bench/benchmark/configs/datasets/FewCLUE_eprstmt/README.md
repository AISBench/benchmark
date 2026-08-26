# FewCLUE_eprstmt
中文 | [English](README_en.md)
## 数据集简介
该数据集任务是中文电商产品评论的情感极性分类任务。给定一条产品评论，需要判断该评论的情感倾向是积极还是消极。

> 🔗 数据集主页链接[https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets/eprstmt](https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets/eprstmt)

## 数据集部署
- 可以从opencompass提供的汇总数据集链接🔗 [https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip)将压缩包中`data/FewCLUE/eprstmt`下的文件复制到`FewCLUE/eprstmt/`中
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip -d OpenCompassData-core-20240207
mkdir -p FewCLUE/eprstmt/
cp -r OpenCompassData-core-20240207/data/FewCLUE/eprstmt/* FewCLUE/eprstmt/
rm -r OpenCompassData-core-20240207/
rm -r OpenCompassData-core-20240207.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree FewCLUE/eprstmt`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    eprstmt/
    ├── dev_0.json
    ├── dev_1.json
    ├── dev_2.json
    ├── dev_3.json
    ├── dev_4.json
    ├── dev_few_all.json
    ├── test.json
    ├── test_public.json
    ├── train_0.json
    ├── train_1.json
    ├── train_2.json
    ├── train_3.json
    ├── train_4.json
    ├── train_few_all.json
    └── unlabeled.json
    ```

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|配套文件导入方式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- | --- |
|FewCLUE_eprstmt_ppl_0_shot_chat|FewCLUE_eprstmt数据集PPL任务|accuracy|0-shot|对话格式|`from ais_bench.benchmark.configs.datasets.FewCLUE_eprstmt.FewCLUE_eprstmt_ppl_0_shot_chat import eprstmt_datasets as datasets`|[FewCLUE_eprstmt_ppl_0_shot_chat.py](FewCLUE_eprstmt_ppl_0_shot_chat.py)|

