# FewCLUE_tnews
中文 | [English](README_en.md)
## 数据集简介
该数据集任务是中文新闻分类任务。给定一条新闻文本，需要判断该新闻属于15个类别中的哪一个，包括农业新闻、旅游新闻、游戏新闻、科技类别公司新闻、体育类别新闻、初升高教育新闻、娱乐圈新闻、投资资讯、军事类别常识、车辆新闻、楼市新闻、环球不含中国类别新闻、书籍文化历史类别新闻、故事类别新闻、股票市场类别新闻等。

> 🔗 数据集主页链接[https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets/tnews](https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets/tnews)

## 数据集部署
- 可以从opencompass提供的汇总数据集链接🔗 [https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip)将压缩包中`data/FewCLUE/tnews`下的文件复制到`FewCLUE/tnews/`中
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip -d OpenCompassData-core-20240207
mkdir -p FewCLUE/tnews/
cp -r OpenCompassData-core-20240207/data/FewCLUE/tnews/* FewCLUE/tnews/
rm -r OpenCompassData-core-20240207/
rm -r OpenCompassData-core-20240207.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree FewCLUE/tnews`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    tnews/
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
|FewCLUE_tnews_ppl_0_shot_chat|FewCLUE_tnews数据集PPL任务|accuracy|0-shot|对话格式|`from ais_bench.benchmark.configs.datasets.FewCLUE_tnews.FewCLUE_tnews_ppl_0_shot_chat import tnews_datasets as datasets`|[FewCLUE_tnews_ppl_0_shot_chat.py](FewCLUE_tnews_ppl_0_shot_chat.py)|