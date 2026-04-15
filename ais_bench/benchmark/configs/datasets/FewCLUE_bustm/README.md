# FewCLUE_bustm
中文 | [English](README_en.md)
## 数据集简介
该数据集是对话短文本语义匹配数据集，源于小布助手。它是OPPO为品牌手机和IoT设备自研的语音助手，为用户提供便捷对话式服务。意图识别是对话系统中的一个核心任务，而对话短文本语义匹配是意图识别的主流算法方案之一。要求根据短文本query-pair，预测它们是否属于同一语义。

> 🔗 数据集主页链接[https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets/bustm](https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets/bustm)

## 数据集部署
- 可以从opencompass提供的汇总数据集链接🔗 [https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip)将压缩包中`data/FewCLUE/bustm`下的文件复制到`FewCLUE/bustm/`中
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip -d OpenCompassData-core-20240207
mkdir -p FewCLUE/bustm/
cp -r OpenCompassData-core-20240207/data/FewCLUE/bustm/* FewCLUE/bustm/
rm -r OpenCompassData-core-20240207/
rm -r OpenCompassData-core-20240207.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree FewCLUE/bustm`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    bustm/
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
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|FewCLUE_bustm_ppl_0_shot_chat|FewCLUE_bustm数据集PPL任务|accuracy|0-shot|对话格式|[FewCLUE_bustm_ppl_0_shot_chat.py](FewCLUE_bustm_ppl_0_shot_chat.py)|