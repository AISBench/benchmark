# FewCLUE_cluewsc
中文 | [English](README_en.md)
## 数据集简介
Winograd Scheme Challenge（WSC）是一类代词消歧的任务，即判断句子中的代词指代的是哪个名词。题目以真假判别的方式出现，如：
句子：这时候放在[床]上[枕头]旁边的[手机]响了，我感到奇怪，因为欠费已被停机两个月，现在[它]突然响了。需要判断“它”指代的是“床”、“枕头”，还是“手机”？
从中国现当代作家文学作品中抽取，再经语言专家人工挑选、标注。

> 🔗 数据集主页链接[https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets/cluewsc](https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets/cluewsc)

## 数据集部署
- 可以从opencompass提供的汇总数据集链接🔗 [https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip)将压缩包中`data/FewCLUE/cluewsc`下的文件复制到`FewCLUE/cluewsc/`中
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip -d OpenCompassData-core-20240207
mkdir -p FewCLUE/cluewsc/
cp -r OpenCompassData-core-20240207/data/FewCLUE/cluewsc/* FewCLUE/cluewsc/
rm -r OpenCompassData-core-20240207/
rm -r OpenCompassData-core-20240207.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree FewCLUE/cluewsc`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    cluewsc/
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
|FewCLUE_cluewsc_ppl_0_shot_chat|FewCLUE_cluewsc数据集PPL任务|accuracy|0-shot|对话格式|[FewCLUE_cluewsc_ppl_0_shot_chat.py](FewCLUE_cluewsc_ppl_0_shot_chat.py)|