# VocalSound
中文 | [English](README_en.md)
## 数据集简介
VocalSound是一个用于人类时声音识别的数据集，说话者包含了不同年龄、性别和国家，共有超过21000条wav格式的语音文件，覆盖了laughter（笑声）、sigh（叹息）、cough（咳嗽）、throat clearing（清嗓子）、sneeze（打喷嚏）、sniff（抽鼻子）等六种不同类型的声音。模型需要判断不同的语音文件属于哪一类的声音。

> 🔗 数据集主页链接[https://huggingface.co/datasets/maoxx241/audio_vocalsound_16k_subset](https://huggingface.co/datasets/maoxx241/audio_vocalsound_16k_subset)

## 数据集部署
- 可以从huggingface的数据集链接🔗 [https://huggingface.co/datasets/maoxx241/audio_vocalsound_16k_subset](https://huggingface.co/datasets/maoxx241/audio_vocalsound_16k_subset)中获取
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
git lfs install
git clone https://huggingface.co/datasets/maoxx241/audio_vocalsound_16k_subset
mv audio_vocalsound_16k_subset vocalsound
mv vocalsound/subset1/* vocalsound/
mv vocalsound/subset2/* vocalsound/
mv vocalsound/subset3/* vocalsound/
mv vocalsound/subset4/* vocalsound/
mv vocalsound/subset5/* vocalsound/
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree vocalsound/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    vocalsound
    ├── f0003_0_cough.wav
    ├── f0004_0_laughter.wav
    └── f0007_0_sneeze.wav
    # ......
    ```

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|配套文件导入方式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- | --- |
|vocalsound_gen|VocalSound数据集生成式任务，⚠️该数据集任务下会直接将音频路径传入服务化，需确保服务化支持该格式输入并且有权限访问该路径音频。|accuracy|0-shot|列表格式（包含文本和音频两种数据）|`from ais_bench.benchmark.configs.datasets.vocalsound.vocalsound_gen import vocalsound_datasets as datasets`|[vocalsound_gen.py](vocalsound_gen.py)|
|vocalsound_gen_base64|VocalSound数据集生成式任务，⚠️该数据集任务下，会将音频数据转化为base64格式再传入服务化，需确保服务化支持该输入格式数据。|accuracy|0-shot|列表格式（包含文本和音频两种数据）|`from ais_bench.benchmark.configs.datasets.vocalsound.vocalsound_gen_base64 import vocalsound_datasets as datasets`|[vocalsound_gen_base64.py](vocalsound_gen_base64.py)|