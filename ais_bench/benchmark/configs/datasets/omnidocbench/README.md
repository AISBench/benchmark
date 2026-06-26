# OmniDocBench
中文 | [English](README_en.md)
## 数据集简介
OmniDocBench是一个针对真实场景下多样性文档解析评测集，具有以下特点：

- 文档类型多样：该评测集涉及1355个PDF页面，涵盖9种文档类型、4种排版类型和3种语言类型。覆盖面广，包含学术文献、财报、报纸、教材、手写笔记等；
- 标注信息丰富：包含15个block级别（文本段落、标题、表格等，总量超过20k）和4个Span级别（文本行、行内公式、角标等，总量超过80k）的文档元素的定位信息，以及每个元素区域的识别结果（文本Text标注，公式LaTeX标注，表格包含LaTeX和HTML两种类型的标注）。OmniDocBench还提供了各个文档组件的阅读顺序的标注。除此之外，在页面和block级别还包含多种属性标签，标注了5种页面属性标签、3种文本属性标签和6种表格属性标签。
- 标注质量高：经过人工筛选，智能标注，人工标注及全量专家质检和大模型质检，数据质量较高。
- 配套评测代码：设计端到端评测及单模块评测代码，保证评测的公平性及准确性。

> 🔗 数据集主页链接[https://huggingface.co/datasets/opendatalab/OmniDocBench](https://huggingface.co/datasets/opendatalab/OmniDocBench)

## 数据集部署
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/opendatalab/OmniDocBench
cd OmniDocBench
git checkout 91fe284bbfacfa687959ae3eb00846ca852aa907 # 注意必须是这个commit id版本的数据集
git lfs pull
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree OmniDocBench/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    OmniDocBench
    ├── images
    │   ├── PPT_1001115_eng_page_003.png
    │   └── PPT_1001115_eng_page_005.png
    │   # ......
    |
    └── OmniDocBench.json
    ```

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|omnidocbench_gen|OmniDocBench数据集生成式任务|accuracy (pass@1)|0-shot|字符串格式|[omnidocbench_gen.py](omnidocbench_gen.py)|

## 使用约束
- 当前仅支持Edit_dist指标（用于测评DeepSeek-OCR模型），其他指标暂不支持，overall为各个维度的Edit_dist评分的均值
- 对于该数据集的测评需安装额外的依赖：`pip3 install -r requirements/datasets/omnidocbench_dependencies.txt`