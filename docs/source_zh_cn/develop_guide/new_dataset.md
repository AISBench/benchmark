# 支持新的数据集和精度评估器

当前 AISBench 支持的数据类型包括：开源数据集、自定义数据集以及合成数据集。在适配新数据集前，建议先参考[自定义数据集](../advanced_tutorials/custom_dataset.md)和[合成数据集](../advanced_tutorials/synthetic_dataset.md)的使用说明，确认能否满足实际需要。

对于无法满足要求的数据集（例如数据集加载方式或精度计算规则与其他数据集存在明显差异），则需要进行适配。在开始前，建议先参考[prompt_template](../prompt/prompt_template.md)和[meta_template](../prompt/meta_template.md)的定义方法，了解 AISBench 对于 prompt 的构建方式、如何将原始数据转化为实际的模型输入，以及中间涉及的组件功能。

具体实现参考如下：

1. 在 `ais_bench/benchmark/datasets` 文件夹下新增数据集脚本 `mydataset.py`，该脚本需要包含：

   - **数据集及其加载方式**：需要定义一个 `MyDataset` 类，实现数据集加载方法 `load`。该方法为静态方法，需要返回 `datasets.Dataset` 类型的数据。这里我们使用 HuggingFace Dataset 作为数据集的统一接口，避免引入额外的逻辑。参考格式如下：

    ```python
    import datasets
    from .base import BaseDataset

    class MyDataset(BaseDataset):

        @staticmethod
        def load(**kwargs) -> datasets.Dataset:
            ... # 实现数据集加载逻辑
            data_list = ... # 数据集列表
            return datasets.Dataset.from_list(data_list)  # 转化数据集列表为 HuggingFace Dataset 对象
    ```

    新增数据集的类建议补充到[`__init__.py`](../../../ais_bench/benchmark/datasets/__init__.py)中，方便后续自动导入。

    具体示例可参考[Aime2024Dataset](../../../ais_bench/benchmark/datasets/aime2024.py)

    对于**多模态数据**，需要在 `load` 函数中采用格式化拼接的方式将 text、image、video、audio 数据拼接成一条数据。后续解析过程中会按照每个数据类型的标记还原并拼接到模型输入中。

    拼接格式示例：

    ```text
    <AIS_TEXT_START>{text}<AIS_CONTENT_TAG><AIS_IMAGE_START>{image}<AIS_CONTENT_TAG><AIS_VIDEO_START>{video}<AIS_CONTENT_TAG><AIS_AUDIO_START>{audio}<AIS_CONTENT_TAG>
    ```

    其中，`{text}`、`{image}`、`{video}`、`{audio}` 为数据集中的文本、图片、视频、音频内容。

    具体示例可参考[MMCustomDataset](../../../ais_bench/benchmark/datasets/mm_custom.py)

   - **（可选）自定义精度评估器**：如果 AISBench 已有的精度评估器不能满足需要，需要用户定义 `MyDatasetEvaluator` 类，实现评分方法 `score`。该方法需要根据输入的 `predictions` 和 `references` 列表，返回一个包含 metrics 及其对应 scores 的字典。由于一个数据集可能存在多种 metric，返回的字典应包含所有相关的评估指标。具体示例如下：

   ```python
   from typing import List
   from ais_bench.benchmark.openicl.icl_evaluator import BaseEvaluator

   class MyDatasetEvaluator(BaseEvaluator):

       def score(self, predictions: List, references: List) -> dict:
           # 实现评估逻辑
           # 返回格式：{"metric_name": score_value, ...}
           pass
   ```

   具体实现可参考[MATHEvaluator](../../../ais_bench/benchmark/datasets/math.py)

   - **（可选）自定义后处理方法**：如果 AISBench 已有的后处理方法不能满足需要，需要用户定义 `mydataset_postprocess` 方法，根据输入的字符串得到相应后处理的结果。该方法通常用于清理模型输出、提取答案等场景。具体示例如下：

   ```python
   def mydataset_postprocess(text: str) -> str:
       # 实现后处理逻辑，例如提取答案、清理格式等
       # 返回处理后的字符串
       pass
   ```

2. 在定义好数据集加载、评测以及数据后处理等方法之后，需要在配置目录[../ais_bench/benchmark/configs/datasets](../../../ais_bench/benchmark/configs/datasets/my_dataset)中新增以下配置my_dataset.py：

   ```python
   from ais_bench.benchmark.datasets import MyDataset, MyDatasetEvaluator, mydataset_postprocess

   # 精度评估配置
   mydataset_eval_cfg = dict(
       evaluator=dict(type=MyDatasetEvaluator),  # 自定义精度评估器类名
       pred_postprocessor=dict(type=mydataset_postprocess)  # 自定义数据后处理方法
   )

   # 数据集读取配置：根据数据集中每个样本的字段进行配置，用于填充 prompt_template
   mydataset_reader_cfg = dict(
       input_columns=["question"],  # 输入字段列表
       output_column="answer"       # 输出字段（标准答案）
   )

   # 推理配置
   mydataset_infer_cfg = dict(
       prompt_template=dict(
           # 提示词模板类名，根据数据类型进行配置：
           # - PromptTemplate: 纯文本输入
           # - MultiTurnPromptTemplate: 多轮对话输入
           # - MMPromptTemplate: 多模态输入
           type=PromptTemplate,
           template=dict(
               round=[
                   dict(
                       role="HUMAN",
                       prompt="{question}\nRemember to put your final answer within \\boxed{}.",
                   ),
               ],
           ),
       ),
       retriever=dict(type=ZeroRetriever),      # 检索器配置
       inferencer=dict(type=GenInferencer),     # 推理器配置
   )

   # 数据集配置列表
   mydataset_datasets = [
       dict(
           type=MyDataset,                    # 自定义数据集类名
           abbr='mydataset',                  # 数据集的唯一标识
           # ... 其他数据集初始化参数 ...
           reader_cfg=mydataset_reader_cfg,   # 数据集读取配置
           infer_cfg=mydataset_infer_cfg,     # 推理配置
           eval_cfg=mydataset_eval_cfg        # 精度评估配置
       )
   ]
   ```

    接着执行命令启动本地评测任务：

    ```bash
    ais_bench --models vllm_api_stream_chat --datasets my_dataset
    ```

3. 补充 README 文档

   在配置目录 `ais_bench/benchmark/configs/datasets/my_dataset/` 下创建 `README.md` 文件，用于说明数据集的部署和使用方法。README 应包含以下内容：

   - **数据集简介**：简要介绍数据集的基本信息、特点、用途等，并附上数据集主页链接（如果存在）。示例格式：

     ```markdown
     # MyDataset
     中文 | [English](README_en.md)
     ## 数据集简介
     MyDataset 是一个用于评估模型在 XXX 任务上表现的基准数据集。该数据集包含 XXX 个样本，涵盖 XXX 个不同类别。

     > 🔗 数据集主页链接[https://example.com/mydataset](https://example.com/mydataset)
     ```

   - **数据集部署**：详细说明数据集的下载和部署步骤，包括：
     - 数据集的下载链接或获取方式
     - 部署路径和目录结构要求
     - 部署步骤（建议提供可执行的命令示例）
     - 目录结构验证方法（建议使用 `tree` 命令展示预期的目录结构）

     示例格式：

     ```markdown
     ## 数据集部署
     - 可以从 XXX 提供的链接🔗 [https://example.com/mydataset.zip](https://example.com/mydataset.zip)下载数据集压缩包。
     - 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
     ```bash
     # linux服务器内，处于工具根路径下
     cd ais_bench/datasets
     wget https://example.com/mydataset.zip
     unzip mydataset.zip
     rm mydataset.zip
     ```

     - 在`{工具根路径}/ais_bench/datasets`目录下执行`tree mydataset/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。

         ```text
         mydataset
         ├── data
         │   └── ...
         └── ...
         ```

     如果数据集通过依赖包方式集成（如 Python 包），则说明安装步骤和环境要求：

     示例格式：

     ```markdown
     ## 数据集部署
     MyDataset数据集通过Python依赖包的方式集成，数据文件包含在 `mydataset-eval` 依赖包中，安装依赖后即可直接使用。

     ### 环境要求
     - **mydataset-eval** 依赖包（包含完整数据集）

     ### 安装步骤
     \`\`\`bash
     pip3 install mydataset-eval
     \`\`\`
     ```

   - **（可选）使用示例**：如果数据集有特殊的使用要求或配置方式，应提供详细的使用示例，包括：
     - 模型配置示例（如果数据集需要特定的模型类型或配置）
     - 执行测评的命令示例
     - 结果展示示例

   - **可用数据集任务**：以表格形式列出所有可用的数据集任务配置，表格应包含以下列：
     - 任务名称：数据集配置的标识符（用于 `--datasets` 参数）
     - 简介：任务的简要说明
     - 评估指标：使用的评估指标（如 accuracy、score 等）
     - few-shot：few-shot 示例数量（如 0-shot、3-shot、5-shot 等）
     - prompt格式：prompt 的格式类型（如字符串格式、对话格式等）
     - 对应源码配置文件路径：配置文件的相对路径

     示例格式：

     ```markdown
     ## 可用数据集任务
     |任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
     | --- | --- | --- | --- | --- | --- |
     |mydataset_gen_0_shot_str|MyDataset数据集生成式任务|accuracy|0-shot|字符串格式|[mydataset_gen_0_shot_str.py](mydataset_gen_0_shot_str.py)|
     |mydataset_gen_5_shot_str|MyDataset数据集生成式任务|accuracy|5-shot|字符串格式|[mydataset_gen_5_shot_str.py](mydataset_gen_5_shot_str.py)|
     ```

   - **（可选）数据集分类**：如果数据集包含多个子类别或测试场景，可以按照不同维度进行分类说明，例如：
     - 单独测试类别：列出各个子类别的配置
     - 测试组别：说明如何批量测试多个相关类别
     - 精确测试配置：说明如何指定特定测试用例进行精确测试

   - **（可选）使用建议**：提供数据集使用的建议和注意事项，帮助用户更好地使用数据集。

   具体示例可参考：
   - [C-Eval README](../../../ais_bench/benchmark/configs/datasets/ceval/README.md)
   - [BFCL README](../../../ais_bench/benchmark/configs/datasets/BFCL/README.md)
   - [BBH README](../../../ais_bench/benchmark/configs/datasets/bbh/README.md)
