# 支持新的推理器（Inferencer）

推理器（Inferencer）是 AISBench 中负责执行模型推理的核心组件，它根据不同的模型类型（API 模型或本地模型）采用不同的推理方式。在适配新的推理器前，建议先参考 [prompt_template](../prompt/prompt_template.md) 和 [meta_template](../prompt/meta_template.md) 的定义方法，了解 AISBench 对于 prompt 的构建方式。

目前 AISBench 已经支持的推理器类型如下：

- **GenInferencer**：用于生成式任务的推理器，支持 API 模型和本地模型
- **MultiTurnGenInferencer**：用于多轮对话任务的推理器，支持 API 模型和本地模型
- **PPLInferencer**：用于困惑度（Perplexity）评估的推理器

针对某些特殊的推理场景或自定义需求，通常需要实现自定义推理器。根据调用的模型类型，推理器需要实现不同的接口：

- **API 模型**：需要实现 `do_request` 异步方法，通过 HTTP 请求调用服务化模型
- **本地模型**：需要实现 `batch_inference` 同步方法，直接调用本地模型进行批量推理

## 新增 API 模型推理器

新增基于 API 模型的推理器，需要在 `ais_bench/benchmark/openicl/icl_inferencer` 下新建 `my_custom_api_inferencer.py` 文件，继承 `BaseApiInferencer`，并根据使用场景实现对应的功能接口。当前支持拓展的接口如下：

- **（必需）`do_request`**：执行单个推理请求，用于 API 模型的异步推理
- **（必需）`get_data_list`**：从 retriever 获取数据列表，用于构建推理数据

```python
from multiprocessing import BoundedSemaphore
from typing import List, Optional
import uuid
import copy
import aiohttp

from ais_bench.benchmark.models.output import RequestOutput
from ais_bench.benchmark.registry import ICL_INFERENCERS
from ais_bench.benchmark.openicl.icl_retriever import BaseRetriever
from ais_bench.benchmark.openicl.icl_inferencer.icl_base_api_inferencer import BaseApiInferencer
from ais_bench.benchmark.openicl.icl_inferencer.output_handler.gen_inferencer_output_handler import GenInferencerOutputHandler


@ICL_INFERENCERS.register_module()
class MyCustomApiInferencer(BaseApiInferencer):
    """自定义 API 模型推理器类。

    Attributes:
        model_cfg: 模型配置
        batch_size (:obj:`int`, optional): 批处理大小
        output_json_filepath (:obj:`str`, optional): 输出 JSON 文件路径
        save_every (:obj:`int`, optional): 每处理多少个样本保存一次中间结果
    """

    def __init__(
        self,
        model_cfg,
        batch_size: Optional[int] = 1,
        mode: Optional[str] = "infer",
        output_json_filepath: Optional[str] = "./icl_inference_output",
        save_every: Optional[int] = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            model_cfg=model_cfg,
            batch_size=batch_size,
            mode=mode,
            output_json_filepath=output_json_filepath,
            save_every=save_every,
            **kwargs,
        )

        # 初始化输出处理器
        self.output_handler = GenInferencerOutputHandler(
            perf_mode=self.perf_mode,
            save_every=self.save_every
        )

    async def do_request(
        self,
        data: dict,
        token_bucket: BoundedSemaphore,
        session: aiohttp.ClientSession
    ):
        """执行单个推理请求。

        Args:
            data: 包含请求数据的字典，通常包含以下字段：
                - prompt: 输入提示词
                - index: 数据索引
                - data_abbr: 数据集标识
                - max_out_len: 最大输出长度
                - gold: 标准答案（可选）
            token_bucket: 用于限流的信号量
            session: HTTP 会话对象
        """
        data = copy.deepcopy(data)
        index = data.pop("index")
        input = data.pop("prompt")
        data_abbr = data.pop("data_abbr")
        max_out_len = data.pop("max_out_len")
        gold = data.pop("gold", None)

        # 生成唯一标识
        uid = str(uuid.uuid4()).replace("-", "")
        output = RequestOutput(self.perf_mode)
        output.uuid = uid

        # 更新状态计数器
        await self.status_counter.post()

        # 调用模型进行推理
        await self.model.generate(input, max_out_len, output, session=session, **data)

        # 更新状态
        if output.success:
            await self.status_counter.rev()
        else:
            await self.status_counter.failed()
        await self.status_counter.finish()
        await self.status_counter.case_finish()

        # 报告结果到输出处理器
        await self.output_handler.report_cache_info(index, input, output, data_abbr, gold)

    def get_data_list(
        self,
        retriever: BaseRetriever,
    ) -> List:
        """从 retriever 获取数据列表。

        Args:
            retriever: 检索器实例，用于获取数据和生成 prompt

        Returns:
            数据列表，每个元素是一个字典，包含推理所需的信息
        """
        data_abbr = retriever.dataset.abbr
        ice_idx_list = retriever.retrieve()
        prompt_list = []

        # 为每个样本生成 prompt
        for idx, ice_idx in enumerate(ice_idx_list):
            ice = retriever.generate_ice(ice_idx)
            prompt = retriever.generate_prompt_for_generate_task(
                idx,
                ice,
                gen_field_replace_token=self.gen_field_replace_token if hasattr(self, 'gen_field_replace_token') else "",
            )
            # 解析模板
            parsed_prompt = self.model.parse_template(prompt, mode="gen")
            prompt_list.append(parsed_prompt)

        self.logger.info(f"Apply ice template finished")

        # 获取标准答案
        gold_ans = retriever.get_gold_ans()

        # 构建数据列表
        data_list = []
        for index, prompt in enumerate(prompt_list):
            data_list.append(
                {
                    "prompt": prompt,
                    "data_abbr": data_abbr,
                    "index": index,
                    "max_out_len": self.model.max_out_len,
                }
            )

        # 添加标准答案
        if gold_ans is not None:
            for index, gold in enumerate(gold_ans):
                data_list[index]["gold"] = gold

        # 数据集指定的 max_out_len 具有最高优先级
        max_out_lens = retriever.dataset_reader.get_max_out_len()
        if max_out_lens is not None:
            self.logger.warning("Dataset-specified max_out_len has highest priority, use dataset-specified max_out_len")
            for index, max_out_len in enumerate(max_out_lens):
                data_list[index]["max_out_len"] = max_out_len if max_out_len else self.model.max_out_len

        return data_list
```

新增推理器类建议补充到[`__init__.py`](../../../ais_bench/benchmark/openicl/icl_inferencer/__init__.py)中，方便后续自动导入。

详细实现可参考：[GenInferencer](../../../ais_bench/benchmark/openicl/icl_inferencer/icl_gen_inferencer.py)

## 新增本地模型推理器

新增基于本地模型的推理器，需要在 `ais_bench/benchmark/openicl/icl_inferencer` 下新建 `my_custom_local_inferencer.py` 文件，继承 `BaseLocalInferencer`，并根据使用场景实现对应的功能接口。当前支持拓展的接口如下：

- **（必需）`batch_inference`**：执行批量推理，用于本地模型的同步推理
- **（必需）`get_data_list`**：从 retriever 获取数据列表，用于构建推理数据

```python
from typing import List, Optional
from torch.utils.data import DataLoader

from ais_bench.benchmark.registry import ICL_INFERENCERS
from ais_bench.benchmark.openicl.icl_retriever import BaseRetriever
from ais_bench.benchmark.openicl.icl_inferencer.icl_base_local_inferencer import BaseLocalInferencer
from ais_bench.benchmark.openicl.icl_inferencer.output_handler.gen_inferencer_output_handler import GenInferencerOutputHandler


@ICL_INFERENCERS.register_module()
class MyCustomLocalInferencer(BaseLocalInferencer):
    """自定义本地模型推理器类。

    Attributes:
        model_cfg: 模型配置
        batch_size (:obj:`int`, optional): 批处理大小
        output_json_filepath (:obj:`str`, optional): 输出 JSON 文件路径
        save_every (:obj:`int`, optional): 每处理多少个样本保存一次中间结果
    """

    def __init__(
        self,
        model_cfg,
        batch_size: Optional[int] = 1,
        output_json_filepath: Optional[str] = "./icl_inference_output",
        save_every: Optional[int] = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            model_cfg=model_cfg,
            batch_size=batch_size,
            output_json_filepath=output_json_filepath,
        )

        self.save_every = save_every

        # 初始化输出处理器
        self.output_handler = GenInferencerOutputHandler(
            perf_mode=False,  # 本地推理器通常不支持性能模式
            save_every=self.save_every
        )

    def batch_inference(
        self,
        datum: dict,
    ) -> None:
        """执行批量推理。

        Args:
            datum: 包含批量数据的字典，通常包含以下字段：
                - prompt: 输入提示词列表
                - index: 数据索引列表
                - data_abbr: 数据集标识列表
                - max_out_len: 最大输出长度列表
                - gold: 标准答案列表（可选）
        """
        indexs = datum.pop("index")
        inputs = datum.pop("prompt")
        data_abbrs = datum.pop("data_abbr")
        max_out_lens = datum.pop("max_out_len")
        golds = datum.pop("gold", [None] * len(inputs))

        # 调用本地模型进行批量推理
        # 本地模型使用模型配置中统一的 max_out_len
        outputs = self.model.generate(inputs, self.model.max_out_len, **datum)

        # 处理每个输出结果
        for index, input, output, data_abbr, gold in zip(
            indexs, inputs, outputs, data_abbrs, golds
        ):
            self.output_handler.report_cache_info_sync(
                index, input, output, data_abbr, gold
            )

    def get_data_list(
        self,
        retriever: BaseRetriever,
    ) -> List:
        """从 retriever 获取数据列表。

        Args:
            retriever: 检索器实例，用于获取数据和生成 prompt

        Returns:
            数据列表，每个元素是一个字典，包含推理所需的信息
        """
        data_abbr = retriever.dataset.abbr
        ice_idx_list = retriever.retrieve()
        prompt_list = []

        # 为每个样本生成 prompt
        for idx, ice_idx in enumerate(ice_idx_list):
            ice = retriever.generate_ice(ice_idx)
            prompt = retriever.generate_prompt_for_generate_task(
                idx,
                ice,
                gen_field_replace_token=self.gen_field_replace_token if hasattr(self, 'gen_field_replace_token') else "",
            )
            # 解析模板
            parsed_prompt = self.model.parse_template(prompt, mode="gen")
            prompt_list.append(parsed_prompt)

        self.logger.info(f"Apply ice template finished")

        # 获取标准答案
        gold_ans = retriever.get_gold_ans()

        # 构建数据列表
        data_list = []
        for index, prompt in enumerate(prompt_list):
            data_list.append(
                {
                    "prompt": prompt,
                    "data_abbr": data_abbr,
                    "index": index,
                    "max_out_len": self.model.max_out_len,
                }
            )

        # 添加标准答案
        if gold_ans is not None:
            for index, gold in enumerate(gold_ans):
                data_list[index]["gold"] = gold

        # 数据集指定的 max_out_len 具有最高优先级
        max_out_lens = retriever.dataset_reader.get_max_out_len()
        if max_out_lens is not None:
            self.logger.warning("Dataset-specified max_out_len has highest priority, use dataset-specified max_out_len")
            for index, max_out_len in enumerate(max_out_lens):
                data_list[index]["max_out_len"] = max_out_len if max_out_len else self.model.max_out_len

        return data_list
```

新增推理器类建议补充到[`__init__.py`](../../../ais_bench/benchmark/openicl/icl_inferencer/__init__.py)中，方便后续自动导入。

详细实现可参考：[GenInferencer](../../../ais_bench/benchmark/openicl/icl_inferencer/icl_gen_inferencer.py)

## 同时支持 API 模型和本地模型的推理器

如果推理器需要同时支持 API 模型和本地模型，可以同时继承 `BaseApiInferencer` 和 `BaseLocalInferencer`，并实现两个基类的必需方法。这样同一个推理器类可以用于两种类型的模型。

```python
from ais_bench.benchmark.registry import ICL_INFERENCERS
from ais_bench.benchmark.openicl.icl_inferencer.icl_base_api_inferencer import BaseApiInferencer
from ais_bench.benchmark.openicl.icl_inferencer.icl_base_local_inferencer import BaseLocalInferencer


@ICL_INFERENCERS.register_module()
class MyCustomInferencer(BaseApiInferencer, BaseLocalInferencer):
    """同时支持 API 模型和本地模型的自定义推理器。

    该类同时继承 BaseApiInferencer 和 BaseLocalInferencer，
    需要实现两个基类的必需方法。
    """

    def __init__(
        self,
        model_cfg,
        batch_size: Optional[int] = 1,
        mode: Optional[str] = "infer",
        output_json_filepath: Optional[str] = "./icl_inference_output",
        save_every: Optional[int] = 1,
        **kwargs,
    ) -> None:
        # 调用两个基类的初始化方法
        BaseApiInferencer.__init__(
            self,
            model_cfg=model_cfg,
            batch_size=batch_size,
            mode=mode,
            output_json_filepath=output_json_filepath,
            save_every=save_every,
            **kwargs,
        )

        # 初始化输出处理器
        self.output_handler = GenInferencerOutputHandler(
            perf_mode=self.perf_mode,
            save_every=self.save_every
        )

    async def do_request(self, data: dict, token_bucket: BoundedSemaphore, session: aiohttp.ClientSession):
        """API 模型的推理方法（必需）"""
        # 实现 API 模型的推理逻辑
        pass

    def batch_inference(self, datum: dict) -> None:
        """本地模型的推理方法（必需）"""
        # 实现本地模型的推理逻辑
        pass

    def get_data_list(self, retriever: BaseRetriever) -> List:
        """获取数据列表（必需）"""
        # 实现数据列表获取逻辑
        pass
```

详细实现可参考：[GenInferencer](../../../ais_bench/benchmark/openicl/icl_inferencer/icl_gen_inferencer.py)

## 在配置文件中使用自定义推理器

定义好自定义推理器后，需要在数据集配置文件中使用它。在 `ais_bench/benchmark/configs/datasets` 下的相应配置文件中，将 `infer_cfg` 中的 `inferencer` 类型设置为自定义推理器类：

```python
from ais_bench.benchmark.openicl.icl_inferencer import MyCustomInferencer
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever

# 推理配置
mydataset_infer_cfg = dict(
    prompt_template=dict(
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
    inferencer=dict(type=MyCustomInferencer), # 自定义推理器配置
)

# 数据集配置列表
mydataset_datasets = [
    dict(
        type=MyDataset,                    # 自定义数据集类名
        # ... 其他数据集初始化参数 ...
        reader_cfg=mydataset_reader_cfg,   # 数据集读取配置
        infer_cfg=mydataset_infer_cfg,     # 推理配置（包含自定义推理器）
        eval_cfg=mydataset_eval_cfg        # 精度评估配置
    )
]
```

## 注意事项

1. **注册装饰器**：自定义推理器必须使用 `@ICL_INFERENCERS.register_module()` 装饰器进行注册，才能被配置系统识别。

2. **输出处理器**：根据实际需求选择合适的输出处理器，常用的有：
   - `GenInferencerOutputHandler`：用于生成式任务的输出处理
   - `PPLInferencerOutputHandler`：用于困惑度评估的输出处理

3. **状态管理**：对于 API 模型推理器，需要注意：
   - 使用 `status_counter` 来跟踪请求状态（post、rev、failed、finish、case_finish）
   - 在 `do_request` 方法中正确更新状态计数器

4. **错误处理**：在推理过程中应该妥善处理异常情况，确保输出结果中包含错误信息，便于后续分析和调试。

5. **性能模式**：如果推理器需要支持性能测评（`mode="perf"`），需要确保：
   - API 模型推理器必须实现 `parse_stream_response` 接口（在模型类中）
   - 正确设置 `perf_mode` 标志
   - 使用 `RequestOutput` 来保存性能相关的指标

6. **数据格式**：`get_data_list` 方法返回的数据列表中的每个字典必须包含以下必需字段：
   - `prompt`：输入提示词
   - `index`：数据索引
   - `data_abbr`：数据集标识
   - `max_out_len`：最大输出长度
   - `gold`：标准答案（可选）
