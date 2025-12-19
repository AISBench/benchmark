# Supporting New Inferencers

Inferencer is the core component in AISBench responsible for executing model inference. It adopts different inference methods according to different model types (API models or local models). Before adapting a new inferencer, it is recommended to first refer to the definition methods of [prompt_template](../prompt/prompt_template.md) and [meta_template](../prompt/meta_template.md) to understand how AISBench constructs prompts.

Currently, AISBench supports the following inferencer types:

- **GenInferencer**: Inferencer for generative tasks, supporting API models and local models
- **MultiTurnGenInferencer**: Inferencer for multi-turn dialogue tasks, supporting API models and local models
- **PPLInferencer**: Inferencer for Perplexity evaluation

For certain special inference scenarios or custom requirements, it is usually necessary to implement custom inferencers. According to the model type being called, inferencers need to implement different interfaces:

- **API Models**: Need to implement the `do_request` async method, calling service models through HTTP requests
- **Local Models**: Need to implement the `batch_inference` sync method, directly calling local models for batch inference

## Adding API Model Inferencers

To add an inferencer based on API models, create a new file `my_custom_api_inferencer.py` in `ais_bench/benchmark/openicl/icl_inferencer`, inherit from `BaseApiInferencer`, and implement the corresponding functional interfaces according to usage scenarios. The currently supported extensible interfaces are as follows:

- **(Required) `do_request`**: Execute a single inference request, used for async inference of API models
- **(Required) `get_data_list`**: Get data list from retriever, used to construct inference data

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
    """Custom API model inferencer class.

    Attributes:
        model_cfg: Model configuration
        batch_size (:obj:`int`, optional): Batch size
        output_json_filepath (:obj:`str`, optional): Output JSON file path
        save_every (:obj:`int`, optional): Save intermediate results every N samples
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

        # Initialize output handler
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
        """Execute a single inference request.

        Args:
            data: Dictionary containing request data, usually includes the following fields:
                - prompt: Input prompt
                - index: Data index
                - data_abbr: Dataset identifier
                - max_out_len: Maximum output length
                - gold: Ground truth (optional)
            token_bucket: Semaphore for rate limiting
            session: HTTP session object
        """
        data = copy.deepcopy(data)
        index = data.pop("index")
        input = data.pop("prompt")
        data_abbr = data.pop("data_abbr")
        max_out_len = data.pop("max_out_len")
        gold = data.pop("gold", None)

        # Generate unique identifier
        uid = str(uuid.uuid4()).replace("-", "")
        output = RequestOutput(self.perf_mode)
        output.uuid = uid

        # Update status counter
        await self.status_counter.post()

        # Call model for inference
        await self.model.generate(input, max_out_len, output, session=session, **data)

        # Update status
        if output.success:
            await self.status_counter.rev()
        else:
            await self.status_counter.failed()
        await self.status_counter.finish()
        await self.status_counter.case_finish()

        # Report results to output handler
        await self.output_handler.report_cache_info(index, input, output, data_abbr, gold)

    def get_data_list(
        self,
        retriever: BaseRetriever,
    ) -> List:
        """Get data list from retriever.

        Args:
            retriever: Retriever instance, used to get data and generate prompts

        Returns:
            Data list, each element is a dictionary containing information needed for inference
        """
        data_abbr = retriever.dataset.abbr
        ice_idx_list = retriever.retrieve()
        prompt_list = []

        # Generate prompt for each sample
        for idx, ice_idx in enumerate(ice_idx_list):
            ice = retriever.generate_ice(ice_idx)
            prompt = retriever.generate_prompt_for_generate_task(
                idx,
                ice,
                gen_field_replace_token=self.gen_field_replace_token if hasattr(self, 'gen_field_replace_token') else "",
            )
            # Parse template
            parsed_prompt = self.model.parse_template(prompt, mode="gen")
            prompt_list.append(parsed_prompt)

        self.logger.info(f"Apply ice template finished")

        # Get ground truth
        gold_ans = retriever.get_gold_ans()

        # Build data list
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

        # Add ground truth
        if gold_ans is not None:
            for index, gold in enumerate(gold_ans):
                data_list[index]["gold"] = gold

        # Dataset-specified max_out_len has highest priority
        max_out_lens = retriever.dataset_reader.get_max_out_len()
        if max_out_lens is not None:
            self.logger.warning("Dataset-specified max_out_len has highest priority, use dataset-specified max_out_len")
            for index, max_out_len in enumerate(max_out_lens):
                data_list[index]["max_out_len"] = max_out_len if max_out_len else self.model.max_out_len

        return data_list
```

It is recommended to add the new inferencer class to [`__init__.py`](../../../ais_bench/benchmark/openicl/icl_inferencer/__init__.py) for convenient automatic import later.

For detailed implementation, refer to: [GenInferencer](../../../ais_bench/benchmark/openicl/icl_inferencer/icl_gen_inferencer.py)

## Adding Local Model Inferencers

To add an inferencer based on local models, create a new file `my_custom_local_inferencer.py` in `ais_bench/benchmark/openicl/icl_inferencer`, inherit from `BaseLocalInferencer`, and implement the corresponding functional interfaces according to usage scenarios. The currently supported extensible interfaces are as follows:

- **(Required) `batch_inference`**: Execute batch inference, used for sync inference of local models
- **(Required) `get_data_list`**: Get data list from retriever, used to construct inference data

```python
from typing import List, Optional
from torch.utils.data import DataLoader

from ais_bench.benchmark.registry import ICL_INFERENCERS
from ais_bench.benchmark.openicl.icl_retriever import BaseRetriever
from ais_bench.benchmark.openicl.icl_inferencer.icl_base_local_inferencer import BaseLocalInferencer
from ais_bench.benchmark.openicl.icl_inferencer.output_handler.gen_inferencer_output_handler import GenInferencerOutputHandler


@ICL_INFERENCERS.register_module()
class MyCustomLocalInferencer(BaseLocalInferencer):
    """Custom local model inferencer class.

    Attributes:
        model_cfg: Model configuration
        batch_size (:obj:`int`, optional): Batch size
        output_json_filepath (:obj:`str`, optional): Output JSON file path
        save_every (:obj:`int`, optional): Save intermediate results every N samples
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

        # Initialize output handler
        self.output_handler = GenInferencerOutputHandler(
            perf_mode=False,  # Local inferencers usually don't support performance mode
            save_every=self.save_every
        )

    def batch_inference(
        self,
        datum: dict,
    ) -> None:
        """Execute batch inference.

        Args:
            datum: Dictionary containing batch data, usually includes the following fields:
                - prompt: Input prompt list
                - index: Data index list
                - data_abbr: Dataset identifier list
                - max_out_len: Maximum output length list
                - gold: Ground truth list (optional)
        """
        indexs = datum.pop("index")
        inputs = datum.pop("prompt")
        data_abbrs = datum.pop("data_abbr")
        max_out_lens = datum.pop("max_out_len")
        golds = datum.pop("gold", [None] * len(inputs))

        # Call local model for batch inference
        # Local models use unified max_out_len from model configuration
        outputs = self.model.generate(inputs, self.model.max_out_len, **datum)

        # Process each output result
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
        """Get data list from retriever.

        Args:
            retriever: Retriever instance, used to get data and generate prompts

        Returns:
            Data list, each element is a dictionary containing information needed for inference
        """
        data_abbr = retriever.dataset.abbr
        ice_idx_list = retriever.retrieve()
        prompt_list = []

        # Generate prompt for each sample
        for idx, ice_idx in enumerate(ice_idx_list):
            ice = retriever.generate_ice(ice_idx)
            prompt = retriever.generate_prompt_for_generate_task(
                idx,
                ice,
                gen_field_replace_token=self.gen_field_replace_token if hasattr(self, 'gen_field_replace_token') else "",
            )
            # Parse template
            parsed_prompt = self.model.parse_template(prompt, mode="gen")
            prompt_list.append(parsed_prompt)

        self.logger.info(f"Apply ice template finished")

        # Get ground truth
        gold_ans = retriever.get_gold_ans()

        # Build data list
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

        # Add ground truth
        if gold_ans is not None:
            for index, gold in enumerate(gold_ans):
                data_list[index]["gold"] = gold

        # Dataset-specified max_out_len has highest priority
        max_out_lens = retriever.dataset_reader.get_max_out_len()
        if max_out_lens is not None:
            self.logger.warning("Dataset-specified max_out_len has highest priority, use dataset-specified max_out_len")
            for index, max_out_len in enumerate(max_out_lens):
                data_list[index]["max_out_len"] = max_out_len if max_out_len else self.model.max_out_len

        return data_list
```

It is recommended to add the new inferencer class to [`__init__.py`](../../../ais_bench/benchmark/openicl/icl_inferencer/__init__.py) for convenient automatic import later.

For detailed implementation, refer to: [GenInferencer](../../../ais_bench/benchmark/openicl/icl_inferencer/icl_gen_inferencer.py)

## Inferencers Supporting Both API Models and Local Models

If an inferencer needs to support both API models and local models, it can inherit from both `BaseApiInferencer` and `BaseLocalInferencer`, and implement the required methods of both base classes. This way, the same inferencer class can be used for both types of models.

```python
from ais_bench.benchmark.registry import ICL_INFERENCERS
from ais_bench.benchmark.openicl.icl_inferencer.icl_base_api_inferencer import BaseApiInferencer
from ais_bench.benchmark.openicl.icl_inferencer.icl_base_local_inferencer import BaseLocalInferencer


@ICL_INFERENCERS.register_module()
class MyCustomInferencer(BaseApiInferencer, BaseLocalInferencer):
    """Custom inferencer supporting both API models and local models.

    This class inherits from both BaseApiInferencer and BaseLocalInferencer,
    and needs to implement the required methods of both base classes.
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
        # Call initialization methods of both base classes
        BaseApiInferencer.__init__(
            self,
            model_cfg=model_cfg,
            batch_size=batch_size,
            mode=mode,
            output_json_filepath=output_json_filepath,
            save_every=save_every,
            **kwargs,
        )

        # Initialize output handler
        self.output_handler = GenInferencerOutputHandler(
            perf_mode=self.perf_mode,
            save_every=self.save_every
        )

    async def do_request(self, data: dict, token_bucket: BoundedSemaphore, session: aiohttp.ClientSession):
        """API model inference method (required)"""
        # Implement API model inference logic
        pass

    def batch_inference(self, datum: dict) -> None:
        """Local model inference method (required)"""
        # Implement local model inference logic
        pass

    def get_data_list(self, retriever: BaseRetriever) -> List:
        """Get data list (required)"""
        # Implement data list retrieval logic
        pass
```

For detailed implementation, refer to: [GenInferencer](../../../ais_bench/benchmark/openicl/icl_inferencer/icl_gen_inferencer.py)

## Using Custom Inferencers in Configuration Files

After defining a custom inferencer, it needs to be used in the dataset configuration file. In the corresponding configuration file under `ais_bench/benchmark/configs/datasets`, set the `inferencer` type in `infer_cfg` to the custom inferencer class:

```python
from ais_bench.benchmark.openicl.icl_inferencer import MyCustomInferencer
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever

# Inference configuration
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
    retriever=dict(type=ZeroRetriever),      # Retriever configuration
    inferencer=dict(type=MyCustomInferencer), # Custom inferencer configuration
)

# Dataset configuration list
mydataset_datasets = [
    dict(
        type=MyDataset,                    # Custom dataset class name
        # ... Other dataset initialization parameters ...
        reader_cfg=mydataset_reader_cfg,   # Dataset reading configuration
        infer_cfg=mydataset_infer_cfg,     # Inference configuration (including custom inferencer)
        eval_cfg=mydataset_eval_cfg        # Accuracy evaluation configuration
    )
]
```

## Notes

1. **Registration Decorator**: Custom inferencers must use the `@ICL_INFERENCERS.register_module()` decorator for registration to be recognized by the configuration system.

2. **Output Handler**: Choose an appropriate output handler according to actual needs. Commonly used ones include:
   - `GenInferencerOutputHandler`: For output processing of generative tasks
   - `PPLInferencerOutputHandler`: For output processing of perplexity evaluation

3. **Status Management**: For API model inferencers, note:
   - Use `status_counter` to track request status (post, rev, failed, finish, case_finish)
   - Correctly update status counter in the `do_request` method

4. **Error Handling**: Exception cases should be properly handled during inference to ensure output results include error information for subsequent analysis and debugging.

5. **Performance Mode**: If the inferencer needs to support performance evaluation (`mode="perf"`), ensure:
   - API model inferencers must implement the `parse_stream_response` interface (in the model class)
   - Correctly set the `perf_mode` flag
   - Use `RequestOutput` to save performance-related metrics

6. **Data Format**: Each dictionary in the data list returned by the `get_data_list` method must contain the following required fields:
   - `prompt`: Input prompt
   - `index`: Data index
   - `data_abbr`: Dataset identifier
   - `max_out_len`: Maximum output length
   - `gold`: Ground truth (optional)

