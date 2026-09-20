from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_TOKENS = 256


def create_aisbench_config(
    *,
    dataset_path: Path | str,
    dataset_abbr: str,
    tokenizer_path: str,
    inference_url: str,
    model: str,
    work_dir: Path | str,
    image_ref: str,
    image_data_url: str,
    output_tokens: int = DEFAULT_OUTPUT_TOKENS,
    batch_size: int = 1,
    api_key: str = "",
    stream: bool = True,
    retry: int = 2,
    generation_kwargs: dict[str, Any] | None = None,
    pred_role: str = "BOT",
    model_abbr: str = "prefix-cache-mm-vllm",
    model_attr: str = "service",
    model_max_out_len: int | None = None,
) -> dict[str, Any]:
    from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
    from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
    from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
    from ais_bench.benchmark.partitioners import NaivePartitioner
    from ais_bench.benchmark.runners.local import LocalRunner
    from ais_bench.benchmark.tasks import OpenICLApiInferTask

    from ais_bench_prefix_cache.datasets import MultimodalPrefixCacheDataset
    from ais_bench_prefix_cache.models import VLLMPrefixCacheChatAPI
    from ais_bench_prefix_cache.openicl import Base64RefMMPromptTemplate

    actual_generation_kwargs = (
        generation_kwargs
        if generation_kwargs is not None
        else {"temperature": 0, "ignore_eos": True}
    )
    actual_max_out_len = (
        model_max_out_len if model_max_out_len is not None else output_tokens
    )
    datasets = [
        dict(
            abbr=dataset_abbr,
            type=MultimodalPrefixCacheDataset,
            path=str(Path(dataset_path).resolve()),
            reader_cfg=dict(
                input_columns=["content", "max_out_len"],
                output_column="answer",
            ),
            infer_cfg=dict(
                prompt_template=dict(
                    type=Base64RefMMPromptTemplate,
                    media={image_ref: image_data_url},
                    template=dict(
                        round=[
                            dict(
                                role="HUMAN",
                                prompt_mm={
                                    "image": {
                                        "type": "image_url",
                                        "image_url": {"url": "{image}"},
                                    },
                                    "text": {
                                        "type": "text",
                                        "text": "{question}",
                                    },
                                },
                            )
                        ]
                    ),
                ),
                retriever=dict(type=ZeroRetriever),
                inferencer=dict(type=GenInferencer),
            ),
            eval_cfg=dict(
                evaluator=dict(type=AccEvaluator),
                pred_role=pred_role,
            ),
        )
    ]
    models = [
        dict(
            type=VLLMPrefixCacheChatAPI,
            attr=model_attr,
            abbr=model_abbr,
            path=tokenizer_path,
            model=model,
            inference_url=inference_url,
            api_key=api_key,
            stream=stream,
            max_out_len=actual_max_out_len,
            retry=retry,
            batch_size=batch_size,
            generation_kwargs=actual_generation_kwargs,
        )
    ]
    infer = dict(
        partitioner=dict(type=NaivePartitioner),
        runner=dict(
            type=LocalRunner,
            max_num_workers=1,
            task=dict(type=OpenICLApiInferTask),
        ),
    )
    summarizer = dict(attr="accuracy", summary_groups=[])
    return {
        "datasets": datasets,
        "models": models,
        "infer": infer,
        "summarizer": summarizer,
        "work_dir": str(Path(work_dir).resolve()),
    }


def build_aisbench_config(
    *,
    dataset_path: Path | str,
    dataset_abbr: str,
    tokenizer_path: str,
    inference_url: str,
    model: str,
    work_dir: Path | str,
    image_ref: str,
    image_data_url: str,
    output_tokens: int = DEFAULT_OUTPUT_TOKENS,
    batch_size: int = 1,
    api_key: str = "",
    stream: bool = True,
    retry: int = 2,
    generation_kwargs: dict[str, Any] | None = None,
    pred_role: str = "BOT",
    model_abbr: str = "prefix-cache-mm-vllm",
    model_attr: str = "service",
    model_max_out_len: int | None = None,
) -> str:
    """Build a standalone MMEngine-compatible AISBench config.

    MMEngine parses imported objects in config files as lazy objects.  Calling
    an imported factory from the generated config therefore raises
    ``RuntimeError``.  Emit the config declarations directly so the file can
    be consumed by both eager and lazy config parsing.
    """
    actual_generation_kwargs = (
        generation_kwargs
        if generation_kwargs is not None
        else {"temperature": 0, "ignore_eos": True}
    )
    actual_max_out_len = (
        model_max_out_len if model_max_out_len is not None else output_tokens
    )
    lines = [
        "# Generated by ais-bench-prefix-cache; do not edit.",
        "from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator",
        "from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer",
        "from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever",
        "from ais_bench.benchmark.partitioners import NaivePartitioner",
        "from ais_bench.benchmark.runners.local import LocalRunner",
        "from ais_bench.benchmark.tasks import OpenICLApiInferTask",
        "from ais_bench_prefix_cache.datasets import MultimodalPrefixCacheDataset",
        "from ais_bench_prefix_cache.models import VLLMPrefixCacheChatAPI",
        "from ais_bench_prefix_cache.openicl import Base64RefMMPromptTemplate",
        "",
        "datasets = [",
        "    dict(",
        f"        abbr={dataset_abbr!r},",
        "        type=MultimodalPrefixCacheDataset,",
        f"        path={str(Path(dataset_path).resolve())!r},",
        "        reader_cfg=dict(",
        "            input_columns=['content', 'max_out_len'],",
        "            output_column='answer',",
        "        ),",
        "        infer_cfg=dict(",
        "            prompt_template=dict(",
        "                type=Base64RefMMPromptTemplate,",
        f"                media={{{image_ref!r}: {image_data_url!r}}},",
        "                template=dict(",
        "                    round=[",
        "                        dict(",
        "                            role='HUMAN',",
        "                            prompt_mm={",
        "                                'image': {",
        "                                    'type': 'image_url',",
        "                                    'image_url': {'url': '{image}'},",
        "                                },",
        "                                'text': {",
        "                                    'type': 'text',",
        "                                    'text': '{question}',",
        "                                },",
        "                            },",
        "                        )",
        "                    ]",
        "                ),",
        "            ),",
        "            retriever=dict(type=ZeroRetriever),",
        "            inferencer=dict(type=GenInferencer),",
        "        ),",
        "        eval_cfg=dict(",
        "            evaluator=dict(type=AccEvaluator),",
        f"            pred_role={pred_role!r},",
        "        ),",
        "    )",
        "]",
        "",
        "models = [",
        "    dict(",
        "        type=VLLMPrefixCacheChatAPI,",
        f"        attr={model_attr!r},",
        f"        abbr={model_abbr!r},",
        f"        path={tokenizer_path!r},",
        f"        model={model!r},",
        f"        inference_url={inference_url!r},",
        f"        api_key={api_key!r},",
        f"        stream={stream!r},",
        f"        max_out_len={actual_max_out_len!r},",
        f"        retry={retry!r},",
        f"        batch_size={batch_size!r},",
        f"        generation_kwargs={actual_generation_kwargs!r},",
        "    )",
        "]",
        "",
        "infer = dict(",
        "    partitioner=dict(type=NaivePartitioner),",
        "    runner=dict(",
        "        type=LocalRunner,",
        "        max_num_workers=1,",
        "        task=dict(type=OpenICLApiInferTask),",
        "    ),",
        ")",
        "",
        "summarizer = dict(attr='accuracy', summary_groups=[])",
        f"work_dir = {str(Path(work_dir).resolve())!r}",
        "",
    ]
    return "\n".join(lines)
