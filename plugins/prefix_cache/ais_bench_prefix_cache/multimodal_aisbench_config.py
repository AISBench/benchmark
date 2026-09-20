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
    arguments = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_abbr": dataset_abbr,
        "tokenizer_path": tokenizer_path,
        "inference_url": inference_url,
        "model": model,
        "work_dir": str(Path(work_dir).resolve()),
        "image_ref": image_ref,
        "image_data_url": image_data_url,
        "output_tokens": output_tokens,
        "batch_size": batch_size,
        "api_key": api_key,
        "stream": stream,
        "retry": retry,
        "generation_kwargs": generation_kwargs,
        "pred_role": pred_role,
        "model_abbr": model_abbr,
        "model_attr": model_attr,
        "model_max_out_len": model_max_out_len,
    }
    lines = [
        "# Generated by ais-bench-prefix-cache; do not edit.",
        "from ais_bench_prefix_cache.multimodal_aisbench_config import create_aisbench_config",
        "",
        "_config = create_aisbench_config(",
    ]
    lines.extend(f"    {name}={value!r}," for name, value in arguments.items())
    lines.extend([
        ")",
        "datasets = _config['datasets']",
        "models = _config['models']",
        "infer = _config['infer']",
        "summarizer = _config['summarizer']",
        "work_dir = _config['work_dir']",
        "del _config",
        "",
    ])
    return "\n".join(lines)
