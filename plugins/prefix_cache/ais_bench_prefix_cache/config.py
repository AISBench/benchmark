from __future__ import annotations

import json
from pathlib import Path

from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever

from .datasets import PrefixCacheDataset
from .models import VLLMPrefixCacheAPI
from .openicl.icl_inferencer import PrefixCacheGenInferencer
from .scenario import load_scenario


def _manifest(scenario):
    """定位并读取指定场景对应的 manifest 工件文件，返回 (路径, 解析后的内容)。

    manifest 记录了 prepare 阶段生成的请求/工件元信息，是后续组装
    AISBench 配置的数据来源。
    """
    path = scenario.output_dir / f"{scenario.run_id}.manifest.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def build_dataset_config(scenario_path: str | Path) -> dict:
    """根据场景配置，构造 AISBench 的 dataset 配置字典。

    指向 prepare 阶段落盘的 requests/full/manifest 工件，并装配
    PromptTemplate、ZeroRetriever、PrefixCacheGenInferencer 与 AccEvaluator，
    使 AISBench 能按 prefix cache 语义读取并执行这批请求。
    """
    scenario = load_scenario(scenario_path)
    manifest_path, manifest = _manifest(scenario)
    base = manifest_path.parent
    return dict(
        abbr=scenario.run_id,
        type=PrefixCacheDataset,
        requests_path=str(base / manifest["artifacts"]["requests"]["name"]),
        full_path=str(base / manifest["artifacts"]["full"]["name"]),
        manifest_path=str(manifest_path),
        reader_cfg=dict(input_columns=["question", "max_out_len"], output_column="answer"),
        infer_cfg=dict(
            prompt_template=dict(type=PromptTemplate, template="{question}"),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=PrefixCacheGenInferencer),
        ),
        eval_cfg=dict(evaluator=dict(type=AccEvaluator), pred_role="BOT"),
    )


def build_model_config(scenario_path: str | Path) -> dict:
    """构造 AISBench 的模型（推理服务）配置字典。

    从场景的 service/tokenizer 段取出模型名、推理地址、api_key 等，
    封装为 VLLMPrefixCacheAPI；max_out_len=1 表示本基准只关心
    前缀命中，不关心实际生成内容。
    """
    scenario = load_scenario(scenario_path)
    service = scenario.section("service")
    tokenizer = scenario.section("tokenizer")
    return dict(
        type=VLLMPrefixCacheAPI,
        abbr=f"{scenario.run_id}-vllm",
        path=tokenizer["path"],
        model=service["model"],
        inference_url=service["inference_url"],
        api_key=service.get("api_key", ""),
        max_out_len=1,
        retry=2,
        generation_kwargs=dict(temperature=0, ignore_eos=True),
        batch_size=1,
    )
