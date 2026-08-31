from __future__ import annotations

import json
import os
from pathlib import Path

from .artifacts import find_latest_execution_manifest
from .scenario import load_scenario


def _manifest(scenario):
    """定位并读取指定场景对应的 manifest 工件文件，返回 (路径, 解析后的内容)。

    manifest 记录了 prepare 阶段生成的请求/工件元信息，是后续组装
    AISBench 配置的数据来源。
    """
    configured = os.environ.get("AISBENCH_PREFIX_CACHE_MANIFEST")
    path = Path(configured).resolve() if configured else scenario.output_dir / "result" / f"{scenario.run_id}.manifest.json"
    if not configured and not path.is_file():
        # 支持用户直接执行 config_examples/prefix_cache_perf.py：从时间戳
        # 结果目录中发现最近一次正式 prepare Manifest。inspect-only Manifest
        # 没有请求工件，不能用于组装 AISBench 数据集。
        found = find_latest_execution_manifest(scenario, {"prepared"})
        if found is not None:
            _, path, _ = found
    return path, json.loads(path.read_text(encoding="utf-8"))


def build_dataset_config(scenario_path: str | Path) -> dict:
    """根据场景配置，构造 AISBench 的 dataset 配置字典。

    指向 prepare 阶段落盘的 requests/full/manifest 工件，并装配
    PromptTemplate、ZeroRetriever、PrefixCacheGenInferencer 与 AccEvaluator，
    使 AISBench 能按 prefix cache 语义读取并执行这批请求。
    """
    # 延迟导入：inspect/prepare/路径解析不应强制加载 AISBench 在线依赖。
    from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
    from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
    from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever

    from .datasets import PrefixCacheDataset
    from .openicl.icl_inferencer import PrefixCacheGenInferencer

    scenario = load_scenario(scenario_path)
    manifest_path, manifest = _manifest(scenario)
    artifacts = manifest["artifacts"]
    return dict(
        abbr=manifest["run_id"],
        type=PrefixCacheDataset,
        requests_path=str(Path(artifacts["requests"]["path"])),
        full_path=str(Path(artifacts["full"]["path"])),
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
    from .models import VLLMPrefixCacheAPI

    scenario = load_scenario(scenario_path)
    _, manifest = _manifest(scenario)
    service = scenario.section("service")
    tokenizer = scenario.section("tokenizer")
    return dict(
        type=VLLMPrefixCacheAPI,
        abbr=f"{manifest['run_id']}-vllm",
        path=tokenizer["path"],
        model=service["model"],
        inference_url=service["inference_url"],
        api_key=service.get("api_key", ""),
        max_out_len=1,
        retry=2,
        generation_kwargs=dict(temperature=0, ignore_eos=True),
        batch_size=1,
    )
