from __future__ import annotations

import copy
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
    dataset_cfg = scenario.section("aisbench")["dataset"]
    return dict(
        abbr=dataset_cfg["abbr"] or manifest["run_id"],
        type=PrefixCacheDataset,
        requests_path=str(Path(artifacts["requests"]["path"])),
        full_path=str(Path(artifacts["full"]["path"])),
        manifest_path=str(manifest_path),
        reader_cfg=dict(
            input_columns=list(dataset_cfg["input_columns"]),
            output_column=dataset_cfg["output_column"],
        ),
        infer_cfg=dict(
            prompt_template=dict(type=PromptTemplate, template=dataset_cfg["prompt_template"]),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=PrefixCacheGenInferencer),
        ),
        eval_cfg=dict(evaluator=dict(type=AccEvaluator), pred_role=dataset_cfg["pred_role"]),
    )


def build_model_config(scenario_path: str | Path) -> dict:
    """构造 AISBench 的模型（推理服务）配置字典。

    从场景的 service/tokenizer/aisbench.model 段取出连接与运行参数，
    封装为 VLLMPrefixCacheAPI。默认启用流式响应以采集 TTFT/TPOT/ITL，
    用户也可在 Scenario 中显式关闭。
    """
    from .models import VLLMPrefixCacheAPI

    scenario = load_scenario(scenario_path)
    _, manifest = _manifest(scenario)
    service = scenario.section("service")
    tokenizer = scenario.section("tokenizer")
    model_cfg = scenario.section("aisbench")["model"]
    return dict(
        type=VLLMPrefixCacheAPI,
        abbr=model_cfg["abbr"] or f"{manifest['run_id']}-vllm",
        path=tokenizer["path"],
        model=service["model"],
        inference_url=service["inference_url"],
        api_key=service.get("api_key", ""),
        stream=model_cfg["stream"],
        max_out_len=model_cfg["max_out_len"],
        retry=model_cfg["retry"],
        generation_kwargs=copy.deepcopy(model_cfg["generation_kwargs"]),
        batch_size=model_cfg["batch_size"],
    )
