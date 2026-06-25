"""
OneIG 全量评测配置文件

OneIG-Bench 是一个面向文生图模型的综合评测基准，包含5个子任务：
- Alignment: 对齐评估（LLM-as-Judge，evaluator 内部调用 Judge 模型）
- Text: 文本评估（LLM-as-Judge，evaluator 内部调用 Judge 模型）
- Reasoning: 推理评估（LLM2CLIP）
- Style: 风格评估（CSD/SE Encoder）
- Diversity: 多样性评估（DreamSim）

使用方式：
    # 1. 克隆 OneIG 官方仓库（获取辅助数据和参考嵌入）
    git clone https://github.com/OneIG-Bench/OneIG-Benchmark.git

    # 2. 手动下载 Style 任务所需文件（见下方说明）

    # 3. 配置 ONEIG_ROOT 和 IMAGE_DIR 后运行评测
    ais_bench ais_bench/configs/oneig_examples/oneig_full_eval.py -m eval

    # 4. 仅汇总结果
    ais_bench ais_bench/configs/oneig_examples/oneig_full_eval.py -m viz

    # 单一任务评测
    修改下面的 TASKS 列表，只包含需要的任务

====================================================================
资源获取方式说明：

一、HuggingFace 自动下载的模型（首次运行时自动下载，无需手动操作）：
    - Judge 模型（Alignment/Text）→ Qwen/Qwen2.5-VL-7B-Instruct
    - LLM2CLIP Clip（Reasoning）→ openai/clip-vit-large-patch14-336
    - LLM2CLIP Vision（Reasoning）→ microsoft/LLM2CLIP-Openai-L-14-336
    - LLM2CLIP LLM（Reasoning）→ microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned
    - SE Encoder（Style）→ xingpng/OneIG-StyleEncoder

二、DreamSim 权重（Diversity）：
    首次运行时 dreamsim 库会自动从 GitHub Releases 下载权重到
    {ONEIG_ROOT}/models/ 目录。如果网络无法访问 GitHub，需手动下载：
    - 下载地址：https://github.com/ssundaram21/dreamsim/releases/download/v0.2.0-checkpoints/dreamsim_ensemble_checkpoint.zip
    - 解压到 {ONEIG_ROOT}/models/ 目录
    - 解压后应包含：dino_vitb16_pretrain.pth, open_clip_vitb16_pretrain.pth.tar,
      clip_vitb16_pretrain.pth.tar, ensemble_lora/

三、必须手动下载的文件（放置在 ONEIG_ROOT 对应目录下）：
    - CSD 编码器 → scripts/style/models/checkpoint.pth
      下载地址：https://drive.google.com/file/d/1FX0xs8p-C7Ob-h5Y4cUhTeOepHzXv_46/view
    - CLIP ViT-L-14 → scripts/style/models/ViT-L-14.pt
      下载地址：https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
    - CSD_embed.pt → scripts/style/CSD_embed.pt（随仓库分发）
    - SE_embed.pt → scripts/style/SE_embed.pt（随仓库分发）

四、随 OneIG 仓库分发的数据文件（git clone 后自动获取）：
    - scripts/alignment/Q_D/*.json（Alignment 问题依赖数据）
    - scripts/text/text_content*.csv（Text 文本内容数据）
    - scripts/reasoning/gt_answer*.json（Reasoning 参考答案）
    - scripts/style/style.csv（Style 风格标签数据）
====================================================================
"""

# =============================================================================
# 基础配置（用户需要修改的部分）
# =============================================================================

# OneIG官方项目绝对路径（需提前克隆）
ONEIG_ROOT = "/path/to/OneIG-Benchmark"

# 语言模式：EN（英文）或 ZH（中文）
MODE = "EN"

# 图片根目录（生成图片存放位置）
IMAGE_DIR = "/path/to/oneig/images"

# 模型名称列表
MODEL_NAMES = ["Qwen-Image"]

# 网格配置列表（与MODEL_NAMES一一对应）
# 格式：'rows,cols'，例如 '2,2' 表示2x2网格
# 必须与 MODEL_NAMES 长度一致
IMAGE_GRIDS = ["1,2"]

# 验证 MODEL_NAMES 和 IMAGE_GRIDS 长度一致性
if len(MODEL_NAMES) != len(IMAGE_GRIDS):
    raise ValueError(
        f"MODEL_NAMES and IMAGE_GRIDS must have the same length!\n"
        f"MODEL_NAMES length: {len(MODEL_NAMES)}\n"
        f"IMAGE_GRIDS length: {len(IMAGE_GRIDS)}\n"
        f"Example: MODEL_NAMES = ['model_a', 'model_b']\n"
        f"         IMAGE_GRIDS = ['2,2', '1,4']"
    )

# 要执行的任务列表
# 可选值：alignment, text, reasoning, style, diversity
# 注释掉不需要的任务可以实现部分评测
TASKS = ['alignment', 'text', 'reasoning', 'style', 'diversity']

# =============================================================================
# 辅助数据路径自动构建（基于 ONEIG_ROOT）
# =============================================================================

# 语言后缀
LANG_SUFFIX = "_zh" if MODE == "ZH" else ""

# 对齐评估辅助数据路径
ALIGNMENT_QD_DIR = ONEIG_ROOT + "/scripts/alignment/Q_D"

# 推理评估辅助数据路径
REASONING_GT_PATH = ONEIG_ROOT + "/scripts/reasoning/gt_answer" + LANG_SUFFIX + ".json"

# 文本评估辅助数据路径
TEXT_CONTENT_PATH = ONEIG_ROOT + "/scripts/text/text_content" + LANG_SUFFIX + ".csv"

# 风格评估辅助数据路径
STYLE_CSV_PATH = ONEIG_ROOT + "/scripts/style/style.csv"
CSD_EMBED_PATH = ONEIG_ROOT + "/scripts/style/CSD_embed.pt"
SE_EMBED_PATH = ONEIG_ROOT + "/scripts/style/SE_embed.pt"

# =============================================================================
# 模型路径配置（支持 HuggingFace 自动下载）
# =============================================================================
# 所有模型名称与官方 OneIG-Benchmark 一致，支持 HuggingFace 自动下载

# Judge 模型（Alignment/Text）- 与官方 inference.py L18 一致
JUDGE_MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"

# Judge 模型随机种子 - 与官方 inference.py L14-15 一致，确保推理结果可复现
JUDGE_SEED = 42

# LLM2CLIP 配置（Reasoning）- 与官方 inference.py L153-155 一致
LLM2CLIP_CFG = dict(
    processor_model="openai/clip-vit-large-patch14-336",  # CLIP processor
    clip_model="microsoft/LLM2CLIP-Openai-L-14-336",      # CLIP model
    llm_model="microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned",  # LLM model
)

# SE 编码器（Style）- 与官方 inference.py L130 一致
SE_ENCODER_PATH = "xingpng/OneIG-StyleEncoder"

# CSD 编码器（Style）- 官方使用本地文件，需提前下载
# 下载地址：https://drive.google.com/file/d/1FX0xs8p-C7Ob-h5Y4cUhTeOepHzXv_46/view
CSD_MODEL_PATH = ONEIG_ROOT + "/scripts/style/models/checkpoint.pth"
# CLIP ViT-L-14（Style）- CSD backbone，需提前下载
# 下载地址：https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
CLIP_MODEL_PATH = ONEIG_ROOT + "/scripts/style/models/ViT-L-14.pt"

# DreamSim 权重缓存目录（Diversity）
# dreamsim(pretrained=True) 会自动下载权重到此目录
# 如果网络不通，需手动下载 dreamsim_ensemble_checkpoint.zip 并解压到此目录
DREAMSIM_CACHE_DIR = ONEIG_ROOT + "/models"

# CSD和SE编码器配置（风格评估）
STYLE_ENCODER_CFG = dict(
    csd_model_path=CSD_MODEL_PATH,
    clip_model_path=CLIP_MODEL_PATH,
    se_model_path=SE_ENCODER_PATH,
)

# =============================================================================
# 数据集配置
# =============================================================================

from ais_bench.benchmark.datasets.oneig import OneIGDataset
from ais_bench.benchmark.tasks.oneig import (
    OneIGAlignmentEvaluator,
    OneIGTextEvaluator,
    OneIGReasoningEvaluator,
    OneIGStyleEvaluator,
    OneIGDiversityEvaluator,
)
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners import LocalRunner
from ais_bench.benchmark.tasks.oneig import OneIGEvalTask
from ais_bench.benchmark.summarizers.oneig import OneIGSummarizer

oneig_datasets = []

# -----------------------------------------------------------------------------
# 对齐评估配置（Alignment）- LLM-as-Judge模式
# Judge 推理由 OneIGAlignmentEvaluator 内部完成
# -----------------------------------------------------------------------------
if 'alignment' in TASKS:
    oneig_datasets.append(dict(
        abbr="oneig_alignment",
        type=OneIGDataset,
        path=IMAGE_DIR,
        task_type="alignment",
        model_names=MODEL_NAMES,
        image_grids=IMAGE_GRIDS,
        mode=MODE,
        reader_cfg=dict(
            input_columns=['question', 'image'],
            output_column='answer'
        ),
        aux_data_paths=dict(
            question_dependency_dir=ALIGNMENT_QD_DIR
        ),
        eval_cfg=dict(
            num_gpus=4,
            evaluator=dict(
                type=OneIGAlignmentEvaluator,
                judge_model_path=JUDGE_MODEL_PATH,
                judge_seed=JUDGE_SEED,  # 与官方 inference.py L14-15 一致
            )
        ),
    ))

# -----------------------------------------------------------------------------
# 文本评估配置（Text）- LLM-as-Judge模式
# Judge 推理由 OneIGTextEvaluator 内部完成
# -----------------------------------------------------------------------------
if 'text' in TASKS:
    oneig_datasets.append(dict(
        abbr="oneig_text",
        type=OneIGDataset,
        path=IMAGE_DIR,
        task_type="text",
        model_names=MODEL_NAMES,
        image_grids=IMAGE_GRIDS,
        mode=MODE,
        reader_cfg=dict(
            input_columns=['expected_text', 'image'],
            output_column='answer'
        ),
        aux_data_paths=dict(
            text_content_csv_path=TEXT_CONTENT_PATH
        ),
        eval_cfg=dict(
            num_gpus=4,
            evaluator=dict(
                type=OneIGTextEvaluator,
                judge_model_path=JUDGE_MODEL_PATH,
                judge_seed=JUDGE_SEED,  # 与官方 inference.py L14-15 一致
                mode=MODE,
            )
        ),
    ))

# -----------------------------------------------------------------------------
# 推理评估配置（Reasoning）- 官方ML模式（LLM2CLIP）
# -----------------------------------------------------------------------------
if 'reasoning' in TASKS:
    oneig_datasets.append(dict(
        abbr="oneig_reasoning",
        type=OneIGDataset,
        path=IMAGE_DIR,
        task_type="reasoning",
        model_names=MODEL_NAMES,
        image_grids=IMAGE_GRIDS,
        mode=MODE,
        reader_cfg=dict(
            input_columns=['gt_answer', 'image'],
            output_column='answer'
        ),
        aux_data_paths=dict(
            gt_answer_path=REASONING_GT_PATH
        ),
        eval_cfg=dict(
            num_gpus=1,
            evaluator=dict(
                type=OneIGReasoningEvaluator,
                oneig_root=ONEIG_ROOT,
                llm2clip_cfg=LLM2CLIP_CFG,
                device="cuda"
            )
        ),
    ))

# -----------------------------------------------------------------------------
# 风格评估配置（Style）- 官方ML模式（CSD/SE Encoder）
# 需提前下载 CSD checkpoint 和 CLIP ViT-L-14 到 scripts/style/models/
# -----------------------------------------------------------------------------
if 'style' in TASKS:
    oneig_datasets.append(dict(
        abbr="oneig_style",
        type=OneIGDataset,
        path=IMAGE_DIR,
        task_type="style",
        model_names=MODEL_NAMES,
        image_grids=IMAGE_GRIDS,
        mode=MODE,
        reader_cfg=dict(
            input_columns=['style_label', 'image'],
            output_column='answer'
        ),
        aux_data_paths=dict(
            style_csv_path=STYLE_CSV_PATH,
        ),
        eval_cfg=dict(
            num_gpus=1,
            evaluator=dict(
                type=OneIGStyleEvaluator,
                oneig_root=ONEIG_ROOT,
                device="cuda",
                csd_embed_path=CSD_EMBED_PATH,
                se_embed_path=SE_EMBED_PATH,
                encoder_cfg=STYLE_ENCODER_CFG
            )
        ),
    ))

# -----------------------------------------------------------------------------
# 多样性评估配置（Diversity）- 官方ML模式（DreamSim）
# DreamSim 权重缓存到 {ONEIG_ROOT}/models/，首次运行自动下载
# 如网络不通需手动下载 dreamsim_ensemble_checkpoint.zip 并解压到该目录
# -----------------------------------------------------------------------------
if 'diversity' in TASKS:
    oneig_datasets.append(dict(
        abbr="oneig_diversity",
        type=OneIGDataset,
        path=IMAGE_DIR,
        task_type="diversity",
        model_names=MODEL_NAMES,
        image_grids=IMAGE_GRIDS,
        mode=MODE,
        reader_cfg=dict(
            input_columns=['image'],
            output_column='answer'
        ),
        eval_cfg=dict(
            num_gpus=1,
            evaluator=dict(
                type=OneIGDiversityEvaluator,
                device="cuda",
                oneig_root=ONEIG_ROOT,
                dreamsim_path=DREAMSIM_CACHE_DIR,
            )
        ),
    ))

# =============================================================================
# 模型配置（Placeholder）
# =============================================================================

# OneIG 是 eval-only 模式，使用 Placeholder 模型
# Placeholder 只提供 abbr 用于生成输出路径，不执行真正的模型推理
models = [
    dict(
        attr="local",
        type="OneIGEvalPlaceholder",  # placeholder, not built in eval
        abbr="oneig_eval",
    )
]

# =============================================================================
# 导出数据集列表
# =============================================================================

datasets = oneig_datasets

# =============================================================================
# 评测配置（使用自定义 OneIGEvalTask）
# =============================================================================

# OneIGEvalTask 继承 BaseTask，覆写 run()，
# 所有5个子任务统一走 evaluator.score() 路径
eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OneIGEvalTask),
    ),
)

# =============================================================================
# 汇总配置
# =============================================================================

summarizer = dict(
    attr="accuracy",
    type=OneIGSummarizer,
    summary_groups=[
        dict(
            name='oneig_total',
            subsets=[
                'oneig_alignment', 'oneig_text', 'oneig_reasoning',
                'oneig_style', 'oneig_diversity'
            ],
            metric='accuracy',  # 只计算 accuracy 指标的平均，不产生 naive_average
        ),
    ],
)
