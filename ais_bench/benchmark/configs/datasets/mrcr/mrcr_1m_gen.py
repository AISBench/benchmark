from ais_bench.benchmark.datasets.mrcr import (
    MRCRDataset,
    MRCREvaluator,
    MRCRPromptTemplate,
)
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever

# ---------------------------------------------------------------------------
# MRCR 1M inference configuration
# ---------------------------------------------------------------------------
# The raw multi-turn ``prompt`` field (JSON-encoded in the parquet rows) is
# forwarded unchanged through MRCRPromptTemplate, faithfully reproducing the
# prompt the official OpenAI runner posts to the chat API.
#
# Deploy openai/mrcr under ``ais_bench/datasets/MRCR`` with one
# sub-directory per needle-count subset (``2needle/`` ``4needle/``
# ``8needle/`` containing the parquet shards).  Switch subset via
# ``subset`` and the token bin via ``length_bin`` (see
# MRCR_BIN_BOUNDARIES in datasets/mrcr.py); ``length_bin=None`` disables
# bin filtering (fast smoke-test path).

mrcr_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='answer',
)

mrcr_infer_cfg = dict(
    prompt_template=dict(type=MRCRPromptTemplate),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

# ---------------------------------------------------------------------------
# Evaluation configuration (rule-based scoring, no LLM judge)
# ---------------------------------------------------------------------------
# Mirrors the official grade(): the response must start with the sample's
# ``random_string_to_prepend``, then SequenceMatcher.ratio() is computed
# against the gold needle text.  MRCREvaluator reads the per-sample
# prefixes from the dataset column and prefers the reasoning-free
# ``content`` field persisted by GenInferencerOutputHandler for thinking
# models.  ``mrcr_postprocess`` is the identity postprocessor:
# predictions are graded raw, exactly as the official script does.

mrcr_eval_cfg = dict(
    evaluator=dict(type=MRCREvaluator),
    pred_postprocessor=dict(type='mrcr_postprocess'),
)

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

mrcr_1m_datasets = [
    dict(
        abbr='mrcr_1m',
        type=MRCRDataset,
        path='ais_bench/datasets/MRCR',
        subset='2needle',
        length_bin='1m',
        reader_cfg=mrcr_reader_cfg,
        infer_cfg=mrcr_infer_cfg,
        eval_cfg=mrcr_eval_cfg,
    )
]
