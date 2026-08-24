from ais_bench.benchmark.datasets.corpusqa import (
    CORPUSQA_JUDGE_SYSTEM_PROMPT,
    CORPUSQA_JUDGE_USER_TEMPLATE,
    CorpusQADataset,
    CorpusQAEvaluator,
    CorpusQAJGDataset,
    CorpusQAPromptTemplate,
)
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever

# ---------------------------------------------------------------------------
# CorpusQA 1M inference configuration
# ---------------------------------------------------------------------------
# The raw multi-turn ``prompt`` field is forwarded unchanged through the
# CorpusQAPromptTemplate, faithfully reproducing the official prompt.

corpusqa_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='answer',
)

corpusqa_infer_cfg = dict(
    prompt_template=dict(type=CorpusQAPromptTemplate),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

# ---------------------------------------------------------------------------
# Judge model configuration (LLM-judge / ORM, following the official script)
# ---------------------------------------------------------------------------
# The judge is served through a DashScope-compatible endpoint.  Adjust the
# ``url`` / ``host_ip`` / ``host_port`` / ``model`` fields below to point at
# your judge service.


corpusqa_judge_infer_cfg = dict(
    judge_reader_cfg=dict(
        input_columns=['question', 'answer', 'model_answer'],
        output_column='model_pred_uuid',
    ),
    judge_model=dict(
        attr='service',
        type=VLLMCustomAPIChat,
        abbr='dashscope-orm',
        path='',
        model='',
        stream=False,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        api_key='',
        host_ip='localhost',
        host_port=8005,
        max_out_len=2048,
        batch_size=8,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.0,
        ),
    ),
    judge_dataset_type=CorpusQAJGDataset,
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', prompt=CORPUSQA_JUDGE_SYSTEM_PROMPT),
            ],
            round=[
                dict(role='HUMAN', prompt=CORPUSQA_JUDGE_USER_TEMPLATE),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------

corpusqa_eval_cfg = dict(
    evaluator=dict(type=CorpusQAEvaluator),
)

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

corpusqa_1m_datasets = [
    dict(
        abbr='corpusqa_1m',
        type=CorpusQADataset,
        path='ais_bench/datasets/CorpusQA/1m_4domains.jsonl',
        reader_cfg=corpusqa_reader_cfg,
        infer_cfg=corpusqa_infer_cfg,
        judge_infer_cfg=corpusqa_judge_infer_cfg,
        eval_cfg=corpusqa_eval_cfg,
    )
]
