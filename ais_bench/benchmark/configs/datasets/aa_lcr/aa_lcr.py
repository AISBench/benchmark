from ais_bench.benchmark.datasets import AALCRDataset, AALCRJGDataset
from ais_bench.benchmark.datasets.aa_lcr.aa_lcr import AALCRJudgeEvaluator, JUDGE_PROMPT
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever

# Model inference configuration
aa_lcr_reader_cfg = dict(
    input_columns=['input'],
    output_column='answers',
)

aa_lcr_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{input}',
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

# Judge model inference configuration
aa_lcr_judge_infer_cfg = dict(
    judge_reader_cfg=dict(
        input_columns=["question", "answers", "model_answer"],
        output_column="model_pred_uuid",
    ),
    judge_model=dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="judge",
        path="",
        model="",
        stream=False,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip="localhost",
        host_port=8005,
        url="",
        max_out_len=512,
        batch_size=100,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.01,
            seed=0,
            chat_template_kwargs=dict(
                enable_thinking=False,
            ),
        ),
    ),
    judge_dataset_type=AALCRJGDataset,
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt=JUDGE_PROMPT),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

# Evaluation configuration using LLM judge evaluator
aa_lcr_eval_cfg = dict(
    evaluator=dict(type=AALCRJudgeEvaluator),
)

# Dataset definitions
aa_lcr_datasets = [
    dict(
        abbr="aa_lcr",
        type=AALCRDataset,
        path="ais_bench/datasets/aa_lcr/extracted_text/AA-LCR_extracted-text.zip",
        reader_cfg=aa_lcr_reader_cfg,
        infer_cfg=aa_lcr_infer_cfg,
        judge_infer_cfg=aa_lcr_judge_infer_cfg,
        eval_cfg=aa_lcr_eval_cfg,
    )
]
