from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template_mm import MMPromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_inferencer.icl_lmm_gen_inferencer import LMMGenInferencer
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content
from ais_bench.benchmark.datasets.g_edit import (
    GEditDataset,
    GEditSCJDGDataset,
    GEditPQJDGDataset,
)
from ais_bench.benchmark.datasets.utils.lmm_judge import get_lmm_point_list, LMMJudgeImageEditEvaluator

SC_GRADER_TEMPLATE = """
RULES:

Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.
The objective is to evaluate how successfully the editing instruction has been executed in the second image.

Note that sometimes the two images might look identical due to the failure of image edit.

From scale 0 to 10:
A score from 0 to 10 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 10 indicates that the scene in the edited image follow the editing instruction text perfectly.)
A second score from 0 to 10 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.

Editing instruction: {question}
""".strip()

PQ_GRADER_TEMPLATE = """
RULES:

The image is an AI-generated image.
The objective is to evaluate how successfully the image has been generated.

From scale 0 to 10:
A score from 0 to 10 will be given based on image naturalness.
(
    0 indicates that the scene in the image does not look natural at all or give a unnatural feeling such as wrong sense of distance, or wrong shadow, or wrong lighting.
    10 indicates that the image looks natural.
)
A second score from 0 to 10 will rate the image artifacts.
(
    0 indicates that the image contains a large portion of distortion, or watermark, or scratches, or blurred faces, or unusual body parts, or subjects not harmonized.
    10 indicates the image has no artifacts.
)
Put the score in a list such that output score = [naturalness, artifacts]
""".strip()

JDG_DATASETS_CLASS_MAP = {
    "SC": GEditSCJDGDataset,
    "PQ": GEditPQJDGDataset,
}

JDG_TEMPLATE_MAP = {
    "SC": SC_GRADER_TEMPLATE,
    "PQ": PQ_GRADER_TEMPLATE,
}


gedit_reader_cfg = dict(
    input_columns=['question', 'image'],
    output_column='task_type'
)


gedit_infer_cfg = dict(
    prompt_template=dict(
        type=MMPromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt_mm={
                    "text": {"type": "text", "text": "{question}"},
                    "image": {"type": "image_url", "image_url": {"url": "data:image/png;base64,{image}"}},
                })
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=LMMGenInferencer)
)

gedit_datasets = []

for metric in ["SC", "PQ"]:
    gedit_judge_infer_cfg = dict(
        judge_reader_cfg = dict(input_columns=["question", "model_answer", "image"], output_column="model_pred_uuid"),
        judge_model=dict(
            attr="service",
            type=VLLMCustomAPIChat,
            abbr=f"{metric}_judge", # Be added after dataset abbr
            path="",
            model="",
            stream=True,
            request_rate=0,
            use_timestamp=False,
            retry=2,
            api_key="",
            host_ip="localhost",
            host_port=8080,
            url="",
            max_out_len=512,
            batch_size=16,
            trust_remote_code=False,
            generation_kwargs=dict(
                temperature=0.01,
                ignore_eos=False,
            ),
            pred_postprocessor=dict(type=extract_non_reasoning_content),
        ),
        judge_dataset_type=JDG_DATASETS_CLASS_MAP[metric],
        prompt_template=dict(
            type=MMPromptTemplate,
            template=dict(
                round=[
                    dict(role='HUMAN', prompt_mm={
                        "text": {"type": "text", "text": JDG_TEMPLATE_MAP[metric]},
                        "image": {"type": "image_url", "image_url": {"url": "data:image/png;base64,{image}"}},
                    })
                ],
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    gedit_eval_cfg = dict(
        evaluator=dict(type=LMMJudgeImageEditEvaluator, metric=metric),
        pred_postprocessor=dict(type=get_lmm_point_list),
    )

    gedit_datasets.append(
        dict(
            abbr=f"gedit",
            type=GEditDataset,
            path="ais_bench/datasets/GEdit-Bench",
            reader_cfg=gedit_reader_cfg,
            infer_cfg=gedit_infer_cfg,
            judge_infer_cfg=gedit_judge_infer_cfg,
            eval_cfg=gedit_eval_cfg,
        )
    )