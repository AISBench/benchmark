from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template_mm import MMPromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import VideoBenchDataset, VideoBenchEvaluator

DEFAULT_NUM_FRAMES = 5 # Default number of frames to sample from each video when loading datasets

videobench_reader_cfg = dict(
    input_columns=['question', 'video_url', 'choices_prompt'],
    output_column='answer'
)


videobench_infer_cfg = dict(
    prompt_template=dict(
        type=MMPromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt_mm={
                    "text": {"type": "text", "text": "{question} Please respond with only the corresponding options and do not provide any explanations"
                                                        + " or additional information. ASSISTANT:"},
                    "image": {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}},
                    "video": {"type": "video_url", "video_url": {"url": "data:video/jpeg;base64,{video}"}},
                    "audio": {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,{audio}"}},
                })
            ]
            )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

videobench_eval_cfg = dict(
    evaluator=dict(type=VideoBenchEvaluator)
)

videobench_datasets = [
    dict(
        abbr='videobench',
        type=VideoBenchDataset,
        path='ais_bench/datasets/videobench', # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        video_type="video_base64",
        num_frames=DEFAULT_NUM_FRAMES,
        reader_cfg=videobench_reader_cfg,
        infer_cfg=videobench_infer_cfg,
        eval_cfg=videobench_eval_cfg
    )
]