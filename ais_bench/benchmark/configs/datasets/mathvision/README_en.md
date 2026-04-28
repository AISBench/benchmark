# MathVision
English | [中文](README.md)

## Dataset Introduction
MathVision (MATH-V) is a multimodal mathematical reasoning benchmark with 3,040 high-quality visual math problems collected from real math competitions. Each sample combines a math question with visual context such as geometry diagrams, plots, or charts, and is designed to evaluate visual understanding, mathematical reasoning, and answer extraction.

> 🔗 Dataset Homepage: [https://modelscope.cn/datasets/evalscope/MathVision/summary](https://modelscope.cn/datasets/evalscope/MathVision/summary)

## Dataset Deployment
- This task follows the MathVision dataset source used by evalscope. The recommended dataset ID is `evalscope/MathVision`.
- `MathVisionDataset` can load the dataset directly with `datasets.load_dataset('evalscope/MathVision', split='test')`, and it also supports a local dataset path.
- Update the `path` field in [mathvision_gen.py](mathvision_gen.py) according to your environment:
```python
# Use the dataset link provided by evalscope
path = 'evalscope/MathVision'

# Or use a local dataset directory
path = '/path/to/mathvision'
```
- For local deployment, it is recommended to place the dataset under `{tool_root_path}/ais_bench/datasets/mathvision`. The directory can be a HuggingFace datasets compatible folder, or contain files such as `test.parquet`, `test-*.parquet`, `test.jsonl`, or `test.json`.
- During dataset loading, image fields are converted into local image files under the `MathVision_images` subdirectory. The inference task passes images to the multimodal model service in the `file://{image}` format.

Example directory structure:
```text
mathvision
├── data
│   └── test-00000-of-00001.parquet
└── MathVision_images
    ├── 0.png
    ├── 1.png
    └── ...
```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
| mathvision_gen | Generative multimodal mathematical reasoning task for MathVision. It supports both multiple-choice and open-answer questions. Multiple-choice questions require the final line to be `ANSWER: [LETTER]`, while open-answer questions require the final answer in `\boxed{}` | Accuracy | 0-shot | Multimodal chat format (text + image) | [mathvision_gen.py](mathvision_gen.py) |
