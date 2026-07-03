# OneIG-Benchmark

**OneIG-Benchmark** is a comprehensive evaluation benchmark for text-to-image models, organized around five dimensions: **Alignment**, **Text Rendering**, **Reasoning**, **Style**, and **Diversity**. The official Standard suite includes **5 sub-tasks** and supports both **EN (English)** and **ZH (Chinese)** language modes.

AISBench **has adapted OneIG-Benchmark**. The `ais_bench/configs/oneig_examples/` directory contains **standalone configuration file** examples for multi-dimensional quality evaluation of **generated images** on **GPU**. OneIG uses an **eval-only** mode and does not include image generation steps. Please generate images using the model under evaluation before running the assessment.

## Dataset Overview

### Background

OneIG-Benchmark is developed to comprehensively evaluate the generation quality of text-to-image models from multiple fine-grained dimensions. Official GitHub: [https://github.com/OneIG-Bench/OneIG-Benchmark](https://github.com/OneIG-Bench/OneIG-Benchmark), Dataset: [https://huggingface.co/datasets/OneIG-Bench/OneIG-Benchmark](https://huggingface.co/datasets/OneIG-Bench/OneIG-Benchmark).

### Key Features

| Feature | Description |
| --- | --- |
| **5-Dimension Evaluation** | Covers alignment, text, reasoning, style, and diversity |
| **Bilingual Support** | EN (English) / ZH (Chinese) modes |
| **LLM-as-Judge** | Alignment and Text tasks use multimodal LLM as judge |
| **ML Model Evaluation** | Reasoning, Style, and Diversity use specialized ML models |
| **Eval-Only Mode** | Only evaluates generated images, no image generation step |
| **Grid Splitting** | Supports automatic splitting of grid-composited images into sub-images |
| **Accuracy Alignment** | Accuracy difference < 1% compared to official evaluation |

### Architecture Overview

The end-to-end evaluation process consists of data preparation and evaluation phases:

```
Data Preparation                      Evaluation Phase
┌──────────────────────┐    ┌───────────────────────────────────────────┐
│ OneIG-Bench.csv      │    │              Evaluation Phase              │
│ (Original Dataset)   │    │                                           │
│       ↓              │    │  images/ directory (images under test)    │
│ Prompt Extraction    │    │       ↓                                   │
│       ↓              │    │  ┌─────────────┐  ┌─────────────┐        │
│ T2I Model Generation │    │  │ Alignment   │  │ Text        │        │
│       ↓              │    │  │ (LLM-Judge) │  │ (LLM-Judge) │        │
│ images/ directory    │───▶│  └─────────────┘  └─────────────┘        │
│ (Evaluation Target)  │    │  ┌─────────────┐  ┌─────────────┐        │
└──────────────────────┘    │  │ Reasoning   │  │ Style       │        │
                            │  │ (LLM2CLIP)  │  │ (CSD+SE)    │        │
                            │  └─────────────┘  └─────────────┘        │
                            │  ┌─────────────┐                         │
                            │  │ Diversity   │  → results/ directory   │
                            │  │ (DreamSim)  │    (evaluation results) │
                            │  └─────────────┘                         │
                            └───────────────────────────────────────────┘
```

**AISBench Adaptation Architecture** (four-layer separation):

```
ais_bench/
├── benchmark/                              # Framework Layer
│   ├── datasets/oneig.py                   # Dataset Loader
│   ├── tasks/oneig/                        # Evaluation Task Package
│   │   ├── __init__.py                     # Module Entry
│   │   ├── oneig_eval.py                   # Evaluation Task (OneIGEvalTask)
│   │   ├── oneig_eval_utils.py             # Utility Functions
│   │   ├── oneig_alignment_eval.py         # Alignment Evaluator
│   │   ├── oneig_text_eval.py              # Text Evaluator
│   │   ├── oneig_reasoning_eval.py         # Reasoning Evaluator
│   │   ├── oneig_style_eval.py             # Style Evaluator
│   │   └── oneig_diversity_eval.py         # Diversity Evaluator
│   └── summarizers/oneig.py                # Score Summarizer
├── configs/oneig_examples/                 # User Example Configs
│   └── oneig_full_eval.py                  # Full Evaluation Config
└── docs/
    ├── source_zh_cn/extended_benchmark/lmm_generate/oneig.md   # Chinese Doc
    └── source_en/extended_benchmark/lmm_generate/oneig.md      # English Doc
```

## Dependencies and Environment

### Base Environment

OneIG evaluation supports **GPU** only. Before starting, ensure AISBench is installed:

```bash
# Clone AISBench repository
git clone https://github.com/AISBench/benchmark.git
cd benchmark/

# Install dependencies
pip install -e ./ --use-pep517
```

### OneIG Official Repository

OneIG evaluation depends on auxiliary data and reference embeddings from the official repository:

```bash
# Clone OneIG code from AISBench organization (with known bugs fixed)
git clone https://github.com/AISBench/OneIG-Benchmark.git
cd OneIG-Benchmark/

# Install dependencies
pip install -r requirements.txt
```

### Model Weights and Resource Downloads

OneIG evaluation involves multiple model weights, categorized as follows:

#### 1. HuggingFace Auto-Download (automatic on first run, no manual action required)

| Model | Used For | HuggingFace Path |
| --- | --- | --- |
| Judge Model | Alignment / Text | `Qwen/Qwen3-VL-8B-Instruct` |
| LLM2CLIP Clip | Reasoning | `openai/clip-vit-large-patch14-336` |
| LLM2CLIP Vision | Reasoning | `microsoft/LLM2CLIP-Openai-L-14-336` |
| LLM2CLIP LLM | Reasoning | `microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned` |
| SE Encoder | Style | `xingpng/OneIG-StyleEncoder` |

#### 2. DreamSim Weights (Diversity)

On first run, the `dreamsim` library automatically downloads weights to `{ONEIG_ROOT}/models/`. If GitHub is inaccessible, download manually:

- Download URL: `https://github.com/ssundaram21/dreamsim/releases/download/v0.2.0-checkpoints/dreamsim_ensemble_checkpoint.zip`
- Extract to `{ONEIG_ROOT}/models/` directory
- Should contain: `dino_vitb16_pretrain.pth`, `open_clip_vitb16_pretrain.pth.tar`, `clip_vitb16_pretrain.pth.tar`, `ensemble_lora/`

#### 3. Manual Download Required (Style Task)

| File | Archive Path | Download URL |
| --- | --- | --- |
| CSD Encoder | `{ONEIG_ROOT}/scripts/style/models/checkpoint.pth` | [Google Drive](https://drive.google.com/file/d/1FX0xs8p-C7Ob-h5Y4cUhTeOepHzXv_46/view) |
| CLIP ViT-L-14 | `{ONEIG_ROOT}/scripts/style/models/ViT-L-14.pt` | [OpenAI Public](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) |

#### 4. Data Files Distributed with OneIG Repository (auto-obtained via `git clone`)

| File | Path | Purpose |
| --- | --- | --- |
| Question Dependencies | `scripts/alignment/Q_D/*.json` | Alignment task Q&A dependencies |
| Text Content Data | `scripts/text/text_content*.csv` | Text task reference texts |
| Reference Answers | `scripts/reasoning/gt_answer*.json` | Reasoning task reference answers |
| Style Labels | `scripts/style/style.csv` | Style task style labels |
| CSD Reference Embeddings | `scripts/style/CSD_embed.pt` | Style task CSD reference vectors |
| SE Reference Embeddings | `scripts/style/SE_embed.pt` | Style task SE reference vectors |

## Quick Start

### Configuration

Edit the config file `ais_bench/configs/oneig_examples/oneig_full_eval.py` and modify the following key parameters:

```python
# OneIG official project absolute path (clone required)
ONEIG_ROOT = "/path/to/OneIG-Benchmark"

# Language mode: EN (English) or ZH (Chinese)
MODE = "EN"

# Image root directory (where generated images are stored)
IMAGE_DIR = "/path/to/oneig/images"

# Model name list (name of the image generation model)
MODEL_NAMES = ["Qwen-Image"]

# Grid configuration list (corresponds to MODEL_NAMES, format: 'rows,cols')
IMAGE_GRIDS = ["2,2"]

# Task list to execute (freely combinable)
TASKS = ['alignment', 'text', 'reasoning', 'style', 'diversity']
```

### Run Evaluation

```bash
# Full evaluation (5 sub-tasks)
ais_bench ais_bench/configs/oneig_examples/oneig_full_eval.py -m eval
```

### Results

After evaluation, results are output to `outputs/default/{timestamp}/`:

```
outputs/default/{timestamp}/
├── configs/
│   └── {timestamp}.py                    # Evaluation config snapshot
├── logs/
│   └── eval/
│       └── oneig_eval/
│           ├── oneig_alignment.out       # Task logs
│           ├── oneig_text.out
│           ├── oneig_reasoning.out
│           ├── oneig_style.out
│           └── oneig_diversity.out
├── results/
│   └── oneig_eval/
│       ├── oneig_alignment.json          # Task results (with per-sample details)
│       ├── oneig_text.json
│       ├── oneig_reasoning.json
│       ├── oneig_style.json
│       └── oneig_diversity.json
└── summary/
    ├── summary_{timestamp}.csv           # Evaluation summary
    ├── summary_{timestamp}.md
    └── summary_{timestamp}.txt
```

## Configuration and Output

### Common Configuration Options

| Option | Purpose | Required |
| --- | --- | --- |
| `ONEIG_ROOT` | OneIG official project absolute path | Yes |
| `MODE` | Language mode: `EN` or `ZH` | Yes |
| `IMAGE_DIR` | Image root directory for evaluation | Yes |
| `MODEL_NAMES` | List of image generation model names | Yes |
| `IMAGE_GRIDS` | Grid configuration list, format `'rows,cols'`, corresponds to `MODEL_NAMES` | Yes |
| `TASKS` | Task list, options: `alignment`, `text`, `reasoning`, `style`, `diversity` | Yes |
| `JUDGE_MODEL_PATH` | Judge model path (Alignment/Text), default `Qwen/Qwen3-VL-8B-Instruct` | No |
| `JUDGE_SEED` | Judge model random seed, default `42` | No |
| `DREAMSIM_CACHE_DIR` | DreamSim weight cache directory, default `{ONEIG_ROOT}/models` | No |

### Preset Configurations

| Config Name | Description | Config File |
| --- | --- | --- |
| oneig_full_eval | Full evaluation config with 5 sub-tasks, freely combinable | `ais_bench/configs/oneig_examples/oneig_full_eval.py` |

### Result Path

Written per sub-task:

```
{work_dir}/results/oneig_eval/oneig_{task}.json
```

Where `{task}` is one of `alignment`, `text`, `reasoning`, `style`, `diversity`.

### Output Format

Each sub-task JSON result file has the following structure (using Alignment as an example):

```json
{
    "accuracy": 88.44,
    "details": [
        {
            "id": "000",
            "class_item": "anime",
            "score": 0.85,
            "image_path": "/path/to/image.png",
            "grid": "2x2",
            "num_splits": 4,
            "judge_details": [
                {
                    "question_id": "Q1",
                    "question": "...",
                    "judge_prompt": "...",
                    "judge_outputs": [
                        {"grid_idx": 0, "raw_output": "Yes", "parsed_answer": "Yes", "score": 1.0}
                    ],
                    "dependency": [0],
                    "filtered_scores": null
                }
            ]
        }
    ],
    "style_scores": null
}
```

The `details` field contains different intermediate data per sub-task:

| Sub-task | Intermediate Data Field | Description |
| --- | --- | --- |
| Alignment | `judge_details` | Per-split Judge Q&A details |
| Text | `ocr_details` | Per-split OCR results and text metrics (ED/CR/WAC) |
| Reasoning | `similarity_details` | Per-split similarity scores |
| Style | `encoder_details` | Per-split CSD/SE similarity and style scores |
| Diversity | `pairwise_distances` | Per-pair split DreamSim distances |

## Evaluation Metrics

### Metric Overview

| Sub-task | Primary Metric | Auxiliary Metrics | Evaluation Method | Evaluation Model |
| --- | --- | --- | --- | --- |
| Alignment | `accuracy` | - | LLM-as-Judge | Qwen3-VL-8B-Instruct |
| Text | `accuracy` | `ED`, `CR`, `WAC` | LLM-as-Judge + OCR | Qwen3-VL-8B-Instruct |
| Reasoning | `accuracy` | - | Feature Similarity | LLM2CLIP |
| Style | `accuracy` | - | Feature Similarity | CSD + SE Encoder |
| Diversity | `accuracy` | `oneig_diversity_{class}` | Perceptual Distance | DreamSim |
| **Total** | `oneig_total` | - | Average of 5 tasks | - |

### Sub-task Evaluation Logic

#### Alignment (LLM-as-Judge)

**Goal**: Evaluate the alignment between generated images and prompts.

**Flow**:
1. Split grid images into sub-images
2. For each sub-image, use Judge model (Qwen3-VL-8B-Instruct) to answer Yes/No questions
3. "Yes" scores 1, "No" scores 0
4. Average all sub-image scores as the sample score
5. Average all sample scores × 100 as accuracy

**Key Parameters**:
- `judge_model_path`: Judge model path
- `judge_seed`: Random seed (default 42, ensures reproducibility)
- `num_gpus`: Supports multi-GPU parallelism (recommended 4)

#### Text (LLM-as-Judge + OCR)

**Goal**: Evaluate the accuracy of text rendering in generated images.

**Flow**:
1. Split grid images into sub-images
2. Use Judge model to perform OCR on each sub-image, extracting text
3. Compare extracted text with reference text, computing three metrics:
   - **ED** (Edit Distance): Edit distance
   - **CR** (Character Ratio): Character ratio
   - **WAC** (Word Accuracy Coincidence): Word accuracy
4. Combine OCR metrics and Judge score to get accuracy

#### Reasoning (LLM2CLIP)

**Goal**: Evaluate the understanding of reasoning-type prompts in generated images.

**Flow**:
1. Split grid images into sub-images
2. Use LLM2CLIP to extract image features and reference answer text features
3. Compute cosine similarity between image and text features
4. Average all sub-image similarities as the sample score
5. Average all sample scores × 100 as accuracy

**Model Components**:
- CLIP Processor: `openai/clip-vit-large-patch14-336`
- CLIP Model: `microsoft/LLM2CLIP-Openai-L-14-336`
- LLM Model: `microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned`

#### Style (CSD + SE Encoder)

**Goal**: Evaluate the style performance of generated images.

**Flow**:
1. Split grid images into sub-images
2. Use CSD (CLIP-Style-Diffusion) encoder to extract style features
3. Use SE (Style Encoder) encoder to extract style features
4. Compute cosine similarity of CSD and SE features with reference style embeddings
5. Average the two similarities as the sub-image style score
6. Average all sub-images, then all samples × 100 as accuracy

**Style Categories** (29 types): abstract_expressionism, art_nouveau, baroque, chinese_ink_painting, cubism, fauvism, impressionism, line_art, minimalism, pointillism, pop_art, rococo, ukiyo-e, clay, crayon, graffiti, lego, comic, pencil_sketch, stone_sculpture, watercolor, celluloid, chibi, cyberpunk, ghibli, impasto, pixar, pixel_art, 3d_rendering

#### Diversity (DreamSim)

**Goal**: Evaluate the diversity among multiple images generated by the same model.

**Flow**:
1. Split grid images into sub-images
2. Use DreamSim model to compute pairwise perceptual distances between all sub-images
3. Average all distance pairs as the sample diversity score
4. Group by class_item (anime, human, object, text, reasoning) for fine-grained metrics
5. Average all sample scores × 100 as accuracy

### Score Aggregation (oneig_total)

`oneig_total` is the simple average of 5 sub-task accuracy values:

```
oneig_total = (alignment + text + reasoning + style + diversity) / 5
```

Additionally, the Diversity task outputs fine-grained metrics grouped by class_item:

| Metric | Description |
| --- | --- |
| `oneig_diversity_anime` | Diversity score for Anime category |
| `oneig_diversity_human` | Diversity score for Portrait category |
| `oneig_diversity_object` | Diversity score for General Object category |
| `oneig_diversity_text` | Diversity score for Text Rendering category |
| `oneig_diversity_reasoning` | Diversity score for Knowledge Reasoning category |

### Example Results

```
dataset             version  metric   mode  oneig_eval
oneig_alignment     a39421   accuracy gen   88.44
oneig_text          a39421   accuracy gen   80.79
oneig_text          a39421   ED        gen   43.32
oneig_text          a39421   CR        gen   0.08
oneig_text          a39421   WAC       gen   0.52
oneig_reasoning     a39421   accuracy gen   29.84
oneig_style         a39421   accuracy gen   35.85
oneig_diversity     a39421   accuracy gen   18.28
oneig_total         -        accuracy gen   50.64
oneig_diversity_anime  -     accuracy gen   9.00
oneig_diversity_human  -     accuracy gen   11.21
oneig_diversity_object -    accuracy gen   13.27
oneig_diversity_text   -     accuracy gen   21.14
oneig_diversity_reasoning - accuracy gen   36.80
```

## Data Format

### Original Dataset Format

The OneIG original dataset is a CSV file (`OneIG-Bench.csv`), where each record contains:

```json
{
    "category": "Anime_Stylization",
    "id": "000",
    "prompt_en": "4boys, 5girls, multiple boys, multiple girls, ...",
    "type": "T, P",
    "prompt_length": "long",
    "class": "None"
}
```

| Field | Description |
| --- | --- |
| `category` | Prompt category: Anime_Stylization, Portrait, General Object, Text Rendering, Knowledge Reasoning, Multilingualism |
| `id` | Unique ID, maintained independently per category |
| `prompt_en` | Text-to-image prompt |
| `type` | Type marker: T (Text), P (Portrait), NP (Non-Portrait) |
| `prompt_length` | Prompt length: short, middle, long |
| `class` | Style category (optional): fauvism, watercolor, None |

### Image Directory Structure

Images for evaluation should be organized as follows:

```
IMAGE_DIR/
├── anime/                      # class_item directory
│   └── {model_name}/           # model name directory
│       ├── 000.png             # image file (first 3 chars of filename = sample_id)
│       ├── 001.png
│       └── ...
├── human/
│   └── {model_name}/
│       ├── 000.png
│       └── ...
├── object/
│   └── {model_name}/
│       └── ...
├── text/
│   └── {model_name}/
│       └── ...
└── reasoning/
    └── {model_name}/
        └── ...
```

class_item directories for each sub-task:

| Sub-task | EN Mode | ZH Mode (additional) |
| --- | --- | --- |
| Alignment | anime, human, object | multilingualism |
| Text | text | - |
| Reasoning | reasoning | - |
| Style | anime | - |
| Diversity | anime, human, object, text, reasoning | multilingualism |

### Grid Splitting

OneIG supports compositing multiple generated images into a grid for batch evaluation. The `IMAGE_GRIDS` config specifies the grid rows and columns:

| Grid Config | Meaning | Sub-images |
| --- | --- | --- |
| `"1,2"` | 1 row, 2 columns | 2 |
| `"2,2"` | 2 rows, 2 columns | 4 |
| `"1,4"` | 1 row, 4 columns | 4 |
| `"3,3"` | 3 rows, 3 columns | 9 |

During evaluation, grid images are automatically split into sub-images, each evaluated independently and averaged.

## Example Code

### Single Task Evaluation

Modify the `TASKS` list in the config file to include only the desired task:

```python
# Evaluate Alignment only
TASKS = ['alignment']
```

Run:

```bash
ais_bench ais_bench/configs/oneig_examples/oneig_full_eval.py -m eval
```

### Full Evaluation

```python
# Evaluate all 5 sub-tasks
TASKS = ['alignment', 'text', 'reasoning', 'style', 'diversity']
```

Run:

```bash
ais_bench ais_bench/configs/oneig_examples/oneig_full_eval.py -m eval
```

### Chinese Mode Evaluation

```python
ONEIG_ROOT = "/path/to/OneIG-Benchmark"
MODE = "ZH"                                    # Switch to Chinese mode
IMAGE_DIR = "/path/to/oneig/images_zh"         # Images generated from Chinese prompts
MODEL_NAMES = ["Qwen-Image"]
IMAGE_GRIDS = ["2,2"]
TASKS = ['alignment', 'text', 'reasoning', 'style', 'diversity']
```

### Multi-Model Comparison

```python
MODEL_NAMES = ["model_a", "model_b"]
IMAGE_GRIDS = ["2,2", "2,2"]                   # Must match MODEL_NAMES length
```

