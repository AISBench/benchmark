## VBench Local Cache Dependency List

- **Cache Root Directory**: Defaults to the environment variable `VBENCH_CACHE_DIR`; if not set, it falls back to `~/.cache/vbench`. The same variable name can also be set at the **top level** of an AISBench VBench example configuration (see "Specifying the Cache Directory in AISBench Configurations" below).
- **One-click Download Script**: `ais_bench/configs/vbench_examples/download_vbench_cache.sh` automatically downloads/prepares resources following the layout below.

### Target Directory Layout

The (default) `~/.cache/vbench/` should at least contain:

- `ViCLIP/ViClip-InternVid-10M-FLT.pth`
- `ViCLIP/bpe_simple_vocab_16e6.txt.gz`
- `aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth`
- `amt_model/amt-s.pth`
- `bert_model/bert-base-uncased/...` (Full snapshot of the HuggingFace BERT repo)
- `caption_model/tag2text_swin_14m.pth`
- `clip_model/ViT-B-32.pt`
- `clip_model/ViT-L-14.pt`
- `dino_model/dino_vitbase16_pretrain.pth`
- `dino_model/facebookresearch_dino_main/...` (Clone of the official DINO repo)
- `grit_model/grit_b_densecap_objectdet.pth`
- `pyiqa_model/musiq_spaq_ckpt-358bb6af.pth`
- `raft_model/models/raft-things.pth` (Plus other RAFT models extracted from the zip)
- `umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth`

### Dependencies and Download Sources

- **CLIP Models**
  - **Paths**: `clip_model/ViT-B-32.pt`, `clip_model/ViT-L-14.pt`
  - **Used by**: `background_consistency`, `appearance_style`, `aesthetic_quality`, etc.
  - **Sources**:
    - `ViT-B-32.pt`: `https://openaipublic.azureedge.net/clip/models/40d3657159.../ViT-B-32.pt`
    - `ViT-L-14.pt`: `https://openaipublic.azureedge.net/clip/models/b8cca3fd4.../ViT-L-14.pt`

- **UMT Model (Human Action)**
  - **Path**: `umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth`
  - **Used by**: `human_action`
  - **Source**: `https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/l16_ptk710_ftk710_ftk400_f16_res224.pth`

- **AMT-S Model (Motion Smoothness)**
  - **Path**: `amt_model/amt-s.pth`
  - **Used by**: `motion_smoothness`
  - **Source**: `https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth`

- **RAFT Optical Flow Model**
  - **Paths**:
    - Root directory: `raft_model/`
    - Main model: `raft_model/models/raft-things.pth`
  - **Used by**: `dynamic_degree`, `static_filter`, etc.
  - **Source (zip)**: `https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip`

- **DINO Model (subject_consistency, local mode)**
  - **Paths**:
    - Repo: `dino_model/facebookresearch_dino_main/`
    - Weights: `dino_model/dino_vitbase16_pretrain.pth`
  - **Used by**: `subject_consistency`
  - **Sources**:
    - Repo: `https://github.com/facebookresearch/dino`
    - Weights: `https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth`

- **Aesthetic Predictor (LAION)**
  - **Path**: `aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth`
  - **Used by**: `aesthetic_quality`
  - **Source**: `https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true`

- **MUSIQ / PyIQA (Image Quality)**
  - **Path**: `pyiqa_model/musiq_spaq_ckpt-358bb6af.pth`
  - **Used by**: `imaging_quality`
  - **Source**: `https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth`

- **GRiT Dense Captioning Model**
  - **Path**: `grit_model/grit_b_densecap_objectdet.pth`
  - **Used by**: `object_class`, `multiple_objects`, `color`, `spatial_relationship`
  - **Source**: `https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/grit_b_densecap_objectdet.pth`

- **Tag2Text Scene Description Model**
  - **Path**: `caption_model/tag2text_swin_14m.pth`
  - **Used by**: `scene`
  - **Source**: `https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/tag2text_swin_14m.pth`

- **ViCLIP Video-Text Model + BPE Vocab**
  - **Paths**:
    - Weights: `ViCLIP/ViClip-InternVid-10M-FLT.pth`
    - BPE: `ViCLIP/bpe_simple_vocab_16e6.txt.gz`
  - **Used by**: `temporal_style`, `overall_consistency`
  - **Sources**:
    - Weights: `https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth`
    - BPE: `https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz`

- **BERT base (bert-base-uncased)**
  - **Path**: `bert_model/bert-base-uncased/` (Full HF repo snapshot)
  - **Used by**: Text encoding parts of `Tag2Text` and `GRiT`
  - **Local Lookup Logic**:
    - First the directory pointed to by the `VBENCH_BERT_PATH` environment variable
    - Otherwise try `CACHE_DIR/bert_model/bert-base-uncased`
    - If neither exists, fall back to the HuggingFace hub id `bert-base-uncased`
  - **Recommended Download** (consistent with the script):
    - Requires `huggingface-cli`, e.g.:
      - `pip install "huggingface_hub[cli]"`
      - `huggingface-cli download bert-base-uncased --local-dir ~/.cache/vbench/bert_model/bert-base-uncased --local-dir-use-symlinks False`

### Usage

1. Make sure `wget` and `git` are installed. If you also need automatic BERT download, install `huggingface-cli` as well.
2. Run from the repository root:

```bash
bash ais_bench/configs/vbench_examples/download_vbench_cache.sh
```

3. To change the cache root directory, set it before running:

```bash
export VBENCH_CACHE_DIR=/your/custom/cache/dir
bash ais_bench/configs/vbench_examples/download_vbench_cache.sh
```

The script automatically skips existing files and is safe to run multiple times.

### Specifying the Cache Directory in AISBench Configurations

In a VBench example configuration (such as `eval_vbench_standard.py`), define the variable at the top level alongside `DATA_PATH`, e.g.:

```python
VBENCH_CACHE_DIR = "/your/custom/cache/dir"
```

The Python-style alias `vbench_cache_dir` is also supported; if both exist, `VBENCH_CACHE_DIR` takes precedence.

**Priority** (effective inside the evaluation subprocess that runs `VBenchEvalTask`, and only **before** vbench is imported for the first time):

1. If the configuration sets a **non-empty** `VBENCH_CACHE_DIR` or `vbench_cache_dir`, it is written to `os.environ['VBENCH_CACHE_DIR']` (with `~` and `$VAR` expanded) and **overrides** any existing same-name environment variable in this subprocess.
2. If not set in the configuration, the `VBENCH_CACHE_DIR` exported in the shell before launching `ais_bench` is used.
3. Otherwise, vbench falls back to the default `~/.cache/vbench`.

**Relationship with the One-click Script**: `download_vbench_cache.sh` only reads **shell environment variables** and does not read the Python configuration file above. To keep the download directory consistent with the evaluation, `export` the same `VBENCH_CACHE_DIR` before running the script, or specify the same absolute path in both places.

---

## Manual Download and Placement Guide (When the Script Fails)

When network or permission issues cause `download_vbench_cache.sh` to fail repeatedly, you can follow this section to **manually download each dependency and place it in the corresponding path**, bypassing the one-click script.

### Global Notes

- **Cache Root Directory `CACHE_DIR`**
  - If `VBENCH_CACHE_DIR` is not set: `CACHE_DIR=~/.cache/vbench`
  - If set: `CACHE_DIR=$VBENCH_CACHE_DIR`
- **Directory Preparation**: Before manual download, it is recommended to create the subdirectories first, e.g.:

```bash
export CACHE_DIR=${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}
mkdir -p "$CACHE_DIR"/{clip_model,umt_model,amt_model,raft_model,dino_model,aesthetic_model/emb_reader,pyiqa_model,grit_model,caption_model,ViCLIP,bert_model}
```

- **Hugging Face Mirror `HF_ENDPOINT` (Optional)**
  - All `https://huggingface.co/...` links can be sped up by replacing the prefix with a mirror (e.g., `https://hf-mirror.com`):
    - Original: `https://huggingface.co/xxx/yyy`
    - Mirror: `https://hf-mirror.com/xxx/yyy`

All "Target Path" entries below are relative to `CACHE_DIR`.

---

### 1. CLIP Models (ViT-B-32 / ViT-L-14)

- **Used by**: `background_consistency`, `appearance_style`, `aesthetic_quality`, etc.
- **Target Paths**:
  - `clip_model/ViT-B-32.pt`
  - `clip_model/ViT-L-14.pt`
- **Official Download Links**:
  - `ViT-B-32.pt`:
    `https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt`
  - `ViT-L-14.pt`:
    `https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt`
- **Command-line Example**:

```bash
export CACHE_DIR=${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}
mkdir -p "$CACHE_DIR/clip_model"

wget -O "$CACHE_DIR/clip_model/ViT-B-32.pt" \
  "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"

wget -O "$CACHE_DIR/clip_model/ViT-L-14.pt" \
  "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
```

- **Browser Method**: Open both links in a browser to download, then move the files to:
  - `ViT-B-32.pt` → `$CACHE_DIR/clip_model/ViT-B-32.pt`
  - `ViT-L-14.pt` → `$CACHE_DIR/clip_model/ViT-L-14.pt`

---

### 2. UMT Model (Human Action)

- **Used by**: `human_action`
- **Target Path**: `umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth`
- **Official Download Link**:
  - Original: `https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/l16_ptk710_ftk710_ftk400_f16_res224.pth`
  - With a mirror, replace the prefix with the mirror site, e.g.:
    `https://hf-mirror.com/OpenGVLab/VBench_Used_Models/resolve/main/l16_ptk710_ftk710_ftk400_f16_res224.pth`
- **Command-line Example**:

```bash
export CACHE_DIR=${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}
mkdir -p "$CACHE_DIR/umt_model"

wget -O "$CACHE_DIR/umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth" \
  "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/l16_ptk710_ftk710_ftk400_f16_res224.pth"
```

- **Browser Method**: Open the link in a browser, then move to:
  `$CACHE_DIR/umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth`

---

### 3. AMT-S Model (Motion Smoothness)

- **Used by**: `motion_smoothness`
- **Target Path**: `amt_model/amt-s.pth`
- **Official Download Link**:
  - Original: `https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth`
- **Command-line Example**:

```bash
export CACHE_DIR=${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}
mkdir -p "$CACHE_DIR/amt_model"

wget -O "$CACHE_DIR/amt_model/amt-s.pth" \
  "https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth"
```

- **Browser Method**: After downloading, move to: `$CACHE_DIR/amt_model/amt-s.pth`

---

### 4. RAFT Optical Flow Model

- **Used by**: `dynamic_degree`, `static_filter`, etc.
- **Target Root Directory**: `raft_model/`
- **Key File**: `raft_model/models/raft-things.pth`
- **Official Download Link (zip)**:
  `https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip`
- **Command-line Example**:

```bash
export CACHE_DIR=${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}
mkdir -p "$CACHE_DIR/raft_model"

wget -O "$CACHE_DIR/raft_model/models.zip" \
  "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip"

cd "$CACHE_DIR/raft_model"
unzip -o models.zip
rm -f models.zip
```

- **Browser Method**:
  1. Download `models.zip` in a browser.
  2. Place `models.zip` under `$CACHE_DIR/raft_model/`.
  3. Extract in that directory: `unzip models.zip`.
  4. Confirm `$CACHE_DIR/raft_model/models/raft-things.pth` exists, then the zip can be deleted.

---

### 5. DINO Model (subject_consistency, Local Mode)

- **Used by**: `subject_consistency`
- **Target Paths**:
  - Repo: `dino_model/facebookresearch_dino_main/`
  - Weights: `dino_model/dino_vitbase16_pretrain.pth`
- **Repo URL**: `https://github.com/facebookresearch/dino`
- **Weights Download Link**:
  `https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth`
- **Command-line Example (Recommended)**:

```bash
export CACHE_DIR=${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}
mkdir -p "$CACHE_DIR/dino_model"

cd "$CACHE_DIR/dino_model"
git clone https://github.com/facebookresearch/dino facebookresearch_dino_main || true

wget -O "$CACHE_DIR/dino_model/dino_vitbase16_pretrain.pth" \
  "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
```

- **Browser Method**:
  1. Download the dino repository zip via Git GUI or browser, extract it, rename the directory to `facebookresearch_dino_main`, and place it under `$CACHE_DIR/dino_model/`.
  2. Open the weights link in a browser, download it, and move to `$CACHE_DIR/dino_model/dino_vitbase16_pretrain.pth`.

---

### 6. Aesthetic Predictor (LAION)

- **Used by**: `aesthetic_quality`
- **Target Path**: `aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth`
- **Official Download Link**:
  `https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true`
- **Command-line Example**:

```bash
export CACHE_DIR=${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}
mkdir -p "$CACHE_DIR/aesthetic_model/emb_reader"

wget -O "$CACHE_DIR/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth" \
  "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true"
```

- **Browser Method**: Open the link (make sure `?raw=true` is included), then move the download to the target path.

---

### 7. MUSIQ / PyIQA Image Quality Model

- **Used by**: `imaging_quality`
- **Target Path**: `pyiqa_model/musiq_spaq_ckpt-358bb6af.pth`
- **Official Download Link**:
  `https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth`
- **Command-line Example**:

```bash
export CACHE_DIR=${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}
mkdir -p "$CACHE_DIR/pyiqa_model"

wget -O "$CACHE_DIR/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth" \
  "https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth"
```

- **Browser Method**: After downloading, move to `$CACHE_DIR/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth`.

---

### 8. GRiT Dense Captioning Model

- **Used by**: `object_class`, `multiple_objects`, `color`, `spatial_relationship`
- **Target Path**: `grit_model/grit_b_densecap_objectdet.pth`
- **Official Download Link**:
  - Original: `https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/grit_b_densecap_objectdet.pth`
- **Command-line Example**:

```bash
export CACHE_DIR=${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}
mkdir -p "$CACHE_DIR/grit_model"

wget -O "$CACHE_DIR/grit_model/grit_b_densecap_objectdet.pth" \
  "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/grit_b_densecap_objectdet.pth"
```

- **Browser Method**: After downloading, move to `$CACHE_DIR/grit_model/grit_b_densecap_objectdet.pth`.

---

### 9. Tag2Text Scene Description Model

- **Used by**: `scene`
- **Target Path**: `caption_model/tag2text_swin_14m.pth`
- **Official Download Link**:
  - Original: `https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/tag2text_swin_14m.pth`
- **Command-line Example**:

```bash
export CACHE_DIR=${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}
mkdir -p "$CACHE_DIR/caption_model"

wget -O "$CACHE_DIR/caption_model/tag2text_swin_14m.pth" \
  "https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/tag2text_swin_14m.pth"
```

- **Browser Method**: After downloading, move to `$CACHE_DIR/caption_model/tag2text_swin_14m.pth`.

---

### 10. ViCLIP Video-Text Model + BPE Vocab

- **Used by**: `temporal_style`, `overall_consistency`
- **Target Paths**:
  - Weights: `ViCLIP/ViClip-InternVid-10M-FLT.pth`
  - Vocab: `ViCLIP/bpe_simple_vocab_16e6.txt.gz`
    (If multiple copies are needed, manually copy them as `bpe_simple_vocab_16e6.txt.gz.{1,2,3}`.)
- **Official Download Links**:
  - Weights (original):
    `https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth`
  - BPE:
    `https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz`
- **Command-line Example**:

```bash
export CACHE_DIR=${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}
mkdir -p "$CACHE_DIR/ViCLIP"

wget -O "$CACHE_DIR/ViCLIP/ViClip-InternVid-10M-FLT.pth" \
  "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth"

wget -O "$CACHE_DIR/ViCLIP/bpe_simple_vocab_16e6.txt.gz" \
  "https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz"
```

---

### 11. BERT base (bert-base-uncased)

- **Used by**: Text encoding for `Tag2Text` and `GRiT`
- **Target Path**: `bert_model/bert-base-uncased/` (Full HF repo snapshot)
- **Lookup Logic Recap**:
  - First use the directory pointed to by the environment variable `VBENCH_BERT_PATH`;
  - Otherwise try `CACHE_DIR/bert_model/bert-base-uncased`;
  - If still missing, download from Hugging Face online.

#### Method A: Use huggingface-cli (Recommended)

1. Install the tool:

```bash
pip install "huggingface_hub[cli]"
```

2. Log in (if necessary, optional): `huggingface-cli login`

3. Download to the cache directory:

```bash
export CACHE_DIR=${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}
mkdir -p "$CACHE_DIR/bert_model/bert-base-uncased"

huggingface-cli download bert-base-uncased \
  --local-dir "$CACHE_DIR/bert_model/bert-base-uncased" \
  --local-dir-use-symlinks False
```

4. To point `VBENCH_BERT_PATH` to that directory:

```bash
export VBENCH_BERT_PATH="$CACHE_DIR/bert_model/bert-base-uncased"
```

#### Method B: Browser or Other Means

1. In a browser, visit `https://huggingface.co/bert-base-uncased` and download the entire model repository (e.g., via "Download files" or git lfs).
2. Rename the directory containing files such as `config.json`, `pytorch_model.bin`, and `vocab.txt` to `bert-base-uncased`, and place it under:

```text
$CACHE_DIR/bert_model/bert-base-uncased/
```

3. Optional: Set `VBENCH_BERT_PATH` to point to that directory.

---

### Notes: Coexistence with the One-click Script

- After completing manual downloads, you **may choose not to run** `scripts/download_vbench_cache.sh` again. As long as the paths and filenames match this guide, VBench can read them normally.
- If the one-click script is run later, it adds `.done` marker files alongside the downloads to skip them next time; this does not overwrite content you placed manually.
- If you use a Hugging Face mirror, simply replace the prefix in the links above with the mirror domain; the rest of the paths and placement remain unchanged.
