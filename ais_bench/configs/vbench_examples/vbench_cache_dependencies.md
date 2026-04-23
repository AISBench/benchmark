## VBench 本地缓存依赖清单

- **缓存根目录**：默认使用环境变量 `VBENCH_CACHE_DIR`，若未设置，则为 `~/.cache/vbench`。另支持在 AISBench 的 VBench 示例配置**顶层**设置同名变量（见下文「在 AISBench 配置中指定缓存目录」）。
- **一键下载脚本**：[`ais_bench/configs/vbench_examples/download_vbench_cache.sh`](download_vbench_cache.sh) 会自动按下述结构下载/准备资源。

### 目录结构目标

最终希望在（默认）`~/.cache/vbench/` 下至少包含：

- `ViCLIP/ViClip-InternVid-10M-FLT.pth`
- `ViCLIP/bpe_simple_vocab_16e6.txt.gz`
- `aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth`
- `amt_model/amt-s.pth`
- `bert_model/bert-base-uncased/...`（HuggingFace BERT 仓库完整快照）
- `caption_model/tag2text_swin_14m.pth`
- `clip_model/ViT-B-32.pt`
- `clip_model/ViT-L-14.pt`
- `dino_model/dino_vitbase16_pretrain.pth`
- `dino_model/facebookresearch_dino_main/...`（DINO 官方仓库克隆）
- `grit_model/grit_b_densecap_objectdet.pth`
- `pyiqa_model/musiq_spaq_ckpt-358bb6af.pth`
- `raft_model/models/raft-things.pth`（以及 zip 解压出的其它 RAFT 模型）
- `umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth`

### 逐项依赖与下载来源

- **CLIP 模型**
  - **路径**：`clip_model/ViT-B-32.pt`，`clip_model/ViT-L-14.pt`
  - **用途**：`background_consistency`、`appearance_style`、`aesthetic_quality` 等
  - **来源**：
    - `ViT-B-32.pt`：`https://openaipublic.azureedge.net/clip/models/40d3657159.../ViT-B-32.pt`
    - `ViT-L-14.pt`：`https://openaipublic.azureedge.net/clip/models/b8cca3fd4.../ViT-L-14.pt`

- **UMT 模型（人类动作）**
  - **路径**：`umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth`
  - **用途**：`human_action`
  - **来源**：`https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/l16_ptk710_ftk710_ftk400_f16_res224.pth`

- **AMT-S 模型（运动平滑度）**
  - **路径**：`amt_model/amt-s.pth`
  - **用途**：`motion_smoothness`
  - **来源**：`https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth`

- **RAFT 光流模型**
  - **路径**：
    - 根目录：`raft_model/`
    - 主模型：`raft_model/models/raft-things.pth`
  - **用途**：`dynamic_degree`、`static_filter` 等
  - **来源（zip）**：`https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip`

- **DINO 模型（subject_consistency，本地模式）**
  - **路径**：
    - 仓库：`dino_model/facebookresearch_dino_main/`
    - 权重：`dino_model/dino_vitbase16_pretrain.pth`
  - **用途**：`subject_consistency` 维度
  - **来源**：
    - 仓库：`https://github.com/facebookresearch/dino`
    - 权重：`https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth`

- **Aesthetic Predictor（LAION）**
  - **路径**：`aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth`
  - **用途**：`aesthetic_quality`
  - **来源**：`https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true`

- **MUSIQ / PyIQA（图像质量）**
  - **路径**：`pyiqa_model/musiq_spaq_ckpt-358bb6af.pth`
  - **用途**：`imaging_quality`
  - **来源**：`https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth`

- **GRiT 稠密标注模型**
  - **路径**：`grit_model/grit_b_densecap_objectdet.pth`
  - **用途**：`object_class`、`multiple_objects`、`color`、`spatial_relationship`
  - **来源**：`https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/grit_b_densecap_objectdet.pth`

- **Tag2Text 场景描述模型**
  - **路径**：`caption_model/tag2text_swin_14m.pth`
  - **用途**：`scene`
  - **来源**：`https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/tag2text_swin_14m.pth`

- **ViCLIP 视频-文本模型 + BPE 词表**
  - **路径**：
    - 权重：`ViCLIP/ViClip-InternVid-10M-FLT.pth`
    - BPE：`ViCLIP/bpe_simple_vocab_16e6.txt.gz`
  - **用途**：`temporal_style`、`overall_consistency`
  - **来源**：
    - 权重：`https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth`
    - BPE：`https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz`

- **BERT base（bert-base-uncased）**
  - **路径**：`bert_model/bert-base-uncased/`（完整 HF 仓库）
  - **用途**：`Tag2Text` 与 `GRiT` 的文本编码部分
  - **本地搜索逻辑**：
    - 优先 `VBENCH_BERT_PATH` 环境变量目录
    - 否则尝试 `CACHE_DIR/bert_model/bert-base-uncased`
    - 若都不存在，则回落到 HuggingFace hub id `bert-base-uncased`
  - **推荐下载方式**（与脚本一致）：
    - 需安装 `huggingface-cli`，例如：
      - `pip install "huggingface_hub[cli]"`
      - `huggingface-cli download bert-base-uncased --local-dir ~/.cache/vbench/bert_model/bert-base-uncased --local-dir-use-symlinks False`

### 使用方式

1. 确认已安装 `wget`、`git`，若需要自动下载 BERT，还需安装 `huggingface-cli`。
2. 在仓库根目录执行：

```bash
bash ais_bench/configs/vbench_examples/download_vbench_cache.sh
```

3. 若需修改缓存根目录，可在执行前设置：

```bash
export VBENCH_CACHE_DIR=/your/custom/cache/dir
bash ais_bench/configs/vbench_examples/download_vbench_cache.sh
```

脚本会自动跳过已存在的文件，多次执行是安全的。

### 在 AISBench 配置中指定缓存目录

在 VBench 示例配置（如 `eval_vbench_standard.py`）中与 `DATA_PATH` 同级定义顶层变量即可，例如：

```python
VBENCH_CACHE_DIR = "/your/custom/cache/dir"
```

也支持 Python 风格的别名 `vbench_cache_dir`；若二者均存在，以 `VBENCH_CACHE_DIR` 为准。

**优先级**（在运行 `VBenchEvalTask` 的测评子进程内、且仅在**首次** `import vbench` 之前生效）：

1. 配置里若设置了**非空**的 `VBENCH_CACHE_DIR` 或 `vbench_cache_dir`，则写入 `os.environ['VBENCH_CACHE_DIR']`（展开 `~` 与 `$VAR`），并**覆盖**该子进程内已有的同名环境变量。
2. 若未在配置中设置，则沿用启动 `ais_bench` 前已在 shell 中 `export` 的 `VBENCH_CACHE_DIR`。
3. 若仍无，则由 vbench 默认使用 `~/.cache/vbench`。

**与一键脚本的关系**：`download_vbench_cache.sh` 只读取 **shell 环境变量**，不会读取上述 Python 配置文件。若希望下载目录与测评一致，请在执行脚本前 `export` 相同的 `VBENCH_CACHE_DIR`，或在两处分别指定同一绝对路径。

---

## 手动下载与放置指南（脚本失败时）

当网络或权限问题导致 `download_vbench_cache.sh` 多次失败时，可以根据本节说明**手动下载每一份依赖并放到对应路径**，从而绕过一键脚本。

### 全局说明

- **缓存根目录 `CACHE_DIR`**
  - 若未设置 `VBENCH_CACHE_DIR`：`CACHE_DIR=~/.cache/vbench`
  - 若已设置：`CACHE_DIR=$VBENCH_CACHE_DIR`
- **目录准备**：手动下载前，建议先创建子目录，例如：

```bash
export CACHE_DIR=${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}
mkdir -p "$CACHE_DIR"/{clip_model,umt_model,amt_model,raft_model,dino_model,aesthetic_model/emb_reader,pyiqa_model,grit_model,caption_model,ViCLIP,bert_model}
```

- **Hugging Face 镜像 `HF_ENDPOINT`（可选）**
  - 所有 `https://huggingface.co/...` 的链接，都可以通过将前缀替换为镜像（例如 `https://hf-mirror.com`）来加速：
    - 原始：`https://huggingface.co/xxx/yyy`
    - 镜像：`https://hf-mirror.com/xxx/yyy`

以下所有“目标路径”默认都是相对于 `CACHE_DIR`。

---

### 1. CLIP 模型（ViT-B-32 / ViT-L-14）

- **用途**：`background_consistency`、`appearance_style`、`aesthetic_quality` 等
- **目标路径**：
  - `clip_model/ViT-B-32.pt`
  - `clip_model/ViT-L-14.pt`
- **官方下载链接**：
  - `ViT-B-32.pt`：
    `https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt`
  - `ViT-L-14.pt`：
    `https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt`
- **命令行示例**：

```bash
export CACHE_DIR=${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}
mkdir -p "$CACHE_DIR/clip_model"

wget -O "$CACHE_DIR/clip_model/ViT-B-32.pt" \
  "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"

wget -O "$CACHE_DIR/clip_model/ViT-L-14.pt" \
  "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
```

- **浏览器方式**：分别用浏览器打开上述两个链接下载，然后把文件移动为：
  - `ViT-B-32.pt` → `$CACHE_DIR/clip_model/ViT-B-32.pt`
  - `ViT-L-14.pt` → `$CACHE_DIR/clip_model/ViT-L-14.pt`

---

### 2. UMT 模型（人类动作）

- **用途**：`human_action`
- **目标路径**：`umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth`
- **官方下载链接**：
  - 原始：`https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/l16_ptk710_ftk710_ftk400_f16_res224.pth`
  - 如使用镜像，将前缀替换为镜像站，例如：
    `https://hf-mirror.com/OpenGVLab/VBench_Used_Models/resolve/main/l16_ptk710_ftk710_ftk400_f16_res224.pth`
- **命令行示例**：

```bash
export CACHE_DIR=${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}
mkdir -p "$CACHE_DIR/umt_model"

wget -O "$CACHE_DIR/umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth" \
  "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/l16_ptk710_ftk710_ftk400_f16_res224.pth"
```

- **浏览器方式**：用浏览器打开上述链接下载，然后移动为：
  `$CACHE_DIR/umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth`

---

### 3. AMT-S 模型（运动平滑度）

- **用途**：`motion_smoothness`
- **目标路径**：`amt_model/amt-s.pth`
- **官方下载链接**：
  - 原始：`https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth`
- **命令行示例**：

```bash
export CACHE_DIR=${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}
mkdir -p "$CACHE_DIR/amt_model"

wget -O "$CACHE_DIR/amt_model/amt-s.pth" \
  "https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth"
```

- **浏览器方式**：下载后移动到：`$CACHE_DIR/amt_model/amt-s.pth`

---

### 4. RAFT 光流模型

- **用途**：`dynamic_degree`、`static_filter` 等
- **目标根目录**：`raft_model/`
- **关键文件**：`raft_model/models/raft-things.pth`
- **官方下载链接（zip）**：
  `https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip`
- **命令行示例**：

```bash
export CACHE_DIR=${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}
mkdir -p "$CACHE_DIR/raft_model"

wget -O "$CACHE_DIR/raft_model/models.zip" \
  "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip"

cd "$CACHE_DIR/raft_model"
unzip -o models.zip
rm -f models.zip
```

- **浏览器方式**：
  1. 浏览器下载 `models.zip`。
  2. 将 `models.zip` 放到 `$CACHE_DIR/raft_model/` 下。
  3. 在该目录执行解压：`unzip models.zip`。
  4. 确认存在 `$CACHE_DIR/raft_model/models/raft-things.pth`，之后可删除 zip。

---

### 5. DINO 模型（subject_consistency，本地模式）

- **用途**：`subject_consistency`
- **目标路径**：
  - 仓库：`dino_model/facebookresearch_dino_main/`
  - 权重：`dino_model/dino_vitbase16_pretrain.pth`
- **仓库地址**：`https://github.com/facebookresearch/dino`
- **权重下载链接**：
  `https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth`
- **命令行示例（推荐）**：

```bash
export CACHE_DIR=${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}
mkdir -p "$CACHE_DIR/dino_model"

cd "$CACHE_DIR/dino_model"
git clone https://github.com/facebookresearch/dino facebookresearch_dino_main || true

wget -O "$CACHE_DIR/dino_model/dino_vitbase16_pretrain.pth" \
  "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
```

- **浏览器方式**：
  1. 使用 Git GUI 或浏览器下载 dino 仓库 zip，解压后重命名目录为 `facebookresearch_dino_main`，放在 `$CACHE_DIR/dino_model/` 下。
  2. 浏览器打开权重链接下载，并移动为 `$CACHE_DIR/dino_model/dino_vitbase16_pretrain.pth`。

---

### 6. Aesthetic Predictor（LAION）

- **用途**：`aesthetic_quality`
- **目标路径**：`aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth`
- **官方下载链接**：
  `https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true`
- **命令行示例**：

```bash
export CACHE_DIR=${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}
mkdir -p "$CACHE_DIR/aesthetic_model/emb_reader"

wget -O "$CACHE_DIR/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth" \
  "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true"
```

- **浏览器方式**：打开链接（确保带有 `?raw=true`），下载后移动到目标路径。

---

### 7. MUSIQ / PyIQA 图像质量模型

- **用途**：`imaging_quality`
- **目标路径**：`pyiqa_model/musiq_spaq_ckpt-358bb6af.pth`
- **官方下载链接**：
  `https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth`
- **命令行示例**：

```bash
export CACHE_DIR=${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}
mkdir -p "$CACHE_DIR/pyiqa_model"

wget -O "$CACHE_DIR/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth" \
  "https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth"
```

- **浏览器方式**：下载后移动为 `$CACHE_DIR/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth`。

---

### 8. GRiT 稠密标注模型

- **用途**：`object_class`、`multiple_objects`、`color`、`spatial_relationship`
- **目标路径**：`grit_model/grit_b_densecap_objectdet.pth`
- **官方下载链接**：
  - 原始：`https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/grit_b_densecap_objectdet.pth`
- **命令行示例**：

```bash
export CACHE_DIR=${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}
mkdir -p "$CACHE_DIR/grit_model"

wget -O "$CACHE_DIR/grit_model/grit_b_densecap_objectdet.pth" \
  "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/grit_b_densecap_objectdet.pth"
```

- **浏览器方式**：下载后移动为 `$CACHE_DIR/grit_model/grit_b_densecap_objectdet.pth`。

---

### 9. Tag2Text 场景描述模型

- **用途**：`scene`
- **目标路径**：`caption_model/tag2text_swin_14m.pth`
- **官方下载链接**：
  - 原始：`https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/tag2text_swin_14m.pth`
- **命令行示例**：

```bash
export CACHE_DIR=${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}
mkdir -p "$CACHE_DIR/caption_model"

wget -O "$CACHE_DIR/caption_model/tag2text_swin_14m.pth" \
  "https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/tag2text_swin_14m.pth"
```

- **浏览器方式**：下载后移动为 `$CACHE_DIR/caption_model/tag2text_swin_14m.pth`。

---

### 10. ViCLIP 视频-文本模型 + BPE 词表

- **用途**：`temporal_style`、`overall_consistency`
- **目标路径**：
  - 权重：`ViCLIP/ViClip-InternVid-10M-FLT.pth`
  - 词表：`ViCLIP/bpe_simple_vocab_16e6.txt.gz`
    （如需多份副本，可手动复制为 `bpe_simple_vocab_16e6.txt.gz.{1,2,3}`）
- **官方下载链接**：
  - 权重（原始）：
    `https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth`
  - BPE：
    `https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz`
- **命令行示例**：

```bash
export CACHE_DIR=${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}
mkdir -p "$CACHE_DIR/ViCLIP"

wget -O "$CACHE_DIR/ViCLIP/ViClip-InternVid-10M-FLT.pth" \
  "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth"

wget -O "$CACHE_DIR/ViCLIP/bpe_simple_vocab_16e6.txt.gz" \
  "https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz"
```

---

### 11. BERT base（bert-base-uncased）

- **用途**：`Tag2Text` 与 `GRiT` 文本编码
- **目标路径**：`bert_model/bert-base-uncased/`（完整 HF 仓库快照）
- **搜索逻辑回顾**：
  - 优先使用环境变量 `VBENCH_BERT_PATH` 指向的目录；
  - 否则尝试 `CACHE_DIR/bert_model/bert-base-uncased`；
  - 若仍不存在，则从 Hugging Face 在线下载。

#### 方式 A：使用 huggingface-cli（推荐）

1. 安装工具：

```bash
pip install "huggingface_hub[cli]"
```

2. 登录（如必要，可选）：`huggingface-cli login`

3. 下载到缓存目录：

```bash
export CACHE_DIR=${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}
mkdir -p "$CACHE_DIR/bert_model/bert-base-uncased"

huggingface-cli download bert-base-uncased \
  --local-dir "$CACHE_DIR/bert_model/bert-base-uncased" \
  --local-dir-use-symlinks False
```

4. 如希望通过 `VBENCH_BERT_PATH` 指定该目录：

```bash
export VBENCH_BERT_PATH="$CACHE_DIR/bert_model/bert-base-uncased"
```

#### 方式 B：浏览器或其他方式

1. 在浏览器中访问 `https://huggingface.co/bert-base-uncased`，下载整个模型仓库（例如使用 “Download files” 或 git lfs）。
2. 将包含 `config.json`、`pytorch_model.bin`、`vocab.txt` 等文件的目录重命名为 `bert-base-uncased`，并放到：

```text
$CACHE_DIR/bert_model/bert-base-uncased/
```

3. 可选：设置 `VBENCH_BERT_PATH` 指向该目录。

---

### 备注：与一键脚本的配合

- 手动下载完成后，**可以选择不再运行** `scripts/download_vbench_cache.sh`，只要路径和文件名与本说明一致，VBench 即可正常读取。
- 如之后仍运行一键脚本，它会在文件旁边补充 `.done` 标记文件，用于下次跳过下载；这不会覆盖你已经手动放置的内容。
- 若你使用 Hugging Face 镜像站，只需在上述链接中将前缀替换为镜像域名即可，其余路径和放置方式保持不变。