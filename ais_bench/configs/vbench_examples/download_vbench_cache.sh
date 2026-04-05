#!/usr/bin/env bash

set -euo pipefail

# 一键下载 VBench 所需权重与资源到本地缓存目录。
# 默认目录为 ~/.cache/vbench，可通过环境变量 VBENCH_CACHE_DIR 覆盖。
# 如需代理，可在外部设置 HTTP_PROXY/HTTPS_PROXY/ALL_PROXY 等环境变量。

CACHE_DIR="${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}"
echo "Using VBench cache dir: ${CACHE_DIR}"

mkdir -p "${CACHE_DIR}"

# 如果设置了 HF_ENDPOINT，则将 huggingface.co 的直链替换为对应镜像前缀
HF_ENDPOINT_TRIMMED="${HF_ENDPOINT:-}"
HF_ENDPOINT_TRIMMED="${HF_ENDPOINT_TRIMMED%/}"

rewrite_hf_url() {
  local url="$1"
  # 仅对 huggingface.co 的链接进行替换
  if [[ -n "${HF_ENDPOINT_TRIMMED}" && "${url}" == https://huggingface.co/* ]]; then
    echo "${HF_ENDPOINT_TRIMMED}${url#https://huggingface.co}"
  else
    echo "${url}"
  fi
}

download_if_not_exists() {
  local url="$1"
  local target_path="$2"
  local target_dir
  target_dir="$(dirname "${target_path}")"
  local done_marker="${target_path}.done"

  mkdir -p "${target_dir}"

  # 只有在「文件存在且完成标记存在」时才跳过，避免上次 Ctrl-C 留下不完整文件
  if [[ -f "${target_path}" && -f "${done_marker}" ]]; then
    echo "[skip] ${target_path} already exists (with .done)."
    return 0
  fi

  if [[ -f "${target_path}" && ! -f "${done_marker}" ]]; then
    echo "[warn] ${target_path} 存在但缺少完成标记，可能是上次中断产生的残留，重新下载..."
    rm -f "${target_path}"
  fi

  echo "[wget] ${url}"
  wget -O "${target_path}" "${url}"
  touch "${done_marker}"
}

hf_download_if_not_exists() {
  local repo_id="$1"
  local remote_path="$2"
  local target_path="$3"
  local target_dir
  target_dir="$(dirname "${target_path}")"
  local done_marker="${target_path}.done"

  mkdir -p "${target_dir}"

  # 使用 .done 标记判断是否真正完成
  if [[ -f "${target_path}" && -f "${done_marker}" ]]; then
    echo "[skip] ${target_path} already exists (with .done)."
    return 0
  fi

  if [[ -f "${target_path}" && ! -f "${done_marker}" ]]; then
    echo "[warn] ${target_path} 存在但缺少完成标记，可能是上次中断产生的残留，重新下载..."
    rm -f "${target_path}"
  fi

  echo "[hf-api] ${repo_id}:${remote_path} -> ${target_path}"

  # 直接用 huggingface_hub Python API 下载，绕过 huggingface-cli / commands 模块
  python - "$repo_id" "$remote_path" "$target_dir" << 'PY'
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.stderr.write(
        "[error] Python 环境未安装 huggingface_hub，无法执行下载。\n"
        "        请先运行: pip install \"huggingface_hub[cli]\" 并完成登录。\n"
    )
    sys.exit(1)

repo_id = sys.argv[1]
remote_path = sys.argv[2]
target_dir = Path(sys.argv[3])

target_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=repo_id,
    local_dir=str(target_dir),
    local_dir_use_symlinks=False,
    allow_patterns=[remote_path],  # 只拉取指定文件
)
PY

  touch "${done_marker}"
}

### 1. CLIP 模型（ViT-B-32 / ViT-L-14）
CLIP_DIR="${CACHE_DIR}/clip_model"
mkdir -p "${CLIP_DIR}"

download_if_not_exists \
  "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt" \
  "${CLIP_DIR}/ViT-B-32.pt"

download_if_not_exists \
  "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt" \
  "${CLIP_DIR}/ViT-L-14.pt"

### 2. UMT 模型
UMT_DIR="${CACHE_DIR}/umt_model"
mkdir -p "${UMT_DIR}"
hf_download_if_not_exists \
  "OpenGVLab/VBench_Used_Models" \
  "l16_ptk710_ftk710_ftk400_f16_res224.pth" \
  "${UMT_DIR}/l16_ptk710_ftk710_ftk400_f16_res224.pth"

### 3. AMT-S 模型（运动平滑度）
AMT_DIR="${CACHE_DIR}/amt_model"
mkdir -p "${AMT_DIR}"
download_if_not_exists \
  "$(rewrite_hf_url "https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth")" \
  "${AMT_DIR}/amt-s.pth"

### 4. RAFT 光流模型（会解压出多个 .pth）
RAFT_ROOT="${CACHE_DIR}/raft_model"
RAFT_MODELS_DIR="${RAFT_ROOT}/models"
mkdir -p "${RAFT_ROOT}"

if [[ -f "${RAFT_MODELS_DIR}/raft-things.pth" ]]; then
  echo "[skip] RAFT models already exist under ${RAFT_MODELS_DIR}."
else
  echo "[wget] RAFT models.zip"
  curl -k "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip" -o "${RAFT_ROOT}/models.zip"
  unzip -o "${RAFT_ROOT}/models.zip" -d "${RAFT_ROOT}"
  rm -f "${RAFT_ROOT}/models.zip"
fi

### 5. DINO 模型（本地模式）
DINO_ROOT="${CACHE_DIR}/dino_model"
DINO_REPO_DIR="${DINO_ROOT}/facebookresearch_dino_main"
DINO_CKPT_PATH="${DINO_ROOT}/dino_vitbase16_pretrain.pth"

mkdir -p "${DINO_ROOT}"
if [[ ! -d "${DINO_REPO_DIR}" ]]; then
  echo "[git] clone facebookresearch/dino -> ${DINO_REPO_DIR}"
  git clone https://github.com/facebookresearch/dino "${DINO_REPO_DIR}"
else
  echo "[skip] DINO repo already exists at ${DINO_REPO_DIR}."
fi

download_if_not_exists \
  "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth" \
  "${DINO_CKPT_PATH}"

### 6. Aesthetic Predictor 线性头
AES_DIR="${CACHE_DIR}/aesthetic_model/emb_reader"
mkdir -p "${AES_DIR}"
download_if_not_exists \
  "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true" \
  "${AES_DIR}/sa_0_4_vit_l_14_linear.pth"

### 7. MUSIQ / PyIQA 图像质量模型
PYIQA_DIR="${CACHE_DIR}/pyiqa_model"
mkdir -p "${PYIQA_DIR}"
download_if_not_exists \
  "https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth" \
  "${PYIQA_DIR}/musiq_spaq_ckpt-358bb6af.pth"

### 8. GRiT 稠密描述与检测模型
GRIT_DIR="${CACHE_DIR}/grit_model"
mkdir -p "${GRIT_DIR}"
hf_download_if_not_exists \
  "OpenGVLab/VBench_Used_Models" \
  "grit_b_densecap_objectdet.pth" \
  "${GRIT_DIR}/grit_b_densecap_objectdet.pth"

### 9. Tag2Text 场景描述模型
CAPTION_DIR="${CACHE_DIR}/caption_model"
mkdir -p "${CAPTION_DIR}"
download_if_not_exists \
  "$(rewrite_hf_url "https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/tag2text_swin_14m.pth")" \
  "${CAPTION_DIR}/tag2text_swin_14m.pth"

### 10. ViCLIP 视频-文本模型 + BPE 词表
VICLIP_DIR="${CACHE_DIR}/ViCLIP"
mkdir -p "${VICLIP_DIR}"

hf_download_if_not_exists \
  "OpenGVLab/VBench_Used_Models" \
  "ViClip-InternVid-10M-FLT.pth" \
  "${VICLIP_DIR}/ViClip-InternVid-10M-FLT.pth"

VICLIP_BPE_PATH="${VICLIP_DIR}/bpe_simple_vocab_16e6.txt.gz"
download_if_not_exists \
  "https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz" \
  "${VICLIP_BPE_PATH}"


### 11. BERT base 模型（用于 Tag2Text / GRiT 文本）
BERT_DIR="${CACHE_DIR}/bert_model/bert-base-uncased"
mkdir -p "${BERT_DIR}"

if [[ -d "${BERT_DIR}/.git" || -f "${BERT_DIR}/config.json" ]]; then
  echo "[skip] bert-base-uncased already exists at ${BERT_DIR}."
else
  echo "[hf-api] downloading bert-base-uncased to ${BERT_DIR} ..."
  python - "${BERT_DIR}" << 'PY'
import sys
import re
import time
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError
except ImportError:
    sys.stderr.write(
        "[warn] Python 环境未安装 huggingface_hub，跳过自动下载 bert-base-uncased。\n"
        "       请先运行: pip install \"huggingface_hub[cli]\" 并完成登录后再重试。\n"
    )
    sys.exit(0)

target_dir = Path(sys.argv[1])
target_dir.mkdir(parents=True, exist_ok=True)

max_attempts = 3
for attempt in range(1, max_attempts + 1):
    try:
        snapshot_download(
            repo_id="bert-base-uncased",
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
        break
    except HfHubHTTPError as e:
        is_429 = False
        if getattr(e, "response", None) is not None:
            is_429 = getattr(e.response, "status_code", None) == 429
        if not is_429 and "429" in str(e):
            is_429 = True
        if is_429:
            retry_after = 120
            resp = getattr(e, "response", None)
            if resp is not None and getattr(resp, "headers", None):
                ra = resp.headers.get("Retry-After")
                if ra is not None:
                    try:
                        retry_after = int(ra)
                    except ValueError:
                        pass
            if retry_after == 120:
                match = re.search(r"[Rr]etry after (\d+) seconds?", str(e))
                if match:
                    retry_after = int(match.group(1))
            if attempt < max_attempts:
                sys.stderr.write(f"[hf-api] 429 Too Many Requests，等待 {retry_after} 秒后重试（{attempt}/{max_attempts}）…\n")
                time.sleep(retry_after)
                continue
        raise
PY
fi

echo "All VBench cache dependencies processed."

