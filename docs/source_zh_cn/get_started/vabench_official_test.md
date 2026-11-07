# VABench 官方评测完整指南

> 适用场景：只有 MP4 视频、官方 VABench 源码和已安装的 Python 依赖，不依赖 AISBench，也不使用 AISBench 的 prepared 目录。
>
> 目标：从一堆 MP4 开始，最终运行官方 `evaluate.py`，得到 `*_eval_results.json/csv`。
>
> 当前环境示例路径：
>
> - 官方源码：`/workspace/code/VABench-main`
> - 模型缓存：`/workspace/vabench-cache`
> - 视频目录：`/workspace/video`
> - 官方 data_dir：`/workspace/vabench_official_data`
> - 运行目录：`/workspace/vabench_official_run`
> - 结果目录：`/workspace/vabench_result`

---

## 0. 结论先行

官方评测不是直接对 `/workspace/video` 运行，而是要先准备一个符合官方结构的 `data_dir`：

```text
data_dir/
├── video/<category>/<idx>.mp4
├── audio/<category>/<idx>.wav
└── json/<category>.json
```

然后在一个包含 `mapping_text/` 的目录中运行：

```bash
cd /workspace/vabench_official_run

export VABENCH_CACHE_DIR=/workspace/vabench-cache
export CUDA_VISIBLE_DEVICES=6,7

python3 /workspace/code/VABench-main/evaluate.py \
  --data_dir /workspace/vabench_official_data \
  --output_dir /workspace/vabench_result
```

最关键的四个约束：

1. MP4 文件名去掉 `.mp4` 后必须等于官方 `idx`；
2. 每个 MP4 必须有音轨，WAV 从 MP4 抽取；
3. `json/<category>.json` 只能保留实际有视频的条目；
4. 运行目录必须有 `mapping_text/final_idx_to_prompt.csv`。

---

## 1. 需要哪些组件

你执行的四条命令：

```bash
pip install nisqa==2.0.post2
pip install -r requirement.txt
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/facebookresearch/ImageBind.git
```

只完成了 **Python 包依赖**。完整评测还需要：

| 组件 | 作用 | 常见缺失表现 |
| --- | --- | --- |
| 系统 `ffmpeg` / `ffprobe` | 抽音频、部分视频预处理 | `ffmpeg: command not found` |
| `libsndfile1` | soundfile / librosa 读取音频 | `OSError: sndfile library not found` |
| 模型权重缓存 | Qwen、CLAP、ViCLIP、ImageBind 等 | `FileNotFoundError: ...pth/onnx/tar` |
| LSA 辅助权重 | 人脸检测 `sfd_face.pth`、`buffalo_l` | LSA 初始化失败或无人脸 |
| 官方 `third_party` | DNSMOS、Synchformer、ViCLIP、SyncNet；前两项需手动补齐 | import 失败 |
| 官方 `data_dir` | 官方 `evaluate.py` 的输入格式 | 找不到视频/音频/JSON |
| `mapping_text/` | 官方代码读取 prompt 映射 | `加载映射表失败` |
| 正确的 GPU/配置 | 15 项指标推理 | CUDA OOM 或设备错误 |

---

## 2. 检查 Python 依赖

先确认实际安装的 torch 和关键包是否可用：

```bash
python3 - <<'PY'
import torch

print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('gpu count:', torch.cuda.device_count())

for name in [
    'decord', 'nisqa', 'audiobox_aesthetics',
    'onnxruntime', 'insightface', 'transformers',
    'qwen_omni_utils', 'imagebind',
]:
    try:
        __import__(name)
        print('OK  ', name)
    except Exception as exc:
        print('FAIL', name, repr(exc))
PY
```

期望类似：

```text
torch: 2.6.0
torch cuda: 12.4
cuda available: True
gpu count: 2
```

注意事项：

- 不要把 `laion_clap` 加进这个必检列表。官方 15 项评分使用的是 `transformers` 的 `ClapModel`，`vabench/second_clap.py` 并不 import `laion_clap`；`laion_clap` 只是 `requirement.txt` 里的依赖。它一 import 就会联网加载 `bert-base-uncased`、`roberta-base`、`facebook/bart-base` 三个 tokenizer，缓存为空且外网受限时会出现 SSL 重试，但这不影响官方评分。
- `nisqa==2.0.post2` 的元数据可能仍然写着 `torch==2.2.1`，而 `requirement.txt` 安装的是 `torch==2.6.0`。
- `pip check` 可能报告 NISQA 与 torch 版本不一致，这是官方安装顺序本身就会留下的元数据冲突。
- 不要为了让 `pip check` 全绿而把 torch 降回 2.2.1；以实际 `torch.__version__`、CUDA 可用性和真实推理结果为准。
- 如果 `torch.cuda.is_available()` 为 `False`，先解决驱动或 PyTorch CUDA wheel 问题，不要继续评测。

---

## 3. 安装系统工具

官方评分依赖系统 `ffmpeg`，不是 Python 包里的 `imageio-ffmpeg`：

```bash
apt-get update
apt-get install -y ffmpeg libsndfile1
```

检查：

```bash
ffmpeg -version | head -1
ffprobe -version | head -1
ldconfig -p | grep sndfile | head
```

---

## 4. 准备模型权重缓存

### 4.1 设置缓存目录

```bash
export VABENCH_CACHE_DIR=/workspace/vabench-cache
```

必须在运行 `evaluate.py` 之前设置。官方源码中多个模块在 import 时就会读取这个变量，部分路径在 import 阶段已经确定。

### 4.2 检查必需文件

```bash
CACHE=/workspace/vabench-cache

for p in \
  "$CACHE/Qwen2.5_omni_7B" \
  "$CACHE/dns_mos/sig_bak_ovr.onnx" \
  "$CACHE/dns_mos/model_v8.onnx" \
  "$CACHE/nisqa/nisqa_mos_only.tar" \
  "$CACHE/audiobox_aesthetics/checkpoint.pt" \
  "$CACHE/clap/laion__clap-htsat-unfused" \
  "$CACHE/viclip/ViCLIP-L_InternVid-FLT-10M.pth" \
  "$CACHE/imagebind/imagebind_huge.pth" \
  "$CACHE/syncnet2/syncnet_v2.model" \
  "$CACHE/synchformer/configs/cfg-24-01-04T16-39-21.yaml" \
  "$CACHE/synchformer/checkpoints/24-01-04T16-39-21.pt" \
  "$CACHE/checkpoints/auxiliary/sfd_face.pth" \
  "$CACHE/checkpoints/auxiliary/buffalo_l"; do
    if [ -e "$p" ]; then
        echo "OK   $p"
    else
        echo "MISS $p"
    fi
done
```

### 4.3 下载官方模型缓存

官方缓存：

```text
https://huggingface.co/datasets/tanABBCC/VABENCH_CACHE_DIR/tree/main
```

下载示例：

```bash
huggingface-cli download tanABBCC/VABENCH_CACHE_DIR \
  --repo-type dataset \
  --local-dir /workspace/vabench-cache
```

缓存很大，完整环境约几十 GB。下载后确认 `Qwen2.5_omni_7B/` 直接位于 `VABENCH_CACHE_DIR` 根目录下，不要多套一层目录。

### 4.4 LSA 辅助权重

LSA 需要两个资源，必须放在 `$VABENCH_CACHE_DIR` 下：

```text
$VABENCH_CACHE_DIR/checkpoints/auxiliary/
├── sfd_face.pth
└── models/
    └── buffalo_l/
        ├── det_10g.onnx
        └── 2d106det.onnx
```

`models/buffalo_l` 也可以是软链，指向 `../buffalo_l`。官方 `FaceDetector` 把 `$VABENCH_CACHE_DIR/checkpoints/auxiliary` 作为 InsightFace 的 `root`，InsightFace 固定查找其下的 `models/buffalo_l`。

#### sfd_face.pth

- 官方地址：
  `https://huggingface.co/ByteDance/LatentSync-1.5/resolve/main/auxiliary/sfd_face.pth`
- Gitee 镜像：
  `https://hf-api.gitee.com/ByteDance/LatentSync-1.5/resolve/main/auxiliary/sfd_face.pth`
- 文件大小：`89,844,381` 字节
- SHA256：`d54a87c2b7543b64729c9a25eafd188da15fd3f6e02f0ecec76ae1b30d86c491`

#### buffalo_l

- 直接下载：
  `https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip`
- Release 页面：
  `https://github.com/deepinsight/insightface/releases/tag/v0.7`
- 说明页：
  `https://github.com/deepinsight/insightface/tree/master/python-package#model-zoo`
- 文件大小：`288,621,354` 字节
- SHA256：`80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f`

注意：`python-package#model-zoo` 是 InsightFace README 的说明段落，不是文件下载链接。`buffalo_l` 的实际文件在 `v0.7` 的 release asset 中，必须使用上面的 `releases/download/v0.7/buffalo_l.zip`。`insightface` 的自动下载代码也使用同一个地址：

```python
BASE_REPO_URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7'
model_url = f"{BASE_REPO_URL}/{name}.zip"
```

#### 下载和放置

```bash
CACHE=/workspace/vabench-cache
AUX="$CACHE/checkpoints/auxiliary"

mkdir -p "$AUX/models"

# sfd_face.pth
wget -O "$AUX/sfd_face.pth" \
  https://huggingface.co/ByteDance/LatentSync-1.5/resolve/main/auxiliary/sfd_face.pth

# buffalo_l
wget -O /tmp/buffalo_l.zip \
  https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
mkdir -p "$AUX/models/buffalo_l"
unzip -q -o /tmp/buffalo_l.zip -d "$AUX/models/buffalo_l"
```

`buffalo_l.zip` 里的 ONNX 文件直接位于压缩包根目录，所以解压到 `models/buffalo_l/` 后应得到：

```text
$AUX/models/buffalo_l/det_10g.onnx
$AUX/models/buffalo_l/2d106det.onnx
$AUX/models/buffalo_l/1k3d68.onnx
$AUX/models/buffalo_l/genderage.onnx
$AUX/models/buffalo_l/w600k_r50.onnx
```

LSA 当前只使用 `detection` 和 `landmark_2d_106`，最少需要 `det_10g.onnx` 和 `2d106det.onnx`，其余文件保留即可。

如果已有 `$AUX/buffalo_l/` 但没有 `$AUX/models/buffalo_l/`，补软链：

```bash
mkdir -p "$AUX/models"
ln -sfn ../buffalo_l "$AUX/models/buffalo_l"
```

检查：

```bash
for f in \
  "$AUX/sfd_face.pth" \
  "$AUX/models/buffalo_l/det_10g.onnx" \
  "$AUX/models/buffalo_l/2d106det.onnx"; do
    test -f "$f" && echo "OK   $f" || echo "MISS $f"
done
```

缺少这些文件时，`second_lsa` 会初始化失败或无法进行人脸检测。

### 4.5 HuggingFace tokenizer 缓存（只有 `laion_clap` 需要）

`VABENCH_CACHE_DIR` 只服务 VABench 官方权重。`transformers` 的 tokenizer 使用另一套 HuggingFace 缓存，默认位于 `~/.cache/huggingface`（可用 `HF_HOME` 修改）。仓库里的 `laion_clap` 在 import 阶段就会执行：

```python
BertTokenizer.from_pretrained("bert-base-uncased")
RobertaTokenizer.from_pretrained("roberta-base")
BartTokenizer.from_pretrained("facebook/bart-base")
```

缓存为空时会联网下载。如果代理阻断了 `huggingface.co`，就会看到：

```text
SSLError(... 'SSL: UNEXPECTED_EOF_WHILE_READING' ...)
Retrying in 1s [Retry 1/5].
```

重试耗尽后 `import laion_clap` 失败。但官方 15 项评分不使用 `laion_clap`，所以这不会阻止 `evaluate.py`。

如果预检确实要求 `import laion_clap` 成功，可在一台能访问 HuggingFace 的机器上准备缓存：

```bash
export HF_HOME=/root/.cache/huggingface

huggingface-cli download bert-base-uncased \
  --include "config.json" "tokenizer*" "vocab*" "special_tokens_map.json"
huggingface-cli download roberta-base \
  --include "config.json" "tokenizer*" "vocab*" "merges.txt" "special_tokens_map.json"
huggingface-cli download facebook/bart-base \
  --include "config.json" "tokenizer*" "vocab*" "merges.txt" "special_tokens_map.json"
```

然后把整个缓存目录复制到目标机同一位置：

```text
$HF_HOME/hub/models--bert-base-uncased/
$HF_HOME/hub/models--roberta-base/
$HF_HOME/hub/models--facebook--bart-base/
```

确认存在后再启用离线模式：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

不要在缓存为空时设置这两个变量，否则会直接报“找不到本地文件”，而不是去下载。

---

## 5. 确认官方源码资源

官方源码不仅需要 `evaluate.py`，还需要以下目录：

```bash
SRC=/workspace/code/VABench-main

for p in \
  "$SRC/mapping/final_idx_to_prompt.csv" \
  "$SRC/json_files/json_text" \
  "$SRC/third_party/DNS-Challenge/DNSMOS/dnsmos_local.py" \
  "$SRC/third_party/Synchformer" \
  "$SRC/third_party/ViCLIP" \
  "$SRC/third_party/syncnet_eval"; do
    if [ -e "$p" ]; then
        echo "OK   $p"
    else
        echo "MISS $p"
    fi
done
```

固定提交的官方仓库**不包含** `third_party/Synchformer` 和 `third_party/DNS-Challenge`，重新 clone 也无法补齐，必须手动处理。`Synchformer` 直接软链模型缓存中的完整源码：

```bash
SRC=/workspace/code/VABench-main
CACHE=/workspace/vabench-cache

mkdir -p "$SRC/third_party"
ln -sfn "$CACHE/synchformer/Synchformer" "$SRC/third_party/Synchformer"
```

`DNS-Challenge` 只缺 `dnsmos_local.py`；缓存中的 `dns_mos/` 是 ONNX 权重，不是源码，不要复制进 `third_party`：

```bash
mkdir -p "$SRC/third_party/DNS-Challenge/DNSMOS"
curl -fL --retry 3 \
  -o "$SRC/third_party/DNS-Challenge/DNSMOS/dnsmos_local.py" \
  https://raw.githubusercontent.com/microsoft/DNS-Challenge/591184a9fcb2cbdec02520fed81a32bbbf9d73ff/DNSMOS/dnsmos_local.py
```

不需要 clone 整个 DNS-Challenge。访问 GitHub 需要代理或自签 CA 时，先按实际环境设置代理，并给 `curl` 加 `--cacert` 或设置 `CURL_CA_BUNDLE`。

补齐后重新执行上面的检查循环。缺少 `dnsmos_local.py` 时 `first_dnsmos` 会报 `DNSMOS library not found`；缺少 `Synchformer` 时 `second_desync` 的延迟导入会失败。

---

## 6. 视频命名规则

官方 `evaluate.py` 通过 `mapping/final_idx_to_prompt.csv` 中的 `idx` 查找视频，因此：

> **MP4 文件名去掉 `.mp4` 后，必须等于官方 `idx`。**

例如官方映射中有：

```text
idx: kitten curled up meowing
```

视频必须是：

```text
kitten curled up meowing.mp4
```

不能是：

```text
小猫.mp4
output_001.mp4
kitten.mp4
```

查看某个提示词对应的 `idx`：

```bash
grep -n "你的提示词片段" \
  /workspace/code/VABench-main/mapping/final_idx_to_prompt.csv
```

第一列就是 `idx`。

### 当前 7 个视频的对应关系

| category | 视频 |
| --- | --- |
| `easy_bioacoustics` | angry goose、angry turkey、bats、bee、kitten |
| `easy_human_language` | news anchor |
| `easy_Indoor_Environment_and_Urban_Atmosphere` | busy caf |

---

## 7. 准备官方 data_dir

### 7.1 脚本方式（推荐）

已有AISBench源码时，使用其中的独立脚本（无需安装AISBench）；没有该源码时使用第7.2节手动步骤：

```bash
python3 /workspace/code/benchmark/ais_bench/tools/vabench/official_prepare.py \
  --source-root /workspace/code/VABench-main \
  --videos /workspace/video \
  --output /workspace/vabench_official_data
```

脚本会：

1. 读取官方 `mapping/final_idx_to_prompt.csv`；
2. 读取官方 `json_files/json_text/*.json`；
3. 用 MP4 文件名匹配官方 `idx`；
4. 复制 MP4 到 `video/<category>/`；
5. 用 ffmpeg 抽取第一条音轨到 `audio/<category>/`；
6. 只保留实际有视频的 JSON 条目，写入 `json/<category>.json`。

输出示例：

```text
easy_bioacoustics: 5 sample(s)
easy_Indoor_Environment_and_Urban_Atmosphere: 1 sample(s)
easy_human_language: 1 sample(s)
```

检查：

```bash
find /workspace/vabench_official_data -maxdepth 3 -type f | sort
```

应该看到：

```text
video/<category>/<idx>.mp4
audio/<category>/<idx>.wav
json/<category>.json
```

脚本要求输出目录是新的或空的。如果要重新生成，换一个新路径，不要覆盖旧结果。

### 7.2 完全手工方式

如果不使用脚本，手工步骤如下。

#### 7.2.1 建目录

```bash
SRC=/workspace/code/VABench-main
VIDEOS=/workspace/video
DATA=/workspace/manual_vabench_data
RUN=/workspace/manual_vabench_run

mkdir -p "$DATA/video" "$DATA/audio" "$DATA/json" "$RUN"
```

#### 7.2.2 复制 MP4

```bash
for f in "$VIDEOS"/*.mp4; do
  idx=$(basename "$f" .mp4)

  case "$idx" in
    "angry goose honking loudly"|"angry turkey gobbling loudly"|"bats echolocate in dark caves"|"bee buzzing near flower stamens"|"kitten curled up meowing")
      category=easy_bioacoustics
      ;;
    "news anchor delivers evening news")
      category=easy_human_language
      ;;
    "busy caf with chatter and coffee")
      category=easy_Indoor_Environment_and_Urban_Atmosphere
      ;;
    *)
      echo "不是本次官方 idx: $idx"
      exit 1
      ;;
  esac

  mkdir -p "$DATA/video/$category" "$DATA/audio/$category"
  cp "$f" "$DATA/video/$category/"
done
```

检查：

```bash
find "$DATA/video" -name '*.mp4' | wc -l   # 应该是 7
```

#### 7.2.3 抽取 WAV

```bash
for video in "$DATA"/video/*/*.mp4; do
  category=$(basename "$(dirname "$video")")
  idx=$(basename "$video" .mp4)

  ffmpeg -nostdin -v error -y \
    -i "$video" \
    -map 0:a:0 -vn -c:a pcm_s16le \
    "$DATA/audio/$category/$idx.wav"
done
```

检查：

```bash
find "$DATA/audio" -name '*.wav' | wc -l   # 应该是 7
```

如果 ffmpeg 报错，通常是 MP4 没有音轨。

#### 7.2.4 生成 `json/<category>.json`

不能直接把官方 `json_files/json_text/<category>.json` 整个复制过去，因为官方 category JSON 包含很多本条视频之外的条目：

```text
easy_bioacoustics.json                         39 条
easy_human_language.json                       64 条
easy_Indoor_Environment_and_Urban_Atmosphere.json  38 条
```

直接复制会让官方评测查找大量不存在的视频，产生警告和空分数。需要只保留实际有视频的条目。

可以用一段一次性 Python 命令完成，不依赖 AISBench：

```bash
python3 - "$SRC" "$DATA" <<'PY'
import csv
import json
import sys
from pathlib import Path

src = Path(sys.argv[1]).resolve()
data = Path(sys.argv[2]).resolve()

# idx -> prompt
mapping = {}
with (src / 'mapping' / 'final_idx_to_prompt.csv').open(
        encoding='utf-8', newline='') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        mapping[row['idx']] = json.loads('"' + row['prompt'] + '"')

# prompt -> (category, official item)
catalog = {}
for path in sorted((src / 'json_files' / 'json_text').glob('*.json')):
    for item in json.loads(path.read_text(encoding='utf-8')):
        catalog[item['prompt_en']] = (path.stem, item)

grouped = {}
for video in sorted((data / 'video').glob('*/*.mp4')):
    idx = video.stem
    if idx not in mapping:
        raise SystemExit(f'不是官方 idx: {idx}')
    prompt = mapping[idx]
    if prompt not in catalog:
        raise SystemExit(f'官方 JSON 中没有对应提示词: {idx}')

    category, item = catalog[prompt]
    audio = data / 'audio' / category / f'{idx}.wav'
    grouped.setdefault(category, []).append(
        dict(item, video_list=[str(video)], audio_list=[str(audio)])
    )

for category, items in grouped.items():
    path = data / 'json' / f'{category}.json'
    path.write_text(
        json.dumps(items, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    print(category, len(items))
PY
```

输出应为：

```text
easy_bioacoustics 5
easy_Indoor_Environment_and_Urban_Atmosphere 1
easy_human_language 1
```

如果完全不想写任何代码，只能手工打开三个官方 JSON，删除不需要的条目再保存。7 个视频对应 5/1/1 条，做得到，但很容易删错，不推荐。

---

## 8. 准备运行目录和 `mapping_text`

官方代码读取的是相对路径：

```python
"./mapping_text/final_idx_to_prompt.csv"
```

但官方源码里的目录叫 `mapping/`。因此**启动 `evaluate.py` 时所在的目录**中必须有 `mapping_text`。

不要求固定使用 `/workspace/vabench_official_run`，任意新目录都可以，只要在其中放软链并从该目录启动：

```bash
# 可改成任意目录，不要求已存在
RUN=/workspace/vabench_official_run

mkdir -p "$RUN"
ln -sfn /workspace/code/VABench-main/mapping "$RUN/mapping_text"
cd "$RUN"

test -f mapping_text/final_idx_to_prompt.csv && echo mapping-ok
```

关键点：

- 软链名必须叫 `mapping_text`，不能直接叫 `mapping`；
- 建议使用绝对路径创建软链，避免移动运行目录后失效；
- 软链只是省去复制；直接复制 `mapping/` 为 `mapping_text/` 也可以；
- 每次换新的运行目录，都要重新创建这个软链。

不要在 `/workspace/code/VABench-main` 下直接运行；那里没有 `mapping_text`，会打印：

```text
加载映射表失败
```

然后所有视频都匹配不到。

---

## 9. 运行官方评测

### 9.1 设置环境变量

```bash
export VABENCH_CACHE_DIR=/workspace/vabench-cache
export CUDA_VISIBLE_DEVICES=6,7
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

说明：

- `CUDA_VISIBLE_DEVICES` 按实际机器修改；物理卡 6/7 在进程内会变成逻辑 0/1。
- `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1` 只在模型缓存完整时使用；如果还需要联网下载，先不要设置。
- `VABENCH_CACHE_DIR` 必须在启动 Python 前设置。

### 9.2 GPU 配置

官方默认配置：

```text
/workspace/code/VABench-main/vabench/config.yaml
```

内容包含：

```yaml
params:
  max_memory: {0: "80GB", 1: "80GB"}
  device_map: "auto"
```

默认假设两张 80GB GPU。如果实际显存不同：

```bash
cp /workspace/code/VABench-main/vabench/config.yaml \
   /workspace/vabench_config.yaml
```

编辑 `/workspace/vabench_config.yaml` 中的 `max_memory`，然后运行时加 `--config`。

### 9.3 运行命令

必须在第 8 节创建的运行目录中执行。下面以 `/workspace/vabench_official_run` 为例；如果你用了其他目录，替换 `cd` 的路径：

```bash
cd /workspace/vabench_official_run

python3 /workspace/code/VABench-main/evaluate.py \
  --data_dir /workspace/vabench_official_data \
  --output_dir /workspace/vabench_result
```

如果使用自定义配置：

```bash
python3 /workspace/code/VABench-main/evaluate.py \
  --data_dir /workspace/vabench_official_data \
  --output_dir /workspace/vabench_result \
  --config /workspace/vabench_config.yaml
```

完整 15 项评分耗时较长，建议在 `tmux` 中运行，或使用 `nohup`：

```bash
mkdir -p /workspace/vabench_result

nohup python3 /workspace/code/VABench-main/evaluate.py \
  --data_dir /workspace/vabench_official_data \
  --output_dir /workspace/vabench_result \
  > /workspace/vabench_result/run.log 2>&1 &

tail -f /workspace/vabench_result/run.log
```

---

## 10. 检查结果

### 10.1 结果文件

```bash
find /workspace/vabench_result \
  \( -name '*_eval_results.json' -o -name '*_eval_results.csv' \)
```

结构：

```text
/workspace/vabench_result/<category>/
  <category>_<时间>_eval_results.json
  <category>_<时间>_eval_results.csv
  <category>_<时间>_full_info.json
```

### 10.2 检查指标数量

```bash
python3 - <<'PY'
import glob
import json

for path in sorted(glob.glob('/workspace/vabench_result/*/*_eval_results.json')):
    data = json.load(open(path))
    print(path, len(data), sorted(data))
PY
```

正常情况下每个 category 有 15 个指标键：

```text
first_dnsmos
first_nisqa
first_audiobox
second_viclip
second_clap
second_desync
second_lsa
second_imagebind
third_alignment
third_audio_reality
third_visual_reality
third_expressiveness
third_artistry
fourth_qa_audio
fourth_qa_vision
```

### 10.3 关于总分

官方 `evaluate.py` 输出的是每个 category 的 15 项原始分数和按 prompt 聚合的 CSV，不计算 0–100 展示分，也不计算单个总分。

如果需要：

- 官方对齐：将原生输出与 AISBench 的 `official_eval_results/` 比较同一有效样本集合的逐项原始分及均分，并核对失败和省略；不拿项目总分代替逐指标对齐；
- 查看项目总分：使用 AISBench 生成的 `report.json`、`total.csv`。当前v2规则在请求全部15项时将缺失指标按0计入预览总分，原始均分仍留空；这是AISBench展示协议，不是官方总分；
- 自己汇总：按 AISBench 文档中的固定归一化公式计算，不要直接把 15 项原始分相加。

---

## 11. 常见错误

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `加载映射表失败` | 当前目录没有 `mapping_text/final_idx_to_prompt.csv` | 在运行目录建 `mapping_text` 软链 |
| `WARNING!!! This required video is not found` | MP4 文件名不是官方 `idx`，或 JSON 保留了大量无视频条目 | 检查文件名；只保留实际视频对应的 JSON 条目 |
| `required audio is not found` | WAV 文件名与 MP4 不一致，或没有抽音频 | 用 ffmpeg 按 `<idx>.wav` 抽取 |
| `FileNotFoundError: ...onnx/pth/tar` | 模型缓存缺失或 `VABENCH_CACHE_DIR` 未设置 | 检查缓存路径和文件 |
| `CUDA out of memory` | GPU 显存不足，config 仍按 2×80GB 配置 | 修改 `max_memory` 或换 GPU |
| `ffmpeg: command not found` | 没装系统 ffmpeg | `apt-get install -y ffmpeg` |
| `OSError: sndfile library not found` | 缺 libsndfile | `apt-get install -y libsndfile1` |
| `MP4 must contain audio` / ffmpeg 失败 | MP4 没有音轨 | 使用带音轨的视频；不要自动补静音来伪造评测 |
| `second_clap` 多次结果不同 | 官方随机 fusion 截断 | 不要按逐样本完全相等验收，多次运行取均值 |
| LSA 无人脸或分数缺失 | 视频中没有人脸，或人脸太小 | 这是输入限制，不是调度失败 |
| `pip check` 报 NISQA/torch 不一致 | NISQA 元数据固定 torch 2.2.1，实际为 2.6.0 | 确认实际 torch 和 GPU 可用，不要盲目降级 |
| `import laion_clap` 打印 `SSLError ... huggingface.co` 和 `Retrying` | `laion_clap` 导入时强制加载 bert/roberta/bart tokenizer，HF 缓存为空且外网被阻断 | 官方 15 项不使用 `laion_clap`，可将其从预检移除；必须导入时按 4.5 准备 HF 缓存 |

---

## 12. 与 AISBench 结果的关系

AISBench 的 VABench 适配层会在 `report.json` 之外生成：

```text
official_eval_results/<category>/
  <category>_<时间>_eval_results.json
  <category>_<时间>_eval_results.csv
  <category>_<时间>_full_info.json
```

这个目录与官方 `evaluate.py` 的输出结构一致，可用于逐条对比原始分数。

AISBench 提供 `compare-official` 命令，直接按类别、指标和媒体文件名对齐两边结果，只比较核心分 `video_results`，忽略不同的绝对媒体路径：

```bash
python3 -m ais_bench.tools.vabench.cli compare-official \
  --aisbench /workspace/code/benchmark/outputs/.../vabench_t2av_artifacts \
  --official /workspace/vabench_result_20261105_071201 \
  --output /workspace/vabench_compare_official.json
```

- `--aisbench` 可以填 `vabench_t2av_artifacts`，也可以直接填其中的 `official_eval_results`；
- `--official` 填原生 `evaluate.py --output_dir`；
- 默认相对误差严格 `<1%`；官方分接近 0 时使用绝对误差 `1e-6`；
- 有差异时逐条打印类别、指标、样本、两边数值和相对/绝对误差；
- 退出码非 0 表示存在差异；`--output` 保存完整 JSON。

注意：`second_clap` 使用官方随机截断，同一输入重复运行也可能出现差异；这类差异会被打印并标注，不应直接判为调用错误。

两者区别：

- 官方 `evaluate.py` 只输出原始指标，不计算归一化总分；
- AISBench 额外计算 15 项 0–100 展示分和等权总分；
- AISBench 的 `second_clap` 仍然调用官方随机逻辑，因此重复运行也可能有波动。

---

## 13. 当前环境已验证内容

已为 `/workspace/video` 的 7 个视频生成：

```text
/workspace/vabench_official_data
├── video/
│   ├── easy_bioacoustics/                 5 个 MP4
│   ├── easy_human_language/               1 个 MP4
│   └── easy_Indoor_Environment_and_Urban_Atmosphere/  1 个 MP4
├── audio/
│   ├── easy_bioacoustics/                 5 个 WAV
│   ├── easy_human_language/               1 个 WAV
│   └── easy_Indoor_Environment_and_Urban_Atmosphere/  1 个 WAV
└── json/
    ├── easy_bioacoustics.json             5 条
    ├── easy_human_language.json           1 条
    └── easy_Indoor_Environment_and_Urban_Atmosphere.json  1 条
```

已使用官方 `VABench.build_full_info_json` 校验：

```text
easy_bioacoustics: 5/5 视频和音频匹配
easy_human_language: 1/1 视频和音频匹配
easy_Indoor_Environment_and_Urban_Atmosphere: 1/1 视频和音频匹配
```

该验证只检查官方 loader 能否找到并匹配视频/音频/JSON，**没有重新跑完整的 15 项模型评分**。完整评分需要按第 9 节命令单独执行。

---

## 14. 最小操作清单

如果只想要一份可执行的顺序：

```bash
# 1. 环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vabench311

export VABENCH_CACHE_DIR=/workspace/vabench-cache
export CUDA_VISIBLE_DEVICES=6,7

# 2. 准备 data_dir
python3 /workspace/code/benchmark/ais_bench/tools/vabench/official_prepare.py \
  --source-root /workspace/code/VABench-main \
  --videos /workspace/video \
  --output /workspace/vabench_official_data

# 3. 准备运行目录
mkdir -p /workspace/vabench_official_run
cd /workspace/vabench_official_run
ln -sfn /workspace/code/VABench-main/mapping mapping_text

# 4. 运行
python3 /workspace/code/VABench-main/evaluate.py \
  --data_dir /workspace/vabench_official_data \
  --output_dir /workspace/vabench_result
```

如果这四步中任何一步报错，优先按第 11 节排查，不要继续往下跑。
