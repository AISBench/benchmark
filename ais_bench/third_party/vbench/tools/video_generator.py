"""VBench 视频生成工具：用户只需实现 generate(prompt, index) -> video，工具自动保存为目标格式。

参考 VBench 官方 Prompt Suite 说明：
- prompts_per_dimension: 各维度 prompt
- all_dimension.txt: 全维度合并
- prompts_per_category: 8 类内容 (Animal, Architecture, Food, Human, Lifestyle, Plant, Scenery, Vehicles)
- all_category.txt: 全类别合并
- temporal_flickering 维度需采样 25 个视频/ prompt
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Union

import numpy as np

# 测评维度 -> Prompt Suite 文件映射（与 VBench_full_info.json 一致）
DIMENSION_TO_PROMPT_SUITE = {
    'subject_consistency': 'subject_consistency',
    'background_consistency': 'scene',
    'temporal_flickering': 'temporal_flickering',
    'motion_smoothness': 'subject_consistency',
    'dynamic_degree': 'subject_consistency',
    'aesthetic_quality': 'overall_consistency',
    'imaging_quality': 'overall_consistency',
    'object_class': 'object_class',
    'multiple_objects': 'multiple_objects',
    'human_action': 'human_action',
    'color': 'color',
    'spatial_relationship': 'spatial_relationship',
    'scene': 'scene',
    'temporal_style': 'temporal_style',
    'appearance_style': 'appearance_style',
    'overall_consistency': 'overall_consistency',
}

# temporal_flickering 需 25 个视频以确保 static filter 后覆盖充分
TEMPORAL_FLICKERING_VIDEOS_PER_PROMPT = 25

# Optional imports for video saving
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _sanitize_filename(prompt: str) -> str:
    """Replace filesystem-unsafe characters in prompt for use as filename."""
    unsafe = r'[/\\:*?"<>|]'
    return re.sub(unsafe, '_', prompt).strip() or 'prompt'


def _save_video(
    video: Union[np.ndarray, "torch.Tensor", str],
    output_path: str,
    fps: int = 8,
) -> None:
    """Save video to mp4. Accepts numpy (T,H,W,C), torch.Tensor, or existing path."""
    if isinstance(video, str):
        if os.path.isfile(video) and video != output_path:
            import shutil
            shutil.copy2(video, output_path)
        return

    # Convert tensor to numpy if needed
    if HAS_TORCH and hasattr(video, 'cpu'):
        arr = video.detach().cpu().numpy()
    else:
        arr = np.asarray(video)

    if arr.ndim == 4:
        # Expect (T, H, W, C) or (T, C, H, W)
        if arr.shape[1] == 3:  # (T, C, H, W) -> (T, H, W, C)
            arr = np.transpose(arr, (0, 2, 3, 1))
        # Normalize to 0-255 if needed
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Expected 4D array (T,H,W,C), got shape {arr.shape}")

    T, H, W, C = arr.shape
    frames = [arr[t] for t in range(T)]

    if HAS_CV2:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
        for frame in frames:
            if C == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            writer.write(frame_bgr)
        writer.release()
    elif HAS_IMAGEIO:
        try:
            imageio.mimsave(output_path, frames, fps=fps, codec='libx264')
        except Exception:
            imageio.mimsave(output_path, frames, fps=fps)
    else:
        raise ImportError(
            "Video saving requires OpenCV or imageio. Install with: "
            "pip install opencv-python  OR  pip install imageio imageio-ffmpeg"
        )


def _get_prompts_root() -> Path:
    return Path(__file__).resolve().parent.parent / "prompts"


def _load_prompts(mode: str, prompt_source, dimension: str | None, category: str | None) -> list[str]:
    """Load prompt list based on mode."""
    root = _get_prompts_root()
    if mode == "standard":
        if not dimension:
            raise ValueError("mode='standard' requires dimension to be set")
        suite = DIMENSION_TO_PROMPT_SUITE.get(dimension, dimension)
        prompt_file = root / "prompts_per_dimension" / f"{suite}.txt"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    elif mode == "all_dimension":
        prompt_file = root / "all_dimension.txt"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    elif mode == "all_category":
        prompt_file = root / "all_category.txt"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    elif mode == "category":
        if not category:
            raise ValueError("mode='category' requires category (e.g. animal, architecture, food, human, lifestyle, plant, scenery, vehicles)")
        prompt_file = root / "prompts_per_category" / f"{category.lower()}.txt"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    elif mode == "custom":
        if prompt_source is None:
            raise ValueError("mode='custom' requires prompt_source (list of prompts or path to txt file)")
        if isinstance(prompt_source, (list, tuple)):
            return list(prompt_source)
        path = Path(prompt_source)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(
            f"mode must be one of 'custom', 'standard', 'all_dimension', 'all_category', 'category', got {mode}"
        )


def run_vbench_generation(
    generate_fn: Callable[[str, int], Union[np.ndarray, "torch.Tensor", str]],
    output_dir: str,
    mode: str = "custom",
    prompt_source: str | list[str] | None = None,
    dimension: str | None = None,
    category: str | None = None,
    videos_per_prompt: int | None = None,
    fps: int = 8,
    sanitize_filename: bool = True,
    seed: int | None = None,
) -> str:
    """运行 VBench 视频生成，自动保存为目标格式。

    Args:
        generate_fn: 用户实现的生成函数 (prompt, index) -> video。
            video 可为: np.ndarray (T,H,W,C) uint8、torch.Tensor、或已保存的视频路径 str。
            建议在内部使用 index 或 seed+index 作为随机种子，确保每个视频多样性且可复现。
        output_dir: 输出目录，供后续 ais_bench eval 使用。
        mode: "custom" | "standard" | "all_dimension" | "all_category" | "category"。
        prompt_source: custom 模式下为 txt 文件路径或 prompt 列表；其他模式忽略。
        dimension: standard 模式下指定维度；会按官方映射选择 prompt suite（如 background_consistency -> scene）。
        category: category 模式下指定类别，如 animal, architecture, food, human, lifestyle, plant, scenery, vehicles。
        videos_per_prompt: 每个 prompt 的视频数。默认 5；temporal_flickering 维度自动为 25。
        fps: 输出视频帧率，默认 8。
        sanitize_filename: 是否对 prompt 做文件名安全处理，默认 True。
        seed: 可选，用于复现。建议在 generate_fn 内使用 seed+index 作为随机种子。

    Returns:
        output_dir，可直接用于 ais_bench --mode eval 的 path 配置。
    """
    prompts = _load_prompts(mode, prompt_source, dimension, category)
    os.makedirs(output_dir, exist_ok=True)

    if videos_per_prompt is None:
        videos_per_prompt = (
            TEMPORAL_FLICKERING_VIDEOS_PER_PROMPT
            if dimension == "temporal_flickering"
            else 5
        )

    total = len(prompts) * videos_per_prompt
    done = 0
    for prompt in prompts:
        base = _sanitize_filename(prompt) if sanitize_filename else prompt
        for i in range(videos_per_prompt):
            out_path = os.path.join(output_dir, f"{base}-{i}.mp4")
            if os.path.isfile(out_path):
                done += 1
                continue
            try:
                if seed is not None and HAS_TORCH:
                    torch.manual_seed(seed + i)
                video = generate_fn(prompt, i)
                _save_video(video, out_path, fps=fps)
            except Exception as e:
                raise RuntimeError(f"Failed to generate video for prompt '{prompt[:50]}...' index {i}: {e}") from e
            done += 1
            if done % 10 == 0 or done == total:
                print(f"VBench generator: {done}/{total} videos saved")

    print(f"VBench generation complete. Output: {output_dir}")
    return output_dir
