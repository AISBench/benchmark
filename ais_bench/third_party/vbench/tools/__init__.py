"""VBench 辅助工具。"""
from .video_generator import (
    DIMENSION_TO_PROMPT_SUITE,
    TEMPORAL_FLICKERING_VIDEOS_PER_PROMPT,
    run_vbench_generation,
)

__all__ = [
    'DIMENSION_TO_PROMPT_SUITE',
    'TEMPORAL_FLICKERING_VIDEOS_PER_PROMPT',
    'run_vbench_generation',
]
