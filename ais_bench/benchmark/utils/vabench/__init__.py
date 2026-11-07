"""Shared VABench protocol constants; no framework or model imports."""

EXPERT_DIMENSIONS = (
    'first_dnsmos', 'first_nisqa', 'first_audiobox', 'second_viclip',
    'second_clap', 'second_desync', 'second_lsa', 'second_imagebind',
)
MLLM_DIMENSIONS = (
    'third_alignment', 'third_audio_reality', 'third_visual_reality',
    'third_expressiveness', 'third_artistry', 'fourth_qa_audio',
    'fourth_qa_vision',
)
DIMENSIONS = EXPERT_DIMENSIONS + MLLM_DIMENSIONS
