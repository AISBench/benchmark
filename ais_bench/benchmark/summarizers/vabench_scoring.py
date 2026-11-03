"""Fixed AISBench display protocol; does not change official model scores."""
import math

from ais_bench.benchmark.utils.vabench import DIMENSIONS


VERSION = 'aisbench-vabench-display-v1'
RULES = {
    'first_dnsmos': (1, 5, False),
    'first_nisqa': (1, 5, False),
    'first_audiobox': (-1.75, 7.25, False),
    'second_viclip': (-1, 1, False),
    'second_clap': (-1, 1, False),
    'second_desync': (0, 2, True),
    'second_lsa': (0, 10, False),
    'second_imagebind': (-1, 1, False),
    'third_alignment': (1, 5, False),
    'third_audio_reality': (1, 5, False),
    'third_visual_reality': (1, 5, False),
    'third_expressiveness': (1, 5, False),
    'third_artistry': (1, 5, False),
    'fourth_qa_audio': (0, 1, False),
    'fourth_qa_vision': (0, 1, False),
}


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def normalize_score(dimension, value):
    """Normalize the raw MEAN, then clip. Never average clipped sample scores."""
    low, high, reverse = RULES[dimension]
    if not finite_number(value):
        return dict(normalized_score=None, mean_out_of_bounds=False)
    ratio = (value - low) / (high - low)
    if reverse:
        ratio = 1 - ratio
    return dict(normalized_score=max(0.0, min(100.0, 100 * ratio)),
                mean_out_of_bounds=value < low or value > high)


def protocol():
    return dict(version=VERSION, scale=[0, 100], clipping='after_raw_mean',
                sample_weighting='equal_valid_video', metric_weighting='equal_15_dimensions',
                official_total=False,
                anchors={d: dict(low=RULES[d][0], high=RULES[d][1],
                                 reverse=RULES[d][2]) for d in DIMENSIONS})


def aggregate(metrics):
    missing = [d for d in DIMENSIONS if not finite_number(metrics.get(d, {}).get('normalized_score'))]
    incomplete = [d for d in DIMENSIONS if metrics.get(d, {}).get('status') != 'complete']
    score = None if missing else sum(metrics[d]['normalized_score'] for d in DIMENSIONS) / len(DIMENSIONS)
    status = 'unavailable' if missing else 'preview' if incomplete else 'formal'
    return dict(score=score, status=status,
                label={'unavailable': '总分不可用', 'preview': '预览总分／不完整', 'formal': '正式总分'}[status],
                valid_dimensions=len(DIMENSIONS) - len(missing), expected_dimensions=len(DIMENSIONS),
                missing_dimensions=missing, incomplete_dimensions=incomplete,
                weights={d: 1 / len(DIMENSIONS) for d in DIMENSIONS},
                definition='AISBench fixed-normalization total, not an official VABench total')
