categories = ['math', 'physics', 'chemistry', 'law', 'engineering', 'other', 'economics', 'health', 'psychology', 'business', 'biology', 'philosophy', 'computer science', 'history']

_mmlu_pro_all = [('mmlu_pro_' + c.replace(' ', '_'), 'accuracy') for c in categories]

_mmlu_pro_weights = {
    'math': 1351,
    'physics': 1299,
    'chemistry': 1132,
    'law': 1101,
    'engineering': 969,
    'other': 924,
    'economics': 844,
    'health': 818,
    'psychology': 798,
    'business': 789,
    'biology': 717,
    'philosophy': 499,
    'computer science': 410,
    'history': 381,
}
_mmlu_pro_weights = {'mmlu_pro_' + k.replace(' ', '_'): v for k, v in _mmlu_pro_weights.items()}

mmlu_pro_summary_groups = [
    {'name': 'mmlu_pro', 'subsets': _mmlu_pro_all},
    {'name': 'mmlu_pro-weighted', 'subsets': _mmlu_pro_all, 'weights': _mmlu_pro_weights},
]