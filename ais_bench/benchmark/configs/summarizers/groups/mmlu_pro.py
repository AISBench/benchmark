categories = ['math', 'physics', 'chemistry', 'law', 'engineering', 'other', 'economics', 'health', 'psychology', 'business', 'biology', 'philosophy', 'computer science', 'history']

mmlu_pro_summary_groups = [
    {'name': 'mmlu_pro', 'subsets': ['mmlu_pro_' + c.replace(' ', '_') for c in categories]},
]

_mmlu_pro_all = ['mmlu_pro_' + c.replace(' ', '_') for c in categories]
_mmlu_pro_weights = {
    'mmlu_pro_math': 1351,
    'mmlu_pro_physics': 1299,
    'mmlu_pro_chemistry': 1132,
    'mmlu_pro_law': 1101,
    'mmlu_pro_engineering': 969,
    'mmlu_pro_other': 924,
    'mmlu_pro_economics': 844,
    'mmlu_pro_health': 818,
    'mmlu_pro_psychology': 798,
    'mmlu_pro_business': 789,
    'mmlu_pro_biology': 717,
    'mmlu_pro_philosophy': 499,
    'mmlu_pro_computer_science': 410,
    'mmlu_pro_history': 381,
}
mmlu_pro_summary_groups.append({'name': 'mmlu_pro-weighted', 'subsets': _mmlu_pro_all, 'weights': _mmlu_pro_weights})
