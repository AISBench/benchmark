subcategories = {
    'agronomy': ['other'],
    'anatomy': ['biology'],
    'ancient_chinese': ['linguistics','china specific'],
    'arts': ['arts'],
    'astronomy': ['physics'],
    'business_ethics': ['business'],
    'chinese_civil_service_exam': ['politics','china specific'],
    'chinese_driving_rule': ['other','china specific'],
    'chinese_food_culture': ['culture','china specific'],
    'chinese_foreign_policy': ['politics','china specific'],
    'chinese_history':['history','china specific'],
    'chinese_literature': ['literature','china specific'],
    'chinese_teacher_qualification': ['education','china specific'],
    'college_actuarial_science':['math'],
    'college_education':['education'],
    'college_engineering_hydrology': ['engineering'],
    'college_law': ['law'],
    'college_mathematics': ['math'],
    'college_medical_statistics':['statistics'],
    'clinical_knowledge': ['other'],
    'college_medicine': ['other'],
    'computer_science': ['computer science'],
    'computer_security': ['other'],
    'conceptual_physics': ['physics'],
    'construction_project_management': ['other','china specific'],
    'economics': ['economics'],
    'education': ['education'],
    'elementary_chinese':['linguistics','china specific'],
    'elementary_commonsense':['other','china specific'],
    'elementary_information_and_technology': ['other'],
    'electrical_engineering': ['engineering'],
    'elementary_mathematics': ['math'],
    'ethnology': ['culture','china specific'],
    'food_science': ['other'],
    'genetics': ['biology'],
    'global_facts': ['global'],
    'high_school_biology': ['biology'],
    'high_school_chemistry': ['chemistry'],
    'high_school_geography': ['geography'],
    'high_school_mathematics': ['math'],
    'high_school_physics': ['physics'],
    'high_school_politics': ['politics','china specific'],
    'human_sexuality': ['other'],
    'international_law': ['law'],
    'journalism': ['sociology'],
    'jurisprudence': ['law'],
    'legal_and_moral_basis': ['other'],
    'logical': ['philosophy'],
    'machine_learning': ['computer science'],
    'management': ['business'],
    'marketing': ['business'],
    'marxist_theory': ['philosophy'],
    'modern_chinese': ['linguistics','china specific'],
    'nutrition': ['other'],
    'philosophy': ['philosophy'],
    'professional_accounting': ['business'],
    'professional_law': ['law'],
    'professional_medicine': ['other'],
    'professional_psychology': ['psychology'],
    'public_relations': ['politics'],
    'security_study': ['politics'],
    'sociology': ['culture'],
    'sports_science': ['other'],
    'traditional_chinese_medicine': ['other','china specific'],
    'virology': ['biology'],
    'world_history':['history'],
    'world_religions': ['global'],
}

categories = {
    'STEM': ['physics', 'chemistry', 'biology', 'computer science', 'math', 'engineering', 'statistics'],
    'Humanities': ['history', 'philosophy', 'law', 'arts', 'literature', 'global'],
    'Social Science': ['linguistics','business', 'politics', 'culture', 'economics', 'geography', 'psychology', 'education', 'sociology'],
    'Other':['other'],
    'China specific': ['china specific'],
}

category2subject = {}
for k, v in categories.items():
    for subject, subcat in subcategories.items():
        for c in subcat:
            if c in v:
                category2subject.setdefault(k, []).append(subject)

cmmlu_summary_groups = []

_cmmlu_humanities = ['cmmlu-' + s for s in category2subject['Humanities']]
cmmlu_summary_groups.append({'name': 'cmmlu-humanities', 'subsets': _cmmlu_humanities})

_cmmlu_stem = ['cmmlu-' + s for s in category2subject['STEM']]
cmmlu_summary_groups.append({'name': 'cmmlu-stem', 'subsets': _cmmlu_stem})

_cmmlu_social_science = ['cmmlu-' + s for s in category2subject['Social Science']]
cmmlu_summary_groups.append({'name': 'cmmlu-social-science', 'subsets': _cmmlu_social_science})

_cmmlu_other = ['cmmlu-' + s for s in category2subject['Other']]
cmmlu_summary_groups.append({'name': 'cmmlu-other', 'subsets': _cmmlu_other})

_cmmlu_china_specific = ['cmmlu-' + s for s in category2subject['China specific']]
cmmlu_summary_groups.append({'name': 'cmmlu-china-specific', 'subsets': _cmmlu_china_specific})

_cmmlu_all = ['cmmlu-' + s for s in subcategories.keys()]
_cmmlu_weights = {'world_history': 161, 'elementary_chinese': 252, 'sociology': 226, 'machine_learning': 122, 'electrical_engineering': 172, 'international_law': 185, 'anatomy': 148, 'traditional_chinese_medicine': 185, 'high_school_biology': 169, 'college_medical_statistics': 106, 'chinese_teacher_qualification': 179, 'philosophy': 105, 'professional_psychology': 232, 'logical': 123, 'agronomy': 169, 'education': 163, 'virology': 169, 'public_relations': 174, 'genetics': 176, 'global_facts': 149, 'professional_medicine': 376, 'chinese_driving_rule': 131, 'nutrition': 145, 'professional_accounting': 175, 'economics': 159, 'astronomy': 165, 'chinese_foreign_policy': 107, 'high_school_chemistry': 132, 'high_school_politics': 143, 'human_sexuality': 126, 'college_mathematics': 105, 'chinese_food_culture': 136, 'computer_science': 204, 'construction_project_management': 139, 'conceptual_physics': 147, 'business_ethics': 209, 'journalism': 172, 'college_law': 108, 'chinese_civil_service_exam': 160, 'management': 210, 'modern_chinese': 116, 'chinese_history': 323, 'college_medicine': 273, 'high_school_physics': 110, 'college_actuarial_science': 106, 'high_school_mathematics': 164, 'elementary_information_and_technology': 238, 'legal_and_moral_basis': 214, 'jurisprudence': 411, 'clinical_knowledge': 237, 'computer_security': 171, 'marxist_theory': 189, 'elementary_mathematics': 230, 'security_study': 135, 'chinese_literature': 204, 'ethnology': 135, 'professional_law': 211, 'food_science': 143, 'elementary_commonsense': 198, 'college_engineering_hydrology': 106, 'marketing': 180, 'college_education': 107, 'arts': 160, 'world_religions': 160, 'ancient_chinese': 164, 'high_school_geography': 118, 'sports_science': 165}
_cmmlu_weights = {'cmmlu-' + k : v for k,v in _cmmlu_weights.items()}
cmmlu_summary_groups.append({'name': 'cmmlu', 'subsets': _cmmlu_all})
cmmlu_summary_groups.append({'name': 'cmmlu-weighted', 'subsets': _cmmlu_all, 'weights': _cmmlu_weights})
