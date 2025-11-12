ceval_summary_groups = []

_ceval_stem = ['computer_network', 'operating_system', 'computer_architecture', 'college_programming', 'college_physics', 'college_chemistry', 'advanced_mathematics', 'probability_and_statistics', 'discrete_mathematics', 'electrical_engineer', 'metrology_engineer', 'high_school_mathematics', 'high_school_physics', 'high_school_chemistry', 'high_school_biology', 'middle_school_mathematics', 'middle_school_biology', 'middle_school_physics', 'middle_school_chemistry', 'veterinary_medicine']
_ceval_stem = ['ceval-' + s for s in _ceval_stem]
ceval_summary_groups.append({'name': 'ceval-stem', 'subsets': _ceval_stem})

_ceval_social_science = ['college_economics', 'business_administration', 'marxism', 'mao_zedong_thought', 'education_science', 'teacher_qualification', 'high_school_politics', 'high_school_geography', 'middle_school_politics', 'middle_school_geography']
_ceval_social_science = ['ceval-' + s for s in _ceval_social_science]
ceval_summary_groups.append({'name': 'ceval-social-science', 'subsets': _ceval_social_science})

_ceval_humanities = ['modern_chinese_history', 'ideological_and_moral_cultivation', 'logic', 'law', 'chinese_language_and_literature', 'art_studies', 'professional_tour_guide', 'legal_professional', 'high_school_chinese', 'high_school_history', 'middle_school_history']
_ceval_humanities = ['ceval-' + s for s in _ceval_humanities]
ceval_summary_groups.append({'name': 'ceval-humanities', 'subsets': _ceval_humanities})

_ceval_other = ['civil_servant', 'sports_science', 'plant_protection', 'basic_medicine', 'clinical_medicine', 'urban_and_rural_planner', 'accountant', 'fire_engineer', 'environmental_impact_assessment_engineer', 'tax_accountant', 'physician']
_ceval_other = ['ceval-' + s for s in _ceval_other]
ceval_summary_groups.append({'name': 'ceval-other', 'subsets': _ceval_other})

_ceval_hard = ['advanced_mathematics', 'discrete_mathematics', 'probability_and_statistics', 'college_chemistry', 'college_physics', 'high_school_mathematics', 'high_school_chemistry', 'high_school_physics']
_ceval_hard = ['ceval-' + s for s in _ceval_hard]
ceval_summary_groups.append({'name': 'ceval-hard', 'subsets': _ceval_hard})

_ceval_all = _ceval_stem + _ceval_social_science + _ceval_humanities + _ceval_other
_ceval_weights = {'professional_tour_guide': 29, 'high_school_geography': 19, 'logic': 22, 'middle_school_politics': 21, 'college_chemistry': 24, 'electrical_engineer': 37, 'high_school_biology': 19, 'metrology_engineer': 24, 'high_school_history': 20, 'physician': 49, 'middle_school_physics': 19, 'marxism': 19, 'college_programming': 37, 'ideological_and_moral_cultivation': 19, 'teacher_qualification': 44, 'college_physics': 19, 'legal_professional': 23, 'computer_network': 19, 'middle_school_biology': 21, 'advanced_mathematics': 19, 'middle_school_chemistry': 20, 'middle_school_geography': 12, 'law': 24, 'college_economics': 55, 'mao_zedong_thought': 24, 'computer_architecture': 21, 'veterinary_medicine': 23, 'education_science': 29, 'art_studies': 33, 'middle_school_history': 22, 'clinical_medicine': 22, 'accountant': 49, 'chinese_language_and_literature': 23, 'modern_chinese_history': 23, 'probability_and_statistics': 18, 'civil_servant': 47, 'basic_medicine': 19, 'high_school_physics': 19, 'high_school_chemistry': 19, 'operating_system': 19, 'high_school_mathematics': 18, 'fire_engineer': 31, 'plant_protection': 22, 'discrete_mathematics': 16, 'environmental_impact_assessment_engineer': 31, 'high_school_chinese': 19, 'business_administration': 33, 'tax_accountant': 49, 'high_school_politics': 19, 'urban_and_rural_planner': 46, 'middle_school_mathematics': 19, 'sports_science': 19}
_ceval_weights = {'ceval-' + k : v for k,v in _ceval_weights.items()}
ceval_summary_groups.append({'name': 'ceval', 'subsets': _ceval_all})
ceval_summary_groups.append({'name': 'ceval-weighted', 'subsets': _ceval_all, 'weights': _ceval_weights})

_ceval_stem = ['computer_network', 'operating_system', 'computer_architecture', 'college_programming', 'college_physics', 'college_chemistry', 'advanced_mathematics', 'probability_and_statistics', 'discrete_mathematics', 'electrical_engineer', 'metrology_engineer', 'high_school_mathematics', 'high_school_physics', 'high_school_chemistry', 'high_school_biology', 'middle_school_mathematics', 'middle_school_biology', 'middle_school_physics', 'middle_school_chemistry', 'veterinary_medicine']
_ceval_stem = ['ceval-test-' + s for s in _ceval_stem]
ceval_summary_groups.append({'name': 'ceval-test-stem', 'subsets': _ceval_stem})

_ceval_social_science = ['college_economics', 'business_administration', 'marxism', 'mao_zedong_thought', 'education_science', 'teacher_qualification', 'high_school_politics', 'high_school_geography', 'middle_school_politics', 'middle_school_geography']
_ceval_social_science = ['ceval-test-' + s for s in _ceval_social_science]
ceval_summary_groups.append({'name': 'ceval-test-social-science', 'subsets': _ceval_social_science})

_ceval_humanities = ['modern_chinese_history', 'ideological_and_moral_cultivation', 'logic', 'law', 'chinese_language_and_literature', 'art_studies', 'professional_tour_guide', 'legal_professional', 'high_school_chinese', 'high_school_history', 'middle_school_history']
_ceval_humanities = ['ceval-test-' + s for s in _ceval_humanities]
ceval_summary_groups.append({'name': 'ceval-test-humanities', 'subsets': _ceval_humanities})

_ceval_other = ['civil_servant', 'sports_science', 'plant_protection', 'basic_medicine', 'clinical_medicine', 'urban_and_rural_planner', 'accountant', 'fire_engineer', 'environmental_impact_assessment_engineer', 'tax_accountant', 'physician']
_ceval_other = ['ceval-test-' + s for s in _ceval_other]
ceval_summary_groups.append({'name': 'ceval-test-other', 'subsets': _ceval_other})

_ceval_hard = ['advanced_mathematics', 'discrete_mathematics', 'probability_and_statistics', 'college_chemistry', 'college_physics', 'high_school_mathematics', 'high_school_chemistry', 'high_school_physics']
_ceval_hard = ['ceval-test-' + s for s in _ceval_hard]
ceval_summary_groups.append({'name': 'ceval-test-hard', 'subsets': _ceval_hard})

_ceval_all = _ceval_stem + _ceval_social_science + _ceval_humanities + _ceval_other
_ceval_test_weights = {'mao_zedong_thought': 219, 'modern_chinese_history': 212, 'legal_professional': 215, 'education_science': 270, 'high_school_biology': 175, 'chinese_language_and_literature': 209, 'accountant': 443, 'middle_school_chemistry': 185, 'plant_protection': 199, 'veterinary_medicine': 210, 'fire_engineer': 282, 'middle_school_biology': 192, 'high_school_mathematics': 166, 'high_school_chinese': 178, 'high_school_chemistry': 172, 'law': 221, 'middle_school_geography': 108, 'discrete_mathematics': 153, 'basic_medicine': 175, 'operating_system': 179, 'electrical_engineer': 339, 'college_programming': 342, 'marxism': 179, 'urban_and_rural_planner': 418, 'computer_architecture': 193, 'professional_tour_guide': 266, 'ideological_and_moral_cultivation': 172, 'high_school_physics': 175, 'clinical_medicine': 200, 'advanced_mathematics': 173, 'high_school_geography': 178, 'environmental_impact_assessment_engineer': 281, 'art_studies': 298, 'tax_accountant': 443, 'business_administration': 301, 'computer_network': 171, 'metrology_engineer': 219, 'logic': 204, 'probability_and_statistics': 166, 'high_school_politics': 176, 'middle_school_mathematics': 177, 'teacher_qualification': 399, 'physician': 443, 'civil_servant': 429, 'college_chemistry': 224, 'sports_science': 180, 'middle_school_history': 207, 'middle_school_politics': 193, 'college_physics': 176, 'college_economics': 497, 'high_school_history': 182, 'middle_school_physics': 178}
_ceval_test_weights = {'ceval-test-' + k : v for k,v in _ceval_test_weights.items()}
ceval_summary_groups.append({'name': 'ceval-test', 'subsets': _ceval_all})
ceval_summary_groups.append({'name': 'ceval-test-weighted', 'subsets': _ceval_all, 'weights': _ceval_test_weights})