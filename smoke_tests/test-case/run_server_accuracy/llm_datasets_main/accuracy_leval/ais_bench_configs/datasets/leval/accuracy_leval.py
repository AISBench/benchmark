from mmengine import read_base

with read_base():
    from .codeu.leval_code_u_gen import LEval_code_u_datasets
    from .coursera.leval_coursera_gen import LEval_coursera_datasets
    from .financialqa.leval_financial_qa_gen import LEval_financialqa_datasets
    from .govreportsumm.leval_gov_report_summ_gen import LEval_govreport_summ_datasets
    from .gsm100.leval_gsm100_gen import LEval_gsm100_datasets
    from .legalcontractqa.leval_legal_contract_qa_gen import LEval_legalqa_datasets
    from .meetingsumm.leval_meeting_summ_gen import LEval_meetingsumm_datasets
    from .multidocqa.leval_multidoc_qa_gen import LEval_multidocqa_datasets
    from .narrativeqa.leval_narrative_qa_gen import LEval_narrativeqa_datasets
    from .naturalquestion.leval_natural_question_gen import LEval_nq_datasets
    from .newssumm.leval_news_summ_gen import LEval_newssumm_datasets
    from .paperassistant.leval_paper_assistant_gen import LEval_ps_summ_datasets
    from .patentsumm.leval_patent_summ_gen import LEval_patent_summ_datasets
    from .quality.leval_quality_gen import LEval_quality_datasets
    from .reviewsumm.leval_review_summ_gen import LEval_review_summ_datasets
    from .scientificqa.leval_scientific_qa_gen import LEval_scientificqa_datasets
    from .scifi.leval_sci_fi_gen import LEval_sci_fi_datasets
    from .topicretrievallongchat.leval_topic_retrieval_gen import LEval_tr_datasets
    from .tpo.leval_tpo_gen import LEval_tpo_datasets
    from .tvshowsumm.leval_tv_show_summ_gen import LEval_tvshow_summ_datasets

for dataset in [
    *LEval_code_u_datasets,
    *LEval_coursera_datasets,
    *LEval_financialqa_datasets,
    *LEval_govreport_summ_datasets,
    *LEval_gsm100_datasets,
    *LEval_legalqa_datasets,
    *LEval_meetingsumm_datasets,
    *LEval_multidocqa_datasets,
    *LEval_narrativeqa_datasets,
    *LEval_nq_datasets,
    *LEval_newssumm_datasets,
    *LEval_ps_summ_datasets,
    *LEval_patent_summ_datasets,
    *LEval_quality_datasets,
    *LEval_review_summ_datasets,
    *LEval_scientificqa_datasets,
    *LEval_sci_fi_datasets,
    *LEval_tr_datasets,
    *LEval_tpo_datasets,
    *LEval_tvshow_summ_datasets
]:
    dataset['reader_cfg']['test_range'] = '[0:10]'