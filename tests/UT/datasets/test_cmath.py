import json
from pathlib import Path
from mmengine.config import Config
import pytest
from ais_bench.benchmark.datasets.cmath import CMATHDataset, CmathEvaluator, extract_digits_prediction, match_digit_response

@pytest.mark.parametrize('suffix', ['jsonl', 'json'])
def test_loader_expect_mapped_columns(tmp_path, suffix):
    rows = [dict(input='三加二？', golden='5', grade=1)]
    path = tmp_path / ('data.' + suffix)
    path.write_text(json.dumps(rows) if suffix == 'json' else json.dumps(rows[0]), encoding='utf-8')
    data = CMATHDataset.load(str(path))
    assert set(data) == {'train', 'test'}
    assert len(data['train']) == 0
    assert len(data['test']) == 1
    assert data['test'][0]['question'] == '三加二？'
    assert data['test'][0]['answer'] == '5'
    assert data['test'][0]['grade'] == 1

@pytest.mark.parametrize('row', [{}, {'input': 'x', 'golden': 'NaN'}, {'input': 'x', 'golden': '1/0'}])
def test_invalid_reference_expect_value_error(tmp_path, row):
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps([row]), encoding='utf-8')
    with pytest.raises(ValueError):
        CMATHDataset.load(str(path))

def test_empty_loader_expect_value_error(tmp_path):
    path = tmp_path / 'empty.json'
    path.write_text('[]', encoding='utf-8')
    with pytest.raises(ValueError):
        CMATHDataset.load(str(path))

@pytest.mark.parametrize('text,ref,expected', [('答案：9','9',True), ('答案：8','9',False), ('没有答案','9',False), (None,'9',False), ('答案：50%','0.5',True), ('答案：1/2','0.5',True), ('答案：1.005','1.0',True), ('答案：1.02','1.0',False)])
def test_scoring_expect_numeric_match(text, ref, expected):
    assert match_digit_response(ref, extract_digits_prediction(text)) is expected

def test_aggregation_expect_failures_in_denominator():
    evaluator = CmathEvaluator()
    result = evaluator.score(['答案：9', '答案：8', None, '无答案'], ['9'] * 4)
    assert result['accuracy'] == 25
    assert len(result['details']) == 4
    assert [d['correct'] for d in result['details']] == [True,False,False,False]
    assert 'error' in evaluator.score([], [])
    assert 'error' in evaluator.score(['9'], [])

@pytest.mark.parametrize('mode', ['chat_prompt', 'str'])
def test_config_contract_expect_wired_components(mode, tmp_path):
    from ais_bench.benchmark.registry import TEXT_POSTPROCESSORS
    path = Path('ais_bench/benchmark/configs/datasets/cmath') / f'cmath_gen_0_shot_cot_{mode}.py'
    cfg = Config.fromfile(str(path))['cmath_datasets'][0]
    assert cfg['type'] is CMATHDataset
    sample = tmp_path / 'sample.jsonl'
    sample.write_text(json.dumps(dict(input='题', golden='9')), encoding='utf-8')
    loaded = cfg['type'](path=str(sample), reader_cfg=cfg['reader_cfg'])
    assert loaded.test[0]['question'] == '题'
    assert loaded.test[0]['answer'] == '9'
    assert cfg['path'] == 'ais_bench/datasets/cmath'
    assert cfg['reader_cfg'] == dict(input_columns=['question'], output_column='answer')
    assert cfg['abbr'] == ('cmath-dev-chat' if mode == 'chat_prompt' else 'cmath-dev-str')
    assert cfg['eval_cfg']['evaluator']['type']().score(['9'], ['9'])['accuracy'] == 100
    processor = cfg['eval_cfg']['pred_postprocessor']['type']
    assert TEXT_POSTPROCESSORS.get('cmath') is processor
    assert processor('答案：9') == '答案：9'


def test_dataset_reader_expect_empty_train(tmp_path):
    path = tmp_path / 'data.jsonl'
    path.write_text(json.dumps(dict(input='三加二？', golden='5')), encoding='utf-8')
    dataset = CMATHDataset(path=str(path), reader_cfg=dict(input_columns=['question'], output_column='answer'))
    assert len(dataset.test) == 1
    assert len(dataset.train) == 0


@pytest.mark.parametrize('candidate', ['3/0', '1/2/3'])
def test_invalid_fraction_expect_incorrect(candidate):
    from ais_bench.benchmark.datasets.cmath import string2num
    with pytest.raises(ValueError):
        string2num(candidate)
    assert not match_digit_response('0', [candidate])

@pytest.mark.parametrize('text', [None, [], {}, 42, 'timeout', '{"error":"failed"}'])
def test_exception_prediction_expect_incorrect(text):
    assert CmathEvaluator().score([text], ['0'])['accuracy'] == 0

@pytest.mark.parametrize('text', ['答案：3分之1', r'答案：\frac{1}{3}'])
def test_fraction_extraction_expect_numeric_equivalence(text):
    assert match_digit_response('1/3', extract_digits_prediction(text))

@pytest.mark.parametrize('mode,expected', [('h', {'1','2'}), ('t', {'4','5'}), ('ht', {'1','2','4','5'}), (None, {'1','2','3','4','5'})])
def test_truncation_expect_selected_candidates(mode, expected):
    assert set(extract_digits_prediction('1\n2\n3\n4\n5', truncation=mode)) == expected

def test_invalid_truncation_expect_value_error():
    with pytest.raises(ValueError):
        extract_digits_prediction('9', truncation='bad')

def test_directory_loader_expect_dev_file(tmp_path):
    (tmp_path / 'cmath_dev.jsonl').write_text(json.dumps(dict(input='题',golden='1')), encoding='utf-8')
    assert len(CMATHDataset.load(str(tmp_path))['test']) == 1

@pytest.mark.parametrize('answer', ['1/2/3', '3/0'])
def test_malformed_reference_expect_rejected(tmp_path, answer):
    (tmp_path / 'cmath_dev.jsonl').write_text(json.dumps(dict(input='题',golden=answer)), encoding='utf-8')
    with pytest.raises(ValueError):
        CMATHDataset.load(str(tmp_path))


def test_middle_hyphen_expect_candidate_extracted():
    from ais_bench.benchmark.datasets.cmath import extract_digits_from_line
    assert '3' in extract_digits_from_line('答案是-3 米')
