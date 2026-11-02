import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from ais_bench.benchmark.datasets.cmath import CMATHDataset, CmathEvaluator, extract_digits_prediction, match_digit_response

@pytest.mark.parametrize('suffix', ['jsonl', 'json'])
def test_loader(tmp_path, suffix):
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
def test_bad_row(tmp_path, row):
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps([row]), encoding='utf-8')
    with pytest.raises(ValueError):
        CMATHDataset.load(str(path))

def test_empty_loader(tmp_path):
    path = tmp_path / 'empty.json'
    path.write_text('[]', encoding='utf-8')
    with pytest.raises(ValueError):
        CMATHDataset.load(str(path))

@pytest.mark.parametrize('text,ref,expected', [('答案：9','9',True), ('答案：8','9',False), ('没有答案','9',False), (None,'9',False), ('答案：50%','0.5',True), ('答案：1/2','0.5',True), ('答案：1.005','1.0',True), ('答案：1.02','1.0',False)])
def test_scoring(text, ref, expected):
    assert match_digit_response(ref, extract_digits_prediction(text)) is expected

def test_aggregation():
    evaluator = CmathEvaluator()
    result = evaluator.score(['答案：9', '答案：8', None, '无答案'], ['9'] * 4)
    assert result['accuracy'] == 25
    assert len(result['details']) == 4
    assert [d['correct'] for d in result['details']] == [True,False,False,False]
    assert 'error' in evaluator.score([], [])
    assert 'error' in evaluator.score(['9'], [])

@pytest.mark.parametrize('mode', ['chat_prompt', 'str'])
def test_config_manager(mode):
    from ais_bench.benchmark.cli.config_manager import ConfigManager
    manager = ConfigManager(SimpleNamespace(datasets=['cmath_gen_0_shot_cot_' + mode], config_dir=''))
    configs = manager._load_datasets_config()
    assert len(configs) == 1
    cfg = configs[0]
    assert cfg['abbr'] == 'cmath-dev'
    cfg['infer_cfg']['prompt_template']['type'](**{k:v for k,v in cfg['infer_cfg']['prompt_template'].items() if k != 'type'})


def test_dataset_reader_integration(tmp_path):
    path = tmp_path / 'data.jsonl'
    path.write_text(json.dumps(dict(input='三加二？', golden='5')), encoding='utf-8')
    dataset = CMATHDataset(path=str(path), reader_cfg=dict(input_columns=['question'], output_column='answer'))
    assert len(dataset.test) == 1
    assert len(dataset.train) == 0
