"""Contract tests for comparing AISBench backups with native VABench output."""
import json
from pathlib import Path
import pytest

from ais_bench.tools.vabench.official_compare import (
    compare_official_results, format_official_comparison)


def _write(root, category, dimension, rows, suffix='2026-01-01-00-00-00'):
    folder = Path(root) / category
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{category}_{suffix}_eval_results.json'
    path.write_text(json.dumps({
        dimension: {'video_results': rows}
    }, ensure_ascii=False), encoding='utf-8')
    return path


def _row(idx, score, root='/data'):
    return {'video_path': f'{root}/video/scene/{idx}.mp4', 'video_results': score}


def test_official_comparison_ignores_media_prefix_and_passes(tmp_path):
    left = tmp_path / 'artifacts'
    right = tmp_path / 'official'
    _write(left / 'official_eval_results', 'scene', 'first_dnsmos',
           [_row('x', 3.0, '/aisbench')])
    _write(right, 'scene', 'first_dnsmos', [_row('x', 3.0, '/official')])
    result = compare_official_results(left, right, ['first_dnsmos'])
    assert result['passed']
    assert result['compared_values'] == 1
    assert result['difference_count'] == 0
    assert 'PASS' in format_official_comparison(result)


def test_official_comparison_prints_value_difference(tmp_path):
    left = tmp_path / 'aisbench'
    right = tmp_path / 'official'
    _write(left, 'scene', 'second_clap', [_row('x', 0.5)])
    _write(right, 'scene', 'second_clap', [_row('x', 0.6)])
    result = compare_official_results(left, right, ['second_clap'])
    assert not result['passed']
    difference = result['differences'][0]
    assert difference['kind'] == 'value_mismatch'
    assert difference['sample'] == 'x'
    assert difference['official'] == 0.6
    assert difference['aisbench'] == 0.5
    report = format_official_comparison(result)
    assert 'second_clap' in report and 'delta=' in report


def test_official_comparison_reports_missing_sample_and_category(tmp_path):
    left = tmp_path / 'aisbench'
    right = tmp_path / 'official'
    _write(left, 'scene', 'first_nisqa', [_row('x', 2.0), _row('y', 2.0)])
    _write(right, 'scene', 'first_nisqa', [_row('x', 2.0)])
    _write(right, 'other', 'first_nisqa', [_row('z', 2.0)])
    result = compare_official_results(left, right, ['first_nisqa'])
    kinds = {(item['kind'], item.get('sample'), item.get('category'))
             for item in result['differences']}
    assert ('missing_sample', 'y', 'scene') in kinds
    assert ('missing_category', None, 'other') in kinds
    assert not result['passed']


def test_official_comparison_tolerance_is_strict_and_near_zero_uses_absolute(tmp_path):
    left = tmp_path / 'aisbench'
    right = tmp_path / 'official'
    _write(left, 'scene', 'second_clap', [_row('x', 0.0)])
    _write(right, 'scene', 'second_clap', [_row('x', 0.0)])
    assert compare_official_results(left, right, ['second_clap'])['passed']
    _write(right, 'scene', 'second_clap', [_row('x', 0.01)])
    assert not compare_official_results(left, right, ['second_clap'])['passed']


def test_official_comparison_rejects_missing_result_directory(tmp_path):
    try:
        compare_official_results(tmp_path / 'missing', tmp_path / 'also-missing')
    except ValueError as exc:
        assert 'does not exist' in str(exc)
    else:
        raise AssertionError('missing directory must fail')


@pytest.mark.parametrize('reference,candidate,passed,rule', [
    (2e-6, 2.5e-6, False, 'relative'),
    (-2e-6, -2.5e-6, False, 'relative'),
    (100, 101, False, 'relative'),
    (-100, -100.5, True, 'relative'),
    (1e-6, 2e-6, True, 'absolute'),
    (-1e-6, -2e-6, True, 'absolute'),
    (1e-6, 2.01e-6, False, 'absolute'),
])
def test_mean_threshold_has_no_absolute_escape_above_near_zero(
        tmp_path, reference, candidate, passed, rule):
    left, right = tmp_path / 'left', tmp_path / 'right'
    _write(left, 'scene', 'second_clap', [_row('x', candidate)])
    _write(right, 'scene', 'second_clap', [_row('x', reference)])
    result = compare_official_results(left, right, ['second_clap'])
    assert result['passed'] is passed
    stats = result['summary']['second_clap']['mean_comparison']
    assert stats['rule'] == rule and stats['passed'] is passed


def test_empty_metric_cannot_hide_behind_another_valid_metric(tmp_path):
    left, right = tmp_path / 'left', tmp_path / 'right'
    for root in (left, right):
        path = _write(root, 'scene', 'second_clap', [])
        path.write_text(json.dumps({
            'second_clap': {'video_results': []},
            'first_dnsmos': {'video_results': [_row('x', 3)]}}), encoding='utf-8')
    result = compare_official_results(left, right, ['second_clap', 'first_dnsmos'])
    assert not result['passed']
    assert not result['summary']['second_clap']['mean_comparison']['passed']
    assert result['summary']['first_dnsmos']['mean_comparison']['passed']
    assert any(item['kind'] == 'no_valid_pairs' for item in result['differences'])


def test_both_empty_results_are_unverified_not_a_pass(tmp_path):
    left, right = tmp_path / 'left', tmp_path / 'right'
    for root in (left, right):
        _write(root, 'scene', 'second_lsa', [])
    result = compare_official_results(left, right, ['second_lsa'])
    assert not result['passed'] and result['compared_values'] == 0
    assert 'no_valid_pairs' in format_official_comparison(result)


def test_raw_mean_acceptance_and_sample_diagnostic_are_separate(tmp_path):
    left, right = tmp_path / 'left', tmp_path / 'right'
    _write(left, 'scene', 'first_nisqa', [_row('x', 4), _row('y', 2)])
    _write(right, 'scene', 'first_nisqa', [_row('x', 3), _row('y', 3)])
    result = compare_official_results(left, right, ['first_nisqa'])
    assert result['passed'] and result['metric_means_passed']
    assert not result['sample_values_passed']
    assert result['summary']['first_nisqa']['mean_comparison']['official_mean'] == 3
    text = format_official_comparison(result)
    assert 'raw-mean comparison: PASS' in text
    assert 'per-sample numeric diagnostic: FAIL' in text


def test_global_mean_weights_videos_not_scenes(tmp_path):
    left, right = tmp_path / 'left', tmp_path / 'right'
    for root in (left, right):
        _write(root, 'big', 'second_clap', [_row('x', 0), _row('y', 0)])
        _write(root, 'small', 'second_clap', [_row('z', 0.9)])
    result = compare_official_results(left, right, ['second_clap'])
    assert result['summary']['second_clap']['mean_comparison']['official_mean'] == pytest.approx(0.3)
    assert result['passed']


def test_comparison_rejects_ambiguous_runs_instead_of_mixing_latest_files(tmp_path):
    left, right = tmp_path / 'left', tmp_path / 'right'
    _write(left, 'scene', 'second_clap', [_row('x', 0)])
    _write(left, 'scene', 'second_clap', [_row('x', 1)], suffix='another-run')
    _write(right, 'scene', 'second_clap', [_row('x', 1)])
    with pytest.raises(ValueError, match='single-run'):
        compare_official_results(left, right, ['second_clap'])


@pytest.mark.parametrize('row', [
    {'idx': 'x', 'video_results': None},
    {'sample_id': 'x', 'video_results': float('nan')},
    {'audio_path': 'x.wav', 'video_results': float('inf')},
    {'video_path': 'x.mp4', 'video_results': True},
])
def test_non_numeric_output_never_certifies_a_mean(tmp_path, row):
    left, right = tmp_path / 'left', tmp_path / 'right'
    _write(left, 'scene', 'second_clap', [row])
    _write(right, 'scene', 'second_clap', [_row('x', 1)])
    result = compare_official_results(left, right, ['second_clap'])
    assert not result['passed']
    assert 'non-numeric' in format_official_comparison(result)


@pytest.mark.parametrize('dimensions,kwargs', [
    ([], {}), (['unknown'], {}), (['second_clap', 'second_clap'], {}),
    (None, {'relative': 0}), (None, {'near_zero': -1}),
    (None, {'absolute': -1}), (None, {'relative': float('nan')}),
])
def test_invalid_comparison_options_are_rejected(tmp_path, dimensions, kwargs):
    with pytest.raises(ValueError):
        compare_official_results(tmp_path, tmp_path, dimensions, **kwargs)


@pytest.mark.parametrize('payload,message', [
    ('{', 'Cannot read'), ('[]', 'not a JSON object'),
    ('{"second_clap": []}', 'not an object'),
    ('{"second_clap": {"video_results": {}}}', 'not a list'),
    ('{"second_clap": {"video_results": [1]}}', 'not an object'),
    ('{"second_clap": {"video_results": [{}]}}', 'no media path'),
])
def test_malformed_native_output_is_reported(tmp_path, payload, message):
    left, right = tmp_path / 'left', tmp_path / 'right'
    path = _write(left, 'scene', 'second_clap', [])
    path.write_text(payload, encoding='utf-8')
    _write(right, 'scene', 'second_clap', [])
    with pytest.raises(ValueError, match=message):
        compare_official_results(left, right, ['second_clap'])


def test_missing_categories_dimensions_and_samples_on_both_sides(tmp_path):
    left, right = tmp_path / 'left', tmp_path / 'right'
    _write(left, 'only_left', 'second_clap', [_row('x', 0)])
    _write(right, 'only_right', 'second_clap', [_row('x', 0)])
    _write(left, 'same', 'second_clap', [_row('a', 0)])
    _write(right, 'same', 'second_clap', [_row('b', 0)])
    _write(left, 'different', 'first_nisqa', [])
    _write(right, 'different', 'first_dnsmos', [])
    result = compare_official_results(left, right, ['first_nisqa', 'first_dnsmos', 'second_clap'])
    assert not result['passed']
    text = format_official_comparison(result)
    assert '[missing category]' in text and '[missing dimension]' in text and '[missing sample]' in text
    empty = tmp_path / 'empty'
    empty.mkdir()
    with pytest.raises(ValueError, match='No .* found'):
        compare_official_results(empty, right)


def test_duplicate_rows_and_windows_media_prefixes(tmp_path):
    left, right = tmp_path / 'left', tmp_path / 'right'
    _write(left, 'scene', 'second_clap', [dict(video_path=r'C:\videos\x.mp4', video_results=1)])
    _write(right, 'scene', 'second_clap', [_row('x', 1)])
    assert compare_official_results(left, right, ['second_clap'])['passed']
    _write(left, 'scene', 'second_clap', [_row('x', 1)] * 2)
    with pytest.raises(ValueError, match='duplicate sample'):
        compare_official_results(left, right, ['second_clap'])


@pytest.mark.parametrize('score,exit_code', [(0.5, 0), (0.8, 1)])
def test_cli_writes_comparison_means_and_returns_acceptance_status(tmp_path, capsys, score, exit_code):
    from ais_bench.tools.vabench.cli import main
    left, right = tmp_path / 'left', tmp_path / 'right'
    _write(left, 'scene', 'second_clap', [_row('x', score)])
    _write(right, 'scene', 'second_clap', [_row('x', 0.5)])
    output = tmp_path / 'comparison.json'
    assert main(['compare-official', '--aisbench', str(left), '--official', str(right),
                 '--dimensions', 'second_clap', '--output', str(output)]) == exit_code
    saved = json.loads(output.read_text(encoding='utf-8'))
    assert saved['summary']['second_clap']['mean_comparison']['official_mean'] == 0.5
    assert 'raw-mean comparison:' in capsys.readouterr().out


@pytest.mark.parametrize('dimension,row', [
    ('third_alignment', dict(video_path='x.mp4', video_results=1,
                            reasoning='No reasoning provided (empty or OOM/parse fallback).')),
    ('third_alignment', dict(video_path='x.mp4', video_results=0)),
    ('fourth_qa_audio', dict(video_path='x.mp4', video_results=0, answers=[])),
    ('fourth_qa_audio', dict(video_path='x.mp4', video_results=0, reasoning='OOM_ERROR')),
])
def test_equal_known_failure_fallbacks_do_not_certify_official_alignment(tmp_path, dimension, row):
    left, right = tmp_path / 'left', tmp_path / 'right'
    for root in (left, right):
        _write(root, 'scene', dimension, [row])
    result = compare_official_results(left, right, [dimension])
    assert not result['passed']
    assert any(item['kind'] == 'invalid_scoring_output' for item in result['differences'])
    assert result['summary'][dimension]['valid_pairs'] == 0
