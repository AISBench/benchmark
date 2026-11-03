"""Official raw scores plus fixed display scores, coverage and an AISBench total."""
import csv
from datetime import datetime
import os
from pathlib import Path
from statistics import mean

from ais_bench.benchmark.utils.vabench import DIMENSIONS
from ais_bench.benchmark.utils.vabench.data import object_digest, read_json, write_json
from .vabench_scoring import RULES, aggregate, finite_number, normalize_score, protocol


def record_failure(dimension, score, raw):
    raw = raw or {}
    reasons = [str(raw.get('reasoning', ''))]
    answers = raw.get('answers')
    if isinstance(answers, list):
        reasons.extend(str(a.get('reason', '')) for a in answers if isinstance(a, dict))
    if any('OOM_ERROR' in reason.upper() for reason in reasons):
        return 'backend_error'
    if not finite_number(score):
        return 'invalid_score'
    if dimension.startswith('third_'):
        if any('empty or oom/parse fallback' in reason.lower() for reason in reasons):
            return 'parse_error'
        if not 1 <= score <= 5:
            return 'invalid_score'
    if dimension.startswith('fourth_'):
        if not 0 <= score <= 1:
            return 'invalid_score'
        if answers is not None and (not isinstance(answers, list) or not answers or any(
                not isinstance(a, dict) or a.get('answer') not in ('yes', 'no')
                or not str(a.get('reason', '')).strip() for a in answers)):
            return 'parse_error'
    return None


def normalize_records(dimension, raw, expected, backend_errors=False, failed_ids=None):
    """Keep missing/failed outputs distinct from legitimate numerical zero.

    ``failed_ids`` marks samples the backend named explicitly, so one bad
    clip in a scene does not invalidate valid scores from its neighbours.
    """
    failed_ids = set(failed_ids or ())
    lookup = {}
    for row in raw:
        path = row.get('video_path') or row.get('audio_path')
        if not path:
            raise ValueError(f'{dimension}: result without media path')
        idx = Path(path).stem
        if idx in lookup:
            raise ValueError(f'{dimension}: duplicate result {idx}')
        lookup[idx] = row
    if set(lookup) - {e['idx'] for e in expected}:
        raise ValueError(f'{dimension}: unexpected result sample')
    result = []
    for entry in expected:
        row = lookup.get(entry['idx'])
        score = row.get('video_results') if row else None
        status = ('backend_error' if backend_errors or entry['idx'] in failed_ids
                  else 'missing' if row is None
                  else record_failure(dimension, score, row) or 'ok')
        result.append(dict(idx=entry['idx'], category=entry['category'], status=status,
                           score=score if status == 'ok' else None,
                           original_score=score if finite_number(score) else None, raw=row))
    return result


def summarize_dimension(records, dimension=None):
    def stats(rows):
        valid = [r['score'] for r in rows if r['status'] == 'ok' and finite_number(r.get('score'))]
        originals = [r.get('original_score', (r.get('raw') or {}).get('video_results', r.get('score')))
                     for r in rows]
        originals = [v for v in originals if finite_number(v)]
        expected = len(rows)
        report = dict(expected=expected, scored=len(valid),
                      missing=sum(r['status'] == 'missing' for r in rows),
                      failed=sum(r['status'] not in ('ok', 'missing', 'not_evaluated') for r in rows),
                      not_evaluated=sum(r['status'] == 'not_evaluated' for r in rows),
                      mean=mean(valid) if valid else None,
                      official_numeric_mean=mean(originals) if originals else None,
                      coverage=len(valid) / expected if expected else 0.0,
                      coverage_percent=100 * len(valid) / expected if expected else 0.0)
        report['omitted'] = report['missing'] if dimension == 'second_lsa' else 0
        report['status'] = ('complete' if expected and len(valid) == expected else
                            'not_evaluated' if report['not_evaluated'] == expected else 'incomplete')
        if dimension is not None:
            report.update(normalize_score(dimension, report['mean']))
            low, high, _ = RULES[dimension]
            report['out_of_bounds_count'] = sum(v < low or v > high for v in valid)
        return report
    report = stats(records)
    report['scenes'] = {cat: stats([r for r in records if r['category'] == cat])
                        for cat in sorted({r['category'] for r in records})}
    return report


def build_report(manifest, dimensions, provenance, requested_dimensions=None):
    scope = manifest.get('evaluation_scope', 'legacy')
    expected = {r['idx']: r for r in manifest['records']}
    if len(expected) != len(manifest['records']):
        raise ValueError('Duplicate input sample IDs')
    details, metrics = {}, {}
    for dimension in DIMENSIONS:
        value = dimensions.get(dimension)
        rows = value.get('records', []) if value is not None else []
        lookup = {r['idx']: r for r in rows}
        if len(lookup) != len(rows) or set(lookup) - set(expected):
            raise ValueError(f'{dimension}: duplicate/unexpected sample records')
        aligned = []
        for idx, entry in expected.items():
            if idx not in lookup:
                aligned.append(dict(idx=idx, category=entry.get('category', 'unknown'), score=None,
                                    original_score=None, raw=None,
                                    status='not_evaluated' if value is None else 'missing'))
                continue
            row = dict(lookup[idx])
            row.setdefault('category', entry.get('category', 'unknown'))
            row.setdefault('original_score', (row.get('raw') or {}).get('video_results', row.get('score')))
            if row['status'] == 'ok':
                row['status'] = record_failure(dimension, row.get('score'), row.get('raw')) or 'ok'
            if row['status'] != 'ok':
                row['score'] = None
            aligned.append(row)
        metrics[dimension] = summarize_dimension(aligned, dimension)
        if value is not None:
            metrics[dimension]['backend_issue_count'] = len(value.get('errors', []))
            if value.get('errors') and metrics[dimension]['status'] == 'complete':
                metrics[dimension]['status'] = 'incomplete'
            details[dimension] = dict(value, records=aligned)
    selected = list(dimensions) if requested_dimensions is None else list(requested_dimensions)
    normalization = protocol()
    scope_note = ''
    if scope == 'smoke':
        scope_note = 'Single-sample smoke test; not full 778-set benchmark acceptance.'
    elif scope == 'custom':
        scope_note = (f'Custom subset of {len(expected)} pinned sample(s); '
                      'not full 778-set benchmark acceptance.')
    total = aggregate(metrics)
    if scope == 'custom' and total['status'] in ('formal', 'preview'):
        total['label'] = ('自定义子集总分（非官方）' if total['status'] == 'formal'
                          else '自定义子集预览总分（非官方）')
    return dict(schema_version=2, input_signature=manifest['input_signature'], evaluation_scope=scope,
                scope_note=scope_note,
                sample_ids=sorted(expected), requested_dimensions=selected,
                provenance=provenance, details=details, metrics=metrics,
                directions={d: 'lower' if d == 'second_desync' else 'higher' for d in DIMENSIONS},
                normalization_protocol=normalization,
                normalization_protocol_sha256=object_digest(normalization), total=total)


def refresh_report(report):
    """Reaggregate existing AISBench per-sample outputs without scoring again."""
    if not report.get('input_signature') or not report.get('sample_ids') or not isinstance(report.get('details'), dict):
        raise ValueError('An AISBench report with input_signature, sample_ids and per-sample details is required')
    categories = {}
    for value in report['details'].values():
        for row in value.get('records', []):
            idx, cat = row['idx'], row.get('category', 'unknown')
            if idx in categories and categories[idx] != cat:
                raise ValueError(f'Inconsistent sample category: {idx}')
            categories[idx] = cat
    manifest = dict(input_signature=report['input_signature'],
                    evaluation_scope=report.get('evaluation_scope', 'legacy'),
                    records=[dict(idx=idx, category=categories.get(idx, 'unknown')) for idx in report['sample_ids']])
    result = build_report(manifest, report['details'], report.get('provenance', {}),
                          report.get('requested_dimensions', list(report['details'])))
    if report.get('postprocessing'):
        result['postprocessing'] = report['postprocessing']
    return result


def display_report(report):
    if report.get('input_signature') and report.get('sample_ids'):
        return refresh_report(report)
    # Metric-only legacy artifacts can display native means, but cannot certify a total.
    metrics = {}
    expected = max((m.get('expected', 0) for m in report.get('metrics', {}).values()), default=0)
    for dimension in DIMENSIONS:
        data = dict(report.get('metrics', {}).get(dimension, {}))
        data.setdefault('mean', None)
        for field in ('scored', 'missing', 'failed', 'not_evaluated'):
            data.setdefault(field, 0)
        data.setdefault('expected', expected)
        data.setdefault('scenes', {})
        data['omitted'] = data['missing'] if dimension == 'second_lsa' else 0
        data['coverage_percent'] = 100 * data['scored'] / data['expected'] if data['expected'] else 0.0
        data['status'] = 'incomplete'
        data.update(normalize_score(dimension, data['mean']))
        data.setdefault('out_of_bounds_count', 0)
        metrics[dimension] = data
    result = dict(report, metrics=metrics, directions={
        d: 'lower' if d == 'second_desync' else 'higher' for d in DIMENSIONS})
    result['total'] = aggregate(metrics)
    result['total'].update(score=None, status='unavailable', label='总分不可用',
                           note='Metric-only legacy report lacks auditable per-sample records')
    return result


def format_summary(report):
    rows = [f'VABench: scope={report.get("evaluation_scope", "legacy")}, '
            f'samples={len(report.get("sample_ids", []))}', '',
            '| 指标 | 原始均分 | 归一化分(0–100) | 有效/输入 | LSA省略 | 缺失(含省略) | 失败 | 覆盖率 | 状态 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for dimension, values in report['metrics'].items():
        raw = '—' if values['mean'] is None else f'{values["mean"]:.5f}'
        score = values.get('normalized_score')
        normalized = '—' if score is None else f'{score:.2f}'
        rows.append(f'| {dimension} | {raw} | {normalized} | {values["scored"]}/{values["expected"]} '
                    f'| {values.get("omitted", 0)} | {values["missing"]} | {values["failed"]} '
                    f'| {values["coverage_percent"]:.1f}% | {values["status"]} |')
    total = report['total']
    value = '—' if total['score'] is None else f'{total["score"]:.2f}/100'
    rows += ['', f'{total["label"]}: {value} '
             f'(有效指标 {total["valid_dimensions"]}/{total["expected_dimensions"]})',
             '总分采用固定归一化后的15项等权平均；这是AISBench展示协议，只代表当前选定样本。']
    if report.get('scope_note'):
        rows += [report['scope_note']]
    return '\n'.join(rows) + '\n'


def export_csv(report, output):
    output = Path(output)
    columns = ['mean', 'normalized_score', 'expected', 'scored', 'omitted', 'missing', 'failed',
               'not_evaluated', 'coverage_percent', 'out_of_bounds_count', 'mean_out_of_bounds', 'status']
    for filename, groups in [('metrics.csv', False), ('scenes.csv', True)]:
        with (output / filename).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['dimension'] + (['category'] if groups else ['direction']) + columns)
            for dim, values in report['metrics'].items():
                entries = values['scenes'].items() if groups else [(report['directions'][dim], values)]
                for category, data in entries:
                    writer.writerow([dim, category] + [data.get(k) for k in columns])
    with (output / 'samples.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dimension', 'idx', 'category', 'status', 'score', 'original_score', 'requested_device'])
        for dim, values in report['details'].items():
            for row in values['records']:
                writer.writerow([dim] + [row.get(k) for k in ('idx', 'category', 'status', 'score', 'original_score')]
                                + [values.get('requested_device', 'unobserved')])
    with (output / 'total.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['score', 'status', 'label', 'valid_dimensions', 'expected_dimensions'])
        writer.writerow([report['total'][k] for k in (
            'score', 'status', 'label', 'valid_dimensions', 'expected_dimensions')])
    (output / 'summary.md').write_text(format_summary(report), encoding='utf-8')


def _media_key(value):
    if not isinstance(value, (str, os.PathLike)):
        return None
    return os.path.normpath(os.fspath(value))


def _prepared_prompt_catalog(prepared):
    """Prompt text per category, keyed by the exact media paths prepared for it."""
    catalog = {}
    if prepared is None:
        return catalog
    root = Path(prepared) / 'json'
    if not root.is_dir():
        return catalog
    for path in sorted(root.glob('*.json')):
        try:
            entries = read_json(path)
        except (OSError, ValueError):
            continue
        if not isinstance(entries, list):
            continue
        entries = [entry for entry in entries if isinstance(entry, dict)]
        prompts, media = [], {}
        for entry in entries:
            prompt = entry.get('prompt_en')
            if not isinstance(prompt, str) or not prompt:
                continue
            if prompt not in prompts:
                prompts.append(prompt)
            for field in ('video_list', 'audio_list'):
                values = entry.get(field)
                if not isinstance(values, list):
                    continue
                for value in values:
                    key = _media_key(value)
                    if key is not None:
                        media[key] = prompt
        catalog[path.stem] = dict(prompts=prompts, media=media, entries=entries)
    return catalog


def _record_prompt(record, raw, media):
    for field in ('video_path', 'audio_path'):
        prompt = media.get(_media_key(raw.get(field)))
        if prompt is not None:
            return prompt
    for field in ('prompt_name', 'prompt_en', 'prompt'):
        prompt = raw.get(field)
        if isinstance(prompt, str) and prompt:
            return prompt
    return str(record.get('idx', ''))


def export_official_results(report, output, prepared=None, timestamp=None):
    """Back up raw scores in the official per-category JSON/CSV layout.

    ``video_results`` keeps every scorer record, including rows AISBench marks
    as parse/backend errors, so it can be diffed against official evaluate.py.
    The CSV repeats the official prompt-level aggregation and writes -1 for a
    prompt without a numeric score.
    """
    output = Path(output)
    # Keep explicit old timestamps accepted, but never use ':' in a filename.
    stamp = (timestamp or datetime.now().strftime('%Y-%m-%d-%H-%M-%S')).replace(':', '-')
    if not stamp or any(c in stamp for c in '/\\') or stamp in ('.', '..'):
        raise ValueError('Timestamp must be a filename component')
    catalog = _prepared_prompt_catalog(prepared)
    details = report.get('details', {})
    dimensions = list(details)
    categories = sorted({row.get('category', 'unknown')
                         for value in details.values()
                         for row in value.get('records', [])})
    for category in categories:
        folder = output / 'official_eval_results' / category
        name = f'{category}_{stamp}'
        folder.mkdir(parents=True, exist_ok=True)
        category_rows, results = {}, {}
        for dimension in dimensions:
            rows = [row for row in details[dimension].get('records', [])
                    if row.get('category', 'unknown') == category]
            category_rows[dimension] = rows
            results[dimension] = dict(video_results=[
                row['raw'] for row in rows if isinstance(row.get('raw'), dict)])
        write_json(folder / f'{name}_eval_results.json', results)
        prompt_info = catalog.get(category, {})
        prompts = list(prompt_info.get('prompts', []))
        media = prompt_info.get('media', {})
        scores = {}
        for dimension, rows in category_rows.items():
            buckets = scores.setdefault(dimension, {})
            for row in rows:
                raw = row.get('raw')
                if not isinstance(raw, dict):
                    continue
                score = raw.get('video_results')
                if not isinstance(score, (int, float)) or isinstance(score, bool):
                    continue
                prompt = _record_prompt(row, raw, media)
                if prompt not in prompts:
                    prompts.append(prompt)
                buckets.setdefault(prompt, []).append(float(score))
        with (folder / f'{name}_eval_results.csv').open(
                'w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['prompt_name'] + dimensions)
            for prompt in prompts:
                values = []
                for dimension in dimensions:
                    items = scores.get(dimension, {}).get(prompt, [])
                    values.append(sum(items) / len(items) if items else -1)
                writer.writerow([prompt] + values)
        if prompt_info.get('entries'):
            write_json(folder / f'{name}_full_info.json', prompt_info['entries'])
    return dict(directory=output / 'official_eval_results', timestamp=stamp,
                categories=categories)


def compare(reference, candidate, dimensions=None, relative=0.01,
            near_zero=1e-6, absolute=1e-6):
    """Same media AND protocol, source, weights, preprocessing, status sets.

    Different devices are allowed: this is also the GPU-vs-NPU audit tool.
    Each dataset mean is sample-weighted, never an average of scene means.
    """
    if not all(finite_number(v) for v in (relative, near_zero, absolute)) or relative <= 0 or near_zero < 0 or absolute < 0:
        raise ValueError('Invalid tolerance')
    dimensions = list(DIMENSIONS if dimensions is None else dimensions)
    if not dimensions or len(set(dimensions)) != len(dimensions):
        raise ValueError('Nonempty, unique dimensions required')
    problems = []
    for field in ('input_signature', 'sample_ids'):
        if not reference.get(field) or reference.get(field) != candidate.get(field):
            problems.append(f'{field} mismatch or missing')
    for key in ('protocol_sha256', 'source_signature', 'integration_signature',
                'weights_signature', 'scoring_signature'):
        left = reference.get('provenance', {}).get(key)
        right = candidate.get('provenance', {}).get(key)
        if not left or left != right:
            problems.append(f'{key} mismatch or missing')
    reports = {}
    for dim in dimensions:
        left = reference.get('details', {}).get(dim)
        right = candidate.get('details', {}).get(dim)
        issues = []
        a = b = None
        if not left or not right:
            issues.append('dimension missing')
        else:
            lrows, rrows = left['records'], right['records']
            ls = {r['idx']: r['status'] for r in lrows}
            rs = {r['idx']: r['status'] for r in rrows}
            if len(ls) != len(lrows) or len(rs) != len(rrows):
                issues.append('duplicate sample result')
            if ls != rs or sorted(ls) != reference.get('sample_ids', []):
                issues.append('sample/status set mismatch')
            if any(s not in ('ok', 'missing') for s in list(ls.values()) + list(rs.values())):
                issues.append('backend failure/invalid score')
            if dim != 'second_lsa' and ('missing' in ls.values() or 'missing' in rs.values()):
                issues.append('unexpected missing score')
            if not any(s == 'ok' for s in ls.values()):
                issues.append('no valid scores')
            if any(not finite_number(r.get('score')) for r in lrows + rrows if r['status'] == 'ok'):
                issues.append('nonfinite score')
            if not issues:
                a = mean(r['score'] for r in lrows if r['status'] == 'ok')
                b = mean(r['score'] for r in rrows if r['status'] == 'ok')
        delta = abs(a - b) if a is not None else None
        error = delta / abs(a) if a is not None and abs(a) > near_zero else None
        passed = (delta is not None and not issues and
                  (delta <= absolute if error is None else error < relative))
        reports[dim] = dict(passed=passed, reference_mean=a, candidate_mean=b,
                            absolute_error=delta, relative_error=error,
                            rule='absolute' if error is None else 'relative', issues=issues)
    return dict(passed=not problems and all(r['passed'] for r in reports.values()),
                protocol_issues=problems, dimensions=reports,
                thresholds=dict(relative_strictly_less_than=relative,
                                near_zero_at_most=near_zero, absolute_at_most=absolute))
