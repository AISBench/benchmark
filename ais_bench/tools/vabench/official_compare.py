"""Compare AISBench official-format backups with native VABench results.

The native ``evaluate.py`` and AISBench write the same per-category
``*_eval_results.json`` shape, but use different absolute media paths.  This
module aligns records by category, dimension and media basename, then compares
raw means of the primary score (``video_results``), with sample diagnostics.
Reasoning/auxiliary fields only identify known scorer failures; their text is
not required to match. Input/model provenance must be checked separately.
"""
import json
import math
from pathlib import Path
from statistics import mean

from ais_bench.benchmark.utils.vabench import DIMENSIONS
from ais_bench.benchmark.summarizers.vabench.results import record_failure


def _is_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _result_root(path):
    path = Path(path)
    nested = path / 'official_eval_results'
    return nested if nested.is_dir() else path


def _result_files(root):
    """Require one run per category; never silently mix multiple runs."""
    if not root.is_dir():
        raise ValueError(f'Result directory does not exist: {root}')
    grouped = {}
    for path in sorted(root.rglob('*_eval_results.json')):
        grouped.setdefault(path.parent.name, []).append(path)
    if not grouped:
        raise ValueError(f'No *_eval_results.json found under: {root}')
    selected = {}
    for category, paths in grouped.items():
        if len(paths) != 1:
            raise ValueError(f'Ambiguous result files for {category}: use a single-run directory')
        selected[category] = paths[0]
    return selected


def _load_category(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'Cannot read result file {path}: {exc}') from exc
    if not isinstance(value, dict):
        raise ValueError(f'Result file is not a JSON object: {path}')
    return value


def _sample_id(row, category, dimension):
    if not isinstance(row, dict):
        raise ValueError(f'{category}/{dimension}: result is not an object')
    for field in ('video_path', 'audio_path'):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return Path(value.replace('\\', '/')).stem
    for field in ('idx', 'sample_id'):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f'{category}/{dimension}: result has no media path or sample id')


def _samples(value, category, dimension):
    if not isinstance(value, dict):
        raise ValueError(f'{category}/{dimension}: dimension result is not an object')
    rows = value.get('video_results')
    if rows is None:
        rows = []
    if not isinstance(rows, list):
        raise ValueError(f'{category}/{dimension}: video_results is not a list')
    lookup = {}
    for row in rows:
        idx = _sample_id(row, category, dimension)
        if idx in lookup:
            raise ValueError(f'{category}/{dimension}: duplicate sample result {idx!r}')
        lookup[idx] = row
    return lookup


def _comparison(reference, candidate, relative, near_zero, absolute):
    delta = abs(candidate - reference)
    error = delta / abs(reference) if abs(reference) > near_zero else None
    return dict(passed=delta <= absolute if error is None else error < relative,
                absolute_error=delta, relative_error=error,
                rule='absolute' if error is None else 'relative')


def compare_official_results(aisbench, official, dimensions=None, relative=0.01,
                             near_zero=1e-6, absolute=1e-6):
    """Diff AISBench's ``official_eval_results`` against native VABench output.

    ``aisbench`` may be either the artifacts directory or its nested
    ``official_eval_results`` directory.  ``official`` is the directory passed
    to native ``evaluate.py --output_dir``.  Numeric tolerances use the native
    official score as the reference. Each metric's raw mean is the acceptance
    value; per-sample value differences are reported separately as diagnostics.
    Above ``near_zero`` only strict relative error applies; otherwise use
    absolute error. Every requested metric requires valid matched samples.
    This numeric comparison does not certify model/input provenance or hidden
    scorer failures; those must be verified from manifests and backend logs.
    """
    if not all(_is_number(v) for v in (relative, near_zero, absolute)):
        raise ValueError('Invalid tolerance')
    if relative <= 0 or near_zero < 0 or absolute < 0:
        raise ValueError('Invalid tolerance')
    dimensions = list(DIMENSIONS if dimensions is None else dimensions)
    if not dimensions or len(set(dimensions)) != len(dimensions):
        raise ValueError('Nonempty, unique dimensions required')
    if any(dimension not in DIMENSIONS for dimension in dimensions):
        raise ValueError('Unknown VABench dimension')

    aisbench_root, official_root = _result_root(aisbench), _result_root(official)
    aisbench_files = _result_files(aisbench_root)
    official_files = _result_files(official_root)

    differences = []
    compared = 0
    summary = {
        dimension: dict(compared=0, matched=0, mismatched=0,
                        missing_official=0, missing_aisbench=0,
                        max_absolute_error=None, max_relative_error=None)
        for dimension in dimensions
    }
    values = {dimension: dict(official=[], aisbench=[]) for dimension in dimensions}
    categories = sorted(set(aisbench_files) | set(official_files))

    def add(kind, **values):
        differences.append(dict(kind=kind, **values))

    for category in categories:
        left_path = aisbench_files.get(category)
        right_path = official_files.get(category)
        if left_path is None:
            add('missing_category', category=category, side='aisbench',
                official_file=str(right_path))
            continue
        if right_path is None:
            add('missing_category', category=category, side='official',
                aisbench_file=str(left_path))
            continue
        left, right = _load_category(left_path), _load_category(right_path)
        for dimension in dimensions:
            if dimension not in left:
                add('missing_dimension', category=category, dimension=dimension,
                    side='aisbench', official_file=str(right_path))
                continue
            if dimension not in right:
                add('missing_dimension', category=category, dimension=dimension,
                    side='official', aisbench_file=str(left_path))
                continue
            left_samples = _samples(left[dimension], category, dimension)
            right_samples = _samples(right[dimension], category, dimension)
            stats = summary[dimension]
            for sample in sorted(set(left_samples) | set(right_samples)):
                if sample not in right_samples:
                    stats['missing_official'] += 1
                    add('missing_sample', category=category, dimension=dimension,
                        sample=sample, side='official',
                        aisbench=left_samples[sample].get('video_results'))
                    continue
                if sample not in left_samples:
                    stats['missing_aisbench'] += 1
                    add('missing_sample', category=category, dimension=dimension,
                        sample=sample, side='aisbench',
                        official=right_samples[sample].get('video_results'))
                    continue
                official_value = right_samples[sample].get('video_results')
                aisbench_value = left_samples[sample].get('video_results')
                compared += 1
                stats['compared'] += 1
                if not _is_number(official_value) or not _is_number(aisbench_value):
                    stats['mismatched'] += 1
                    add('non_numeric', category=category, dimension=dimension,
                        sample=sample, official=official_value,
                        aisbench=aisbench_value)
                    continue
                official_issue = record_failure(dimension, official_value, right_samples[sample])
                aisbench_issue = record_failure(dimension, aisbench_value, left_samples[sample])
                if official_issue or aisbench_issue:
                    stats['mismatched'] += 1
                    add('invalid_scoring_output', category=category, dimension=dimension,
                        sample=sample, official_issue=official_issue, aisbench_issue=aisbench_issue)
                    continue
                official_value, aisbench_value = float(official_value), float(aisbench_value)
                values[dimension]['official'].append(official_value)
                values[dimension]['aisbench'].append(aisbench_value)
                comparison = _comparison(official_value, aisbench_value,
                                         relative, near_zero, absolute)
                delta, error = comparison['absolute_error'], comparison['relative_error']
                if stats['max_absolute_error'] is None or delta > stats['max_absolute_error']:
                    stats['max_absolute_error'] = delta
                if error is not None and (stats['max_relative_error'] is None
                                          or error > stats['max_relative_error']):
                    stats['max_relative_error'] = error
                mismatched = not comparison['passed']
                if mismatched:
                    stats['mismatched'] += 1
                    add('value_mismatch', category=category, dimension=dimension,
                        sample=sample, official=official_value,
                        aisbench=aisbench_value, absolute_error=delta,
                        relative_error=error,
                        rule='absolute' if error is None else 'relative')
                else:
                    stats['matched'] += 1

    for dimension in dimensions:
        paired = values[dimension]
        stats = summary[dimension]
        stats['valid_pairs'] = len(paired['official'])
        reference = mean(paired['official']) if paired['official'] else None
        candidate = mean(paired['aisbench']) if paired['aisbench'] else None
        comparison = (_comparison(reference, candidate, relative, near_zero, absolute)
                      if reference is not None else dict(
                          passed=False, absolute_error=None, relative_error=None, rule=None))
        stats['mean_comparison'] = dict(comparison, official_mean=reference,
                                        aisbench_mean=candidate)
        if reference is None:
            add('no_valid_pairs', dimension=dimension)
        elif not comparison['passed']:
            add('mean_mismatch', dimension=dimension, official=reference,
                aisbench=candidate, **comparison)
    # A per-sample deviation is diagnostic, not a replacement for the agreed
    # mean-based threshold. Missing/non-numeric samples always block a pass.
    structural = [item for item in differences
                  if item['kind'] not in ('value_mismatch', 'mean_mismatch')]
    means_passed = all(stats['mean_comparison']['passed'] for stats in summary.values())
    sample_values_passed = not structural and not any(
        item['kind'] == 'value_mismatch' for item in differences)
    notes = ['Numeric comparison only: verify input/model provenance and failure states separately.']
    if any(item.get('dimension') == 'second_clap' for item in differences):
        notes.append('second_clap uses official random truncation; repeated runs may differ.')
    return dict(
        passed=not structural and means_passed,
        metric_means_passed=means_passed,
        sample_values_passed=sample_values_passed,
        acceptance='equal_valid_video_raw_means_and_matching_sample_sets',
        aisbench_root=str(aisbench_root),
        official_root=str(official_root),
        categories=categories,
        dimensions=dimensions,
        compared_values=compared,
        difference_count=len(differences),
        differences=differences,
        summary=summary,
        selected_files={category: dict(
            aisbench=str(aisbench_files[category]) if category in aisbench_files else None,
            official=str(official_files[category]) if category in official_files else None)
            for category in categories},
        notes=notes,
        thresholds=dict(relative_strictly_less_than=relative,
                        near_zero_at_most=near_zero, absolute_at_most=absolute),
        compared_field='video_results',
        scope='Native raw numeric output vs AISBench official-format backup; provenance is not verified.',
    )


def _format_number(value):
    if not _is_number(value):
        return repr(value)
    return f'{float(value):.8g}'


def format_official_comparison(result):
    """Human-readable report; all differences remain in the JSON result."""
    lines = [
        f"Official VABench raw-mean comparison: {'PASS' if result['passed'] else 'FAIL'}",
        f"  official: {result['official_root']}",
        f"  aisbench: {result['aisbench_root']}",
        f"  categories: {len(result['categories'])}, dimensions: {len(result['dimensions'])}, "
        f"compared values: {result['compared_values']}",
        f"  differences: {result['difference_count']}",
        f"  per-sample numeric diagnostic: {'PASS' if result['sample_values_passed'] else 'FAIL'}",
    ]
    for dimension, stats in result['summary'].items():
        item = stats['mean_comparison']
        lines.append(f"  {dimension}: {'PASS' if item['passed'] else 'FAIL'} "
                     f"official_mean={_format_number(item['official_mean'])} "
                     f"aisbench_mean={_format_number(item['aisbench_mean'])} "
                     f"valid_pairs={stats['valid_pairs']}")
    if not result['differences']:
        lines.append('  All compared video_results values match within tolerance.')
        lines.extend(f'Note: {note}' for note in result.get('notes', []))
        return '\n'.join(lines)

    lines.append('Differences:')
    for item in result['differences']:
        location = f"{item.get('category', '?')}/{item.get('dimension', '?')}"
        if item['kind'] == 'value_mismatch':
            rel = item.get('relative_error')
            rel_text = 'n/a' if rel is None else f'{rel * 100:.4f}%'
            lines.append(
                f"  [value] {location} / {item['sample']}: "
                f"official={_format_number(item['official'])} "
                f"aisbench={_format_number(item['aisbench'])} "
                f"delta={_format_number(item['absolute_error'])} rel={rel_text}")
        elif item['kind'] == 'missing_sample':
            lines.append(f"  [missing sample] {location} / {item['sample']}: "
                         f"missing in {item['side']}")
        elif item['kind'] == 'missing_dimension':
            lines.append(f"  [missing dimension] {location}: missing in {item['side']}")
        elif item['kind'] == 'missing_category':
            lines.append(f"  [missing category] {item['category']}: missing in {item['side']}")
        elif item['kind'] == 'non_numeric':
            lines.append(
                f"  [non-numeric] {location} / {item['sample']}: "
                f"official={_format_number(item['official'])} "
                f"aisbench={_format_number(item['aisbench'])}")
        else:
            lines.append(f"  [{item['kind']}] {location}: {item}")
    for note in result.get('notes', []):
        lines.append(f'Note: {note}')
    return '\n'.join(lines)
