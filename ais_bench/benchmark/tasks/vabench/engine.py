"""Run official algorithms in isolated, device-specific subprocesses."""
import os
from pathlib import Path
import subprocess
import sys

from ais_bench.benchmark.summarizers.vabench.results import (
    build_report, export_csv, export_official_results, normalize_records)
from ais_bench.benchmark.utils.vabench import DIMENSIONS, MLLM_DIMENSIONS
from ais_bench.benchmark.utils.vabench.data import (
    PACKAGE_ROOT, PROTOCOL_PATH, catalog, digest, object_digest, read_json,
    select_prepared, verify_prepared, write_json)

from .resources import (integration_signature, preflight, source_signature,
                        weight_fingerprints)


def _validate_prepared_split(manifest, source_root):
    """Official splits stay exact; custom is an explicit non-empty 778 subset."""
    standard = {r['idx'] for r in catalog(source_root, 'standard')}
    actual = {r['idx'] for r in manifest['records']}
    scope = manifest.get('evaluation_scope')
    if scope == 'smoke':
        if len(actual) != 1 or not actual <= standard:
            raise ValueError('Smoke run requires exactly one pinned T2AV sample')
        return
    if scope == 'custom':
        if not actual:
            raise ValueError('Custom run requires at least one sample')
        if not actual <= standard:
            raise ValueError('Custom run contains samples outside the pinned 778 catalog')
        return
    if scope not in (None, 'legacy', 'standard') or actual != standard:
        raise ValueError('Prepared sample set must be the pinned 778 split, '
                         'an explicit custom subset, or a single-sample smoke run')


def run(prepared, source_root, cache_dir, output, dimensions=None, device='cuda:0',
        python=None, device_overrides=None, params=None, dim_params=None,
        mode='adapter', timeout=86400, progress=None, test_range=None):
    dimensions = list(DIMENSIONS if dimensions is None else dimensions)
    if not dimensions or len(set(dimensions)) != len(dimensions) or set(dimensions) - set(DIMENSIONS):
        raise ValueError('Unknown, duplicate, or empty VABench dimensions')
    if mode not in ('adapter', 'official'):
        raise ValueError('mode must be adapter or official')
    overrides = dict(device_overrides or {})
    if set(overrides) - set(DIMENSIONS):
        raise ValueError('Unknown device override')
    if any(d in MLLM_DIMENSIONS for d in overrides):
        raise ValueError('MLLM dimensions share a single device; use device=')
    manifest, grouped = verify_prepared(prepared, test_range)
    manifest, grouped, selection = select_prepared(manifest, grouped, test_range)
    _validate_prepared_split(manifest, source_root)
    if selection['applied']:
        print(f"Applied reader_cfg.test_range {test_range!r}: "
              f"{selection['selected']}/{selection['total']} samples", flush=True)
    output = Path(output).resolve()
    if (output / 'report.json').exists():
        raise FileExistsError(f'Choose a new run directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'input_manifest.json', manifest)
    # DNSMOS official ONNX scorer uses CPU. Declare it, including on CUDA runs.
    overrides.setdefault('first_dnsmos', 'cpu')
    jobs = [[d] for d in dimensions if d not in MLLM_DIMENSIONS]
    mllm = [d for d in dimensions if d in MLLM_DIMENSIONS]
    if mllm:
        jobs.append(mllm)
    print('Hashing local weights for provenance (may take several minutes).', flush=True)
    hashes, missing_weights = weight_fingerprints(dimensions, cache_dir)
    provenance = dict(protocol_sha256=digest(PROTOCOL_PATH),
                      source_signature=source_signature(source_root),
                      integration_signature=integration_signature(),
                      weights_signature=object_digest(hashes) if not missing_weights else None,
                      scoring_signature=object_digest(dict(params=params or {}, dim_params=dim_params or {})),
                      weights=hashes, missing_weights=missing_weights,
                      mode=mode, device=device, device_overrides=overrides,
                      scoring_params=params or {}, dim_params=dim_params or {},
                      selection=selection, environments=[])
    results = {}
    for number, dims in enumerate(jobs):
        work = output / f'job_{number:02d}'
        work.mkdir(parents=True, exist_ok=True)
        request = dict(prepared=str(Path(prepared).resolve()),
                       source_root=str(Path(source_root).resolve()),
                       cache_dir=str(Path(cache_dir).resolve()),
                       device=overrides.get(dims[0], device), dimensions=dims,
                       work_dir=str(work), response=str(work / 'response.json'),
                       records=manifest['records'], grouped=grouped,
                       params=params or {}, dim_params=dim_params or {}, mode=mode)
        write_json(work / 'request.json', request)
        missing = preflight(source_root, cache_dir, dims)
        error = None
        response = {}
        if missing:
            error = 'Missing local resources: ' + ', '.join(missing)
        else:
            env = dict(os.environ)
            package_root = str(PACKAGE_ROOT.parent)
            env['PYTHONPATH'] = package_root + os.pathsep + env.get('PYTHONPATH', '')
            env['PYTHONUTF8'] = '1'
            env['PYTHONHASHSEED'] = '42'
            command = [python or sys.executable, '-m',
                       'ais_bench.benchmark.tasks.vabench.worker', str(work / 'request.json')]
            print(f'Running {dims} on {request["device"]}', flush=True)
            try:
                with (work / 'process.log').open('w', encoding='utf-8') as log:
                    proc = subprocess.run(command, env=env, cwd=work, stdout=log,
                                          stderr=subprocess.STDOUT, timeout=timeout)
                response = read_json(work / 'response.json')
                error = response.get('error')
                if proc.returncode and not error:
                    error = f'Worker exited {proc.returncode}; inspect process.log'
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
        if error:
            for dim in dims:
                results[dim] = dict(records=normalize_records(
                    dim, [], manifest['records'], backend_errors=True),
                    errors=[dict(error=error)], requested_device=request['device'],
                    device_observations=[])
        else:
            for dim in dims:
                results[dim] = response['details'][dim]
            provenance['environments'].append(response.get('environment', {}))
        report = build_report(manifest, results, provenance, requested_dimensions=dimensions)
        write_json(output / 'partial_report.json', report)
        if progress is not None:
            # Progress reporting must never affect scoring.
            try:
                progress(number + 1, len(jobs), list(dims))
            except Exception:
                pass
    report = build_report(manifest, results, provenance, requested_dimensions=dimensions)
    write_json(output / 'report.json', report)
    export_csv(report, output)
    export_official_results(report, output, prepared=Path(prepared).resolve(),
                            records=manifest['records'])
    return report


def successful(report, allow_missing=True):
    """Allow official LSA omissions, but require actual valid scoring output.

    Missing non-LSA rows, backend/parse errors and not-evaluated metrics fail.
    A zero-filled preview remains available independently of task success.
    """
    selected = report.get('requested_dimensions', list(report.get('details', {})) or list(report['metrics']))

    def acceptable(dimension, metric):
        scored = metric.get('scored', 0) or 0
        missing = metric.get('missing', 0) or 0
        failed = metric.get('failed', 0) or 0
        not_evaluated = metric.get('not_evaluated', 0) or 0
        expected = metric.get('expected', scored + missing + failed + not_evaluated)
        status = metric.get(
            'status', 'complete' if missing == 0 and scored == expected else 'incomplete')
        if not expected or failed or not_evaluated:
            return False
        if status == 'complete':
            return expected == scored
        if not allow_missing or dimension != 'second_lsa' or status != 'incomplete':
            return False
        # Missing-only incompleteness is skippable; a backend issue is not.
        if metric.get('backend_issue_count'):
            return False
        return missing > 0 and scored + missing == expected

    metrics = report['metrics']
    return (bool(selected) and any(metrics.get(d, {}).get('scored', 0) for d in selected)
            and all(acceptable(d, metrics.get(d, {})) for d in selected))
