"""Lightweight preparation, runtime diagnostics, evaluation and comparison.

Usage: python3 -m ais_bench.tools.vabench.cli --help
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from ais_bench.benchmark.summarizers.vabench_results import (
    compare, export_csv, export_official_results, format_summary, refresh_report)
from ais_bench.benchmark.tasks.vabench.engine import run, successful
from ais_bench.benchmark.tasks.vabench.resources import preflight
from ais_bench.benchmark.utils.vabench import DIMENSIONS
from ais_bench.benchmark.utils.vabench.data import (
    PROTOCOL_PATH, catalog, catalog_videos, digest, media_tool, prepare, read_json,
    write_json)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('catalog', 'prepare', 'doctor', 'run'):
        cmd = commands.add_parser(command)
        cmd.add_argument('--source-root', required=True)
        if command == 'catalog':
            cmd.add_argument('--subset', choices=('standard', 'smoke'), default='standard')
            cmd.add_argument('--sample-id', help='Exact pinned idx; required only for --subset smoke')
        if command == 'prepare':
            cmd.add_argument('--subset', choices=('standard', 'smoke', 'custom'), default='standard')
            cmd.add_argument('--sample-id', help='Exact pinned idx; required only for --subset smoke')
        if command in ('catalog', 'prepare', 'run'):
            cmd.add_argument('--output', required=True)
        if command == 'prepare':
            cmd.add_argument('--videos', required=True)
        if command in ('doctor', 'run'):
            cmd.add_argument('--cache-dir', required=True)
            cmd.add_argument('--dimensions', nargs='+', choices=DIMENSIONS, default=list(DIMENSIONS))
        if command == 'run':
            cmd.add_argument('--prepared', required=True)
            cmd.add_argument('--device', default='cuda:0')
            cmd.add_argument('--python', default=sys.executable)
            cmd.add_argument('--mode', choices=('adapter', 'official'), default='adapter')
            cmd.add_argument('--options', help='JSON: params, dim_params, device_overrides, timeout')
    cmp = commands.add_parser('compare')
    cmp.add_argument('--reference', required=True)
    cmp.add_argument('--candidate', required=True)
    cmp.add_argument('--output', required=True)
    cmp.add_argument('--dimensions', nargs='+', choices=DIMENSIONS, default=list(DIMENSIONS))
    cmp.add_argument('--relative', type=float, default=0.01)
    cmp.add_argument('--near-zero', type=float, default=1e-6)
    cmp.add_argument('--absolute', type=float, default=1e-6)
    summary = commands.add_parser('summarize', help='Reaggregate an AISBench per-sample report without model inference')
    summary.add_argument('--report', required=True)
    summary.add_argument('--output', required=True, help='New output directory; original report is retained')
    summary.add_argument('--prepared', help='Prepared directory used to restore official prompt names')
    args = parser.parse_args(argv)
    if args.command == 'catalog':
        entries = catalog(args.source_root, args.subset, sample_id=args.sample_id)
        protocol = read_json(PROTOCOL_PATH)
        write_json(args.output, dict(subset=args.subset, count=len(entries),
                                    upstream_commit=protocol['commit'], entries=entries))
        print(f'Wrote {len(entries)} prompts to {args.output}')
    elif args.command == 'prepare':
        if args.subset == 'custom':
            if args.sample_id:
                parser.error('--sample-id is only valid with --subset smoke')
            entries = catalog_videos(args.source_root, args.videos)
        else:
            entries = catalog(args.source_root, args.subset, sample_id=args.sample_id)
        result = prepare(entries, args.videos, args.output, subset=args.subset)
        print(f'Prepared {result["subset_size"]} samples at {args.output}')
        if args.subset == 'custom':
            print('Custom subset of the pinned 778; reports are not full 778-set results.')
    elif args.command == 'doctor':
        info = dict(python=sys.version, executable=sys.executable,
                    platform=platform.platform(),
                    ffmpeg=shutil.which(media_tool('ffmpeg')), ffprobe=shutil.which(media_tool('ffprobe')),
                    modules={m: importlib.util.find_spec(m) is not None for m in (
                        'torch', 'torch_npu', 'transformers', 'qwen_omni_utils',
                        'nisqa', 'audiobox_aesthetics', 'onnxruntime', 'decord', 'imagebind')},
                    missing_resources=preflight(args.source_root, args.cache_dir, args.dimensions))
        try:
            import torch
            info['torch'] = torch.__version__
            info['cuda_available'] = torch.cuda.is_available()
            if info['modules']['torch_npu']:
                import torch_npu  # noqa: F401
                info['npu_available'] = torch.npu.is_available()
                info['npu_count'] = torch.npu.device_count()
        except Exception as exc:
            info['device_error'] = str(exc)
        if shutil.which('npu-smi'):
            result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True)
            info['npu_smi'] = result.stdout or result.stderr
        print(json.dumps(info, ensure_ascii=False, indent=2))
        required = {'torch'}
        by_dimension = {'first_dnsmos': ['onnxruntime'], 'first_nisqa': ['nisqa'],
                        'first_audiobox': ['audiobox_aesthetics'],
                        'second_viclip': ['decord'], 'second_imagebind': ['imagebind'],
                        'second_lsa': ['onnxruntime']}
        for dimension in args.dimensions:
            required.update(by_dimension.get(dimension, []))
            if dimension.startswith(('third_', 'fourth_')):
                required.update(['transformers', 'qwen_omni_utils'])
        return int(bool(info['missing_resources']) or not info['ffmpeg'] or not info['ffprobe']
                   or any(not info['modules'][name] for name in required))
    elif args.command == 'run':
        options = read_json(args.options) if args.options else {}
        allowed = {'params', 'dim_params', 'device_overrides', 'timeout'}
        if set(options) - allowed:
            parser.error(f'Unknown options: {set(options) - allowed}')
        result = run(args.prepared, args.source_root, args.cache_dir, args.output,
                     args.dimensions, device=args.device, python=args.python, mode=args.mode, **options)
        print(format_summary(result))
        print(f'Report: {Path(args.output).resolve() / "report.json"}')
        return int(not successful(result))
    elif args.command == 'summarize':
        destination = Path(args.output).resolve()
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f'Choose a new summary directory: {destination}')
        result = refresh_report(read_json(args.report))
        result['postprocessing'] = dict(source_report=str(Path(args.report).resolve()),
                                        source_report_sha256=digest(args.report), model_inference_rerun=False)
        destination.mkdir(parents=True, exist_ok=True)
        write_json(destination / 'report.json', result)
        export_csv(result, destination)
        export_official_results(result, destination, prepared=args.prepared)
        print(format_summary(result))
        print(f'Report: {destination / "report.json"}')
        return 0
    else:
        result = compare(read_json(args.reference), read_json(args.candidate),
                         args.dimensions, args.relative, args.near_zero, args.absolute)
        write_json(args.output, result)
        print('PASS' if result['passed'] else 'FAIL')
        return int(not result['passed'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
