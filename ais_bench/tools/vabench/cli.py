"""Lightweight input preparation, runtime diagnostics and native result comparison.

Usage: python3 -m ais_bench.tools.vabench.cli --help
"""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from ais_bench.benchmark.tasks.vabench.resources import preflight
from ais_bench.benchmark.tasks.vabench.worker import initialize_device
from ais_bench.benchmark.utils.vabench import DIMENSIONS
from ais_bench.benchmark.utils.vabench.data import (
    PROTOCOL_PATH, catalog, catalog_videos, media_tool, prepare, read_json,
    write_json)
from ais_bench.tools.vabench.official_compare import (
    compare_official_results, format_official_comparison)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('catalog', 'prepare', 'doctor'):
        cmd = commands.add_parser(command)
        cmd.add_argument('--source-root', required=True)
        if command == 'catalog':
            cmd.add_argument('--subset', choices=('standard', 'smoke'), default='standard')
            cmd.add_argument('--sample-id', help='Exact pinned idx; required only for --subset smoke')
        if command == 'prepare':
            cmd.add_argument('--subset', choices=('standard', 'smoke', 'custom'), default='standard')
            cmd.add_argument('--sample-id', help='Exact pinned idx; required only for --subset smoke')
        if command in ('catalog', 'prepare'):
            cmd.add_argument('--output', required=True)
        if command == 'prepare':
            cmd.add_argument('--videos', required=True)
        if command == 'doctor':
            cmd.add_argument('--cache-dir', required=True)
            cmd.add_argument('--dimensions', nargs='+', choices=DIMENSIONS, default=list(DIMENSIONS))
            cmd.add_argument('--device', default='cuda:0')
    official_cmp = commands.add_parser(
        'compare-official',
        help='Compare AISBench official-format backups with native VABench output')
    official_cmp.add_argument('--aisbench', required=True,
                              help='Artifacts directory or its official_eval_results directory')
    official_cmp.add_argument('--official', required=True,
                              help='Native evaluate.py --output_dir')
    official_cmp.add_argument('--output', help='Optional JSON comparison report')
    official_cmp.add_argument('--dimensions', nargs='+', choices=DIMENSIONS,
                              default=list(DIMENSIONS))
    official_cmp.add_argument('--relative', type=float, default=0.01)
    official_cmp.add_argument('--near-zero', type=float, default=1e-6)
    official_cmp.add_argument('--absolute', type=float, default=1e-6)
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
                    requested_device=args.device,
                    missing_resources=preflight(args.source_root, args.cache_dir, args.dimensions, strict=True))
        try:
            import torch
            info['torch'] = torch.__version__
            info['cuda_available'] = torch.cuda.is_available()
            if info['modules']['torch_npu']:
                import torch_npu  # noqa: F401
                info['npu_available'] = torch.npu.is_available()
                info['npu_count'] = torch.npu.device_count()
            target = initialize_device(args.device)
            probe = torch.ones(1, device=target)
            probe.cpu()  # Force a tiny operation/copy before declaring the target usable.
            info['device_probe'] = dict(passed=True, observed_device=str(probe.device))
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
        return int(bool(info.get('device_error')) or bool(info['missing_resources']) or not info['ffmpeg'] or not info['ffprobe']
                   or any(not info['modules'][name] for name in required))
    elif args.command == 'compare-official':
        result = compare_official_results(
            args.aisbench, args.official, args.dimensions,
            args.relative, args.near_zero, args.absolute)
        if args.output:
            write_json(args.output, result)
        print(format_official_comparison(result))
        if args.output:
            print(f'Comparison report: {Path(args.output).resolve()}')
        return int(not result['passed'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
