"""Discover installed AISBench layout inside the selected image; read-only."""
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil


def discover():
    module = importlib.util.find_spec('ais_bench')
    if module is None or not module.submodule_search_locations:
        raise ValueError('AISBench Python package not installed in image')
    package = Path(next(iter(module.submodule_search_locations))).resolve()
    candidates = []
    for parent in [package, *package.parents]:
        if (parent/'ais_bench/benchmark/configs/models/vllm_api').is_dir() and (parent/'ais_bench/datasets').is_dir():
            candidates.append(str(parent))
    if len(candidates) != 1:
        raise ValueError('Cannot uniquely locate supported AISBench workspace: '+str(candidates))
    executable = shutil.which('ais_bench')
    if not executable:
        raise ValueError('ais_bench command absent from image PATH')
    versions={}
    for name in ['vllm','vllm-ascend','torch','torch-npu']:
        try:
            versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name]=None
    return {'package_versions':versions,'aisbench_work_path': candidates[0], 'executable': executable,
            'version': importlib.metadata.version('ais_bench_benchmark')}


if __name__ == '__main__':
    print(json.dumps(discover()))
