"""Container/tester-local AISBench preflight. Never installs or sends model requests."""
import argparse
import importlib.metadata
from pathlib import Path
import os
import sys
import sysconfig
from . import aisbench_cli
from .durable import atomic_json
from .json_io import load_json
from .protocol import ProtocolError


def inspect_environment(tester,case,*,config_path=None):
    executable=Path(tester['aisbench'])
    python=Path(tester['python'])
    model=Path(tester['model_path'])
    if python.resolve()!=Path(sys.executable).resolve():
        raise ProtocolError('INVALID_INPUT: preflight interpreter differs from binding')
    if (not executable.is_file() or not os.access(executable,os.X_OK) or not model.is_dir() or
            executable.resolve().parent!=Path(sysconfig.get_path('scripts')).resolve()):
        raise ProtocolError('INVALID_INPUT: AISBench entry point or tokenizer assets unavailable in bound environment')
    owners=[]
    for distribution in importlib.metadata.distributions():
        if any(entry.group=='console_scripts' and entry.name==executable.name for entry in distribution.entry_points):
            owners.append(dict(name=distribution.metadata['Name'],version=distribution.version))
    if len(owners)!=1:
        raise ProtocolError('INVALID_INPUT: AISBench console entry point ownership ambiguous or absent')
    data=aisbench_cli.dataset_for({'datasets':tester['datasets']},case,check_files=True)
    if data.get('type')=='aisbench_synthetic':
        dataset_evidence=aisbench_cli.check_synthetic_support(case)
    else:
        aisbench_cli.check_dataset(data,case['formal_requests'])
        dataset_evidence=dict(status='CHECKED',path=data['path'])
    if config_path is not None:
        from mmengine.config import Config
        config=Config.fromfile(str(config_path))
        if not config.get('models') or not config.get('datasets'):
            raise ProtocolError('INVALID_INPUT: native benchmark config incomplete')
    return dict(schema_version='1.0',status='VERIFIED',python=str(Path(sys.executable)),
                executable=str(executable),distributions=owners,dataset=dataset_evidence,
                tokenizer_assets=str(model),scope='entry point, imports, config and dataset preflight')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--request',required=True)
    parser.add_argument('--result',required=True)
    args=parser.parse_args()
    request=load_json(args.request)
    result=inspect_environment(request['tester'],request['case'],config_path=request['config'])
    atomic_json(Path(args.result),result)


if __name__=='__main__':
    main()
