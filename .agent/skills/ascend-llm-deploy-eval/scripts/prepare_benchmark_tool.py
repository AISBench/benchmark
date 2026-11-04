"""Create an output directory and inventory immutable shared tool sources."""
import argparse
import hashlib
import json
from pathlib import Path

def prepare(source,destination):
    if destination.exists(): raise ValueError('Destination already exists; do not overwrite case evidence')
    source=source.resolve()
    required=['aisbench_test.py','generate_dataset.py','data_picker.py','default_api.py','save_file.py','cal_prefix_hit_rate.py','GSM8K.jsonl','config.py','runtime_config.py','dataset_cache.py','log_format.py']
    manifest={}
    for name in required:
        path=source/name
        if not path.is_file(): raise ValueError('Missing tool source: '+name)
        manifest[name]={'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    destination.mkdir(parents=True)
    (destination/'source-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args(); prepare(args.source,args.destination)
