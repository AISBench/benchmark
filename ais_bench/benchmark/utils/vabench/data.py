"""Pinned T2AV protocol and lossless media preparation (stdlib only)."""
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import tempfile


PROTOCOL_PATH = Path(__file__).with_name('protocol.json')
PACKAGE_ROOT = Path(__file__).resolve().parents[3]


def require_protocol():
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError(
            f'VABench protocol missing: {PROTOCOL_PATH}. Run from the AISBench source '
            'checkout with the VABench integration and install it in editable mode as needed.')
    return PROTOCOL_PATH


def media_tool(name):
    return os.environ.get(f'VABENCH_{name.upper()}', name)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    """Replace atomically: interrupted work must not look complete."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def object_digest(data):
    return hashlib.sha256(json.dumps(
        data, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def text_digest(path):
    return hashlib.sha256(Path(path).read_text(
        encoding='utf-8').encode('utf-8')).hexdigest()


def validate_idx(idx):
    if (not idx or idx in ('.', '..') or '/' in idx or '\\' in idx
            or Path(idx).is_absolute()):
        raise ValueError(f'Unsafe official idx: {idx!r}')


def catalog(source_root, subset='standard', sample_id=None):
    """Read exact upstream prompts, never repair or fuzzy-match them."""
    root = Path(source_root).resolve()
    protocol = read_json(require_protocol())
    if subset not in ('standard', 'smoke'):
        raise ValueError('subset must be standard or smoke')
    if subset == 'smoke':
        if sample_id not in protocol['standard_ids']:
            raise ValueError('smoke requires one sample_id from the pinned 778 IDs')
    elif sample_id is not None:
        raise ValueError('sample_id is only supported with subset=smoke')
    changed = [p for p, sha in protocol['data_sha256'].items()
               if not (root / p).is_file() or text_digest(root / p) != sha]
    if changed:
        raise ValueError(f'Official data differs from pinned protocol: {changed}')
    mapping = {}
    ids = set()
    with (root / 'mapping/final_idx_to_prompt.csv').open(
            encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            prompt = json.loads('"' + row['prompt'] + '"')
            idx = row['idx']
            validate_idx(idx)
            if idx in ids or prompt in mapping:
                raise ValueError('Duplicate idx/prompt in official mapping')
            ids.add(idx)
            mapping[prompt] = idx
    selected = {sample_id} if subset == 'smoke' else set(protocol['standard_ids'])
    entries = []
    for file in sorted((root / 'json_files/json_text').glob('*.json')):
        for item in read_json(file):
            idx = mapping.get(item['prompt_en'])
            if idx in selected:
                entries.append(dict(idx=idx, category=file.stem, metadata=item))
    if len(entries) != len(selected) or {e['idx'] for e in entries} != selected:
        raise ValueError('Pinned subset does not match JSON/mapping')
    return entries


def catalog_videos(source_root, videos_path):
    """Pinned entries for every official MP4 stem found under videos_path.

    Custom runs stay inside the pinned 778 prompts because the official
    metrics need each sample's JSON prompt and auxiliary metadata. Unknown
    videos are rejected instead of being skipped silently.
    """
    entries = catalog(source_root, 'standard')
    by_idx = {entry['idx']: entry for entry in entries}
    videos = Path(videos_path).resolve()
    if not videos.is_dir():
        raise FileNotFoundError(videos)
    by_stem = {}
    for path in videos.rglob('*'):
        if path.is_file() and path.suffix.lower() == '.mp4':
            by_stem.setdefault(path.stem, []).append(path)
    unknown = sorted(set(by_stem) - set(by_idx))
    if unknown:
        raise ValueError(f'Videos outside the pinned 778 catalog: {unknown}')
    ambiguous = sorted(idx for idx, paths in by_stem.items() if len(paths) > 1)
    if ambiguous:
        raise ValueError(f'Ambiguous MP4 stems: {ambiguous}')
    selected = [by_idx[idx] for idx in sorted(by_stem)]
    if not selected:
        raise ValueError('No pinned T2AV videos found in the given directory')
    return selected


def probe(path):
    result = subprocess.run([
        media_tool('ffprobe'), '-v', 'error', '-show_streams', '-show_format',
        '-of', 'json', str(path)], check=True, capture_output=True,
        text=True, encoding='utf-8')
    info = json.loads(result.stdout)
    video = [s for s in info['streams'] if s['codec_type'] == 'video']
    audio = [s for s in info['streams'] if s['codec_type'] == 'audio']
    if not video or not audio:
        raise ValueError(f'MP4 must contain video and audio: {path}')
    duration = float(info.get('format', {}).get('duration', 0))
    if duration <= 0:
        raise ValueError(f'Invalid media duration: {path}')
    return dict(duration=duration, video=video[0], audio=audio[0])


def prepare(entries, videos_path, output_path, subset=None):
    """Strict input set, original MP4 + unresampled PCM audio for experts.

    Missing/ambiguous samples fail preparation, rather than shrinking a mean.
    Subsets are prepared once before parallel evaluation tasks.
    """
    protocol_path = require_protocol()
    if subset not in (None, 'standard', 'smoke', 'custom'):
        raise ValueError('Preparation subset must be standard, smoke or custom')
    if subset == 'smoke' and len(entries) != 1:
        raise ValueError('smoke preparation requires exactly one sample')
    videos = Path(videos_path).resolve()
    output = Path(output_path).resolve()
    if not videos.is_dir():
        raise FileNotFoundError(videos)
    if (output / 'manifest.json').exists():
        raise FileExistsError(f'Choose a new preparation directory: {output}')
    by_stem = {}
    for path in videos.rglob('*'):
        if path.is_file() and path.suffix.lower() == '.mp4':
            by_stem.setdefault(path.stem, []).append(path)
    missing = [e['idx'] for e in entries if not by_stem.get(e['idx'])]
    duplicate = [e['idx'] for e in entries if len(by_stem.get(e['idx'], [])) > 1]
    if missing or duplicate:
        raise ValueError(f'Missing MP4 ({len(missing)}): {missing}; '
                         f'ambiguous MP4 ({len(duplicate)}): {duplicate}')
    grouped, records = {}, []
    for entry in entries:
        idx, category = entry['idx'], entry['category']
        validate_idx(idx)
        path = by_stem[idx][0]
        info = probe(path)
        wav = output / 'audio' / category / (idx + '.wav')
        wav.parent.mkdir(parents=True, exist_ok=True)
        # No trim, channel mixing, resampling, or video re-encode.
        # The official individual metrics perform their own preprocessing.
        subprocess.run([
            media_tool('ffmpeg'), '-nostdin', '-v', 'error', '-y', '-i', str(path),
            '-map', '0:a:0', '-vn', '-c:a', 'pcm_s16le', str(wav)], check=True)
        item = dict(entry['metadata'], video_list=[str(path)], audio_list=[str(wav)])
        grouped.setdefault(category, []).append(item)
        records.append(dict(idx=idx, category=category, video_path=str(path),
                            audio_path=str(wav), video_sha256=digest(path),
                            audio_sha256=digest(wav), duration=info['duration'],
                            short_video=info['duration'] < 5))
    for category, items in grouped.items():
        write_json(output / 'json' / f'{category}.json', items)
    manifest = dict(schema_version=1, subset_size=len(entries),
                    evaluation_scope=subset or 'custom',
                    protocol_sha256=digest(protocol_path), records=records,
                    ffmpeg=subprocess.check_output(
                        [media_tool('ffmpeg'), '-version'], text=True,
                        encoding='utf-8').splitlines()[0])
    manifest['input_signature'] = input_signature(records)
    manifest['metadata_signature'] = object_digest(grouped)
    write_json(output / 'manifest.json', manifest)
    return manifest


def input_signature(records):
    return object_digest(sorted([
        {k: r[k] for k in ('idx', 'category', 'video_sha256', 'audio_sha256')}
        for r in records], key=lambda r: r['idx']))


def select_indices(count, test_range=None):
    """Return AISBench ``reader_cfg.test_range`` indices for a record list.

    ``--num-prompts N`` is converted by the CLI to ``[:N]``.  String ranges use
    the same index-list syntax as ``ICLDatasetReader`` (``[:N]``, ``[2:5]``);
    integer/float ranges retain its deterministic random-selection behavior.
    Invalid ranges keep the full list, matching the CLI warning path.
    """
    indices = list(range(count))
    if test_range is None:
        return indices
    if isinstance(test_range, bool):
        return indices
    if isinstance(test_range, (int, float)):
        if test_range >= count or test_range <= 0:
            return indices
        size = int(test_range * count) if test_range < 1 else int(test_range)
        random.Random(x=size).shuffle(indices)
        return indices[:size]
    if not isinstance(test_range, str):
        return indices
    match = re.fullmatch(r'\s*\[(.*)\]\s*', test_range, re.S)
    if not match:
        return indices
    try:
        node = ast.parse(f'items[{match.group(1)}]', mode='eval').body
    except SyntaxError:
        return indices
    def _bound(value):
        if value is None:
            return None
        if (isinstance(value, ast.Constant)
                and isinstance(value.value, int)
                and not isinstance(value.value, bool)):
            return value.value
        if (isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub)
                and isinstance(value.operand, ast.Constant)
                and isinstance(value.operand.value, int)
                and not isinstance(value.operand.value, bool)):
            return -value.operand.value
        raise ValueError('non-integer slice bound')

    slices = []
    try:
        while isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            part = node.slice
            slices.append(slice(_bound(part.lower), _bound(part.upper), _bound(part.step)))
            node = node.value
        if not isinstance(node, ast.Name) or node.id != 'items':
            return indices
        selected = indices
        for part in reversed(slices):
            selected = selected[part]
        return selected
    except ValueError:
        return indices


def _metadata_ids(entries):
    ids = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for field in ('video_list', 'audio_list'):
            values = entry.get(field) or []
            if isinstance(values, str):
                values = [values]
            ids.update(Path(value).stem for value in values
                       if isinstance(value, (str, os.PathLike)))
    return ids


def select_prepared(manifest, grouped, test_range=None):
    """Restrict an in-memory prepared manifest without touching its files.

    The worker receives one metadata blob per scene, so both the manifest rows
    and each scene's ``video_list``/``audio_list`` entries must be filtered.
    Otherwise official scorers would still process unselected videos and the
    response would contain unexpected samples.
    """
    records = manifest['records']
    indices = select_indices(len(records), test_range)
    # A full-length slice can still reorder samples, e.g. [::-1].
    if indices == list(range(len(records))):
        return manifest, grouped, dict(
            test_range=test_range, selected=len(records),
            total=len(records), applied=False)
    selected = [records[index] for index in indices]
    if not selected:
        raise ValueError('reader_cfg.test_range selected no VABench samples')
    wanted = {}
    for record in selected:
        wanted.setdefault(record['category'], set()).add(record['idx'])
    positions = {record['idx']: position for position, record in enumerate(selected)}
    picked = {}
    # Process scenes in first-selected order, and rows within each scene in
    # selected order. Never mutate the prepared metadata on disk or in memory.
    for category, keep in wanted.items():
        entries = grouped.get(category)
        if not isinstance(entries, list):
            continue
        filtered = [entry for entry in entries
                    if _metadata_ids([entry]) & keep]
        filtered.sort(key=lambda entry: min(
            positions[idx] for idx in _metadata_ids([entry]) & keep))
        if filtered:
            picked[category] = filtered
    covered = set()
    for entries in picked.values():
        covered.update(_metadata_ids(entries))
    missing = {record['idx'] for record in selected} - covered
    if missing:
        raise ValueError('Selected VABench samples missing from prepared metadata: '
                         + ', '.join(sorted(missing)))
    selected_manifest = dict(manifest)
    selected_manifest['records'] = selected
    selected_manifest['subset_size'] = len(selected)
    if (len(selected) < len(records)
            and manifest.get('evaluation_scope') not in ('custom', 'smoke')):
        selected_manifest['evaluation_scope'] = 'custom'
    selected_manifest['input_signature'] = input_signature(selected)
    selected_manifest['metadata_signature'] = object_digest(picked)
    selected_manifest['selection'] = dict(
        test_range=test_range, selected=len(selected), total=len(records),
        source_input_signature=manifest['input_signature'])
    return selected_manifest, picked, dict(
        test_range=test_range, selected=len(selected),
        total=len(records), applied=True)


def verify_prepared(prepared, test_range=None):
    """Recheck immutable inputs before each run (including overwritten WAV).

    Only records selected by ``test_range`` are opened and hashed.  A
    ``--num-prompts`` run must not require media for samples it will not score;
    manifest/metadata signatures still cover the full prepared snapshot.
    """
    protocol_path = require_protocol()
    root = Path(prepared).resolve()
    manifest = read_json(root / 'manifest.json')
    if manifest['protocol_sha256'] != digest(protocol_path):
        raise ValueError('Prepared protocol differs from installed protocol')
    records = manifest['records']
    if len({r['idx'] for r in records}) != len(records):
        raise ValueError('Duplicate samples in preparation manifest')
    selected = [records[index] for index in select_indices(len(records), test_range)]
    for r in selected:
        for media in ('video', 'audio'):
            if digest(r[f'{media}_path']) != r[f'{media}_sha256']:
                raise ValueError(f'Input changed after preparation: {r["idx"]} {media}')
    grouped = {p.stem: read_json(p) for p in sorted((root / 'json').glob('*.json'))}
    if object_digest(grouped) != manifest['metadata_signature']:
        raise ValueError('Prepared metadata changed')
    if input_signature(records) != manifest['input_signature']:
        raise ValueError('Input manifest signature mismatch')
    return manifest, grouped
