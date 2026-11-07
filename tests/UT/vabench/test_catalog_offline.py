"""Pinned-data contract tests with local fixtures, independent of VABench checkout."""
import csv
import json

import pytest

from ais_bench.benchmark.utils.vabench import data


@pytest.fixture
def catalog_fixture(tmp_path, monkeypatch):
    root = tmp_path / 'source'
    (root / 'mapping').mkdir(parents=True)
    (root / 'json_files/json_text').mkdir(parents=True)
    entries = [dict(idx='alpha', category='easy_scene', metadata={'prompt_en': 'A "quoted" sound.'}),
               dict(idx='beta', category='hard_scene', metadata={'prompt_en': 'Another event.'})]
    mapping = root / 'mapping/final_idx_to_prompt.csv'
    with mapping.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['idx', 'prompt'])
        for entry in entries:
            writer.writerow([entry['idx'], json.dumps(entry['metadata']['prompt_en'])[1:-1]])
    for entry in entries:
        data.write_json(root / 'json_files/json_text' / (entry['category'] + '.json'), [entry['metadata']])
    protocol_path = tmp_path / 'protocol.json'
    monkeypatch.setattr(data, 'PROTOCOL_PATH', protocol_path)
    def pin():
        data.write_json(protocol_path, dict(standard_ids=[entry['idx'] for entry in entries],
            data_sha256={path.relative_to(root).as_posix(): data.text_digest(path)
                         for path in root.rglob('*') if path.is_file()}))
    pin()
    return root, entries, mapping, pin


def test_offline_catalog_restores_escaped_prompt_and_exact_scope(catalog_fixture):
    root, entries, _, _ = catalog_fixture
    assert data.catalog(root) == entries
    assert data.catalog(root, 'smoke', sample_id='beta') == [entries[1]]
    with pytest.raises(ValueError, match='sample_id'):
        data.catalog(root, 'smoke', sample_id='unknown')


@pytest.mark.parametrize('duplicate', ['idx', 'prompt'])
def test_offline_catalog_rejects_duplicate_mapping_even_when_pinned(catalog_fixture, duplicate):
    root, entries, mapping, pin = catalog_fixture
    with mapping.open('a', encoding='utf-8', newline='') as stream:
        csv.writer(stream, delimiter='\t').writerow([
            'alpha' if duplicate == 'idx' else 'new',
            'Unique prompt' if duplicate == 'idx' else json.dumps(entries[0]['metadata']['prompt_en'])[1:-1]])
    pin()
    with pytest.raises(ValueError, match='Duplicate idx/prompt'):
        data.catalog(root)


def test_offline_catalog_rejects_metadata_drift_and_missing_selected_item(catalog_fixture):
    root, entries, _, pin = catalog_fixture
    file = root / 'json_files/json_text/hard_scene.json'
    data.write_json(file, [])
    with pytest.raises(ValueError, match='differs from pinned'):
        data.catalog(root)
    pin()
    with pytest.raises(ValueError, match='does not match JSON/mapping'):
        data.catalog(root)


def test_offline_video_catalog_does_not_skip_unknown_or_duplicate_inputs(catalog_fixture, tmp_path):
    root, entries, _, _ = catalog_fixture
    videos = tmp_path / 'videos'; videos.mkdir()
    (videos / 'alpha.mp4').touch()
    assert data.catalog_videos(root, videos) == [entries[0]]
    unknown = videos / 'unknown.mp4'; unknown.touch()
    with pytest.raises(ValueError, match='outside the pinned'):
        data.catalog_videos(root, videos)
    unknown.unlink()
    (videos / 'nested').mkdir()
    (videos / 'nested/alpha.mp4').touch()
    with pytest.raises(ValueError, match='Ambiguous'):
        data.catalog_videos(root, videos)
