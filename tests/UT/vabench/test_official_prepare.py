"""Official input preparation works as a standalone file without GPU packages."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from ais_bench.tools.vabench import official_prepare as prepare


@pytest.fixture
def inputs(tmp_path):
    source, videos, output = (tmp_path / name for name in ('source', 'videos', 'data'))
    source.mkdir()
    (source / 'evaluate.py').touch()
    (source / 'mapping').mkdir()
    (source / 'mapping/final_idx_to_prompt.csv').write_text(
        'idx\tprompt\nkitten curled up meowing\tA kitten meows.\n', encoding='utf-8')
    folder = source / 'json_files/json_text'
    folder.mkdir(parents=True)
    (folder / 'scene.json').write_text(json.dumps([
        dict(prompt_en='A kitten meows.', dimension=['second_clap'])]), encoding='utf-8')
    videos.mkdir()
    (videos / 'kitten curled up meowing.mp4').write_bytes(b'original media')
    return source, videos, output


def _args(inputs):
    source, videos, output = inputs
    return ['--source-root', str(source), '--videos', str(videos), '--output', str(output)]


def test_builds_exact_official_paths_metadata_and_copies_without_reencoding(inputs, monkeypatch, capsys):
    source, videos, output = inputs
    calls = []
    def extract(video, wav):
        calls.append((video, wav))
        wav.write_bytes(b'pcm fixture')
    monkeypatch.setattr(prepare, 'extract_audio', extract)
    prepare.main(_args(inputs))
    video = output / 'video/scene/kitten curled up meowing.mp4'
    audio = output / 'audio/scene/kitten curled up meowing.wav'
    assert video.read_bytes() == b'original media'
    assert audio.read_bytes() == b'pcm fixture'
    rows = json.loads((output / 'json/scene.json').read_text(encoding='utf-8'))
    assert rows[0]['video_list'] == [str(video)] and rows[0]['audio_list'] == [str(audio)]
    assert rows[0]['prompt_en'] == 'A kitten meows.' and len(calls) == 1
    assert 'Total samples: 1' in capsys.readouterr().out


@pytest.mark.parametrize('scenario,message', [
    ('source', 'not a VABench'), ('videos', 'video directory not found'),
    ('output', 'new or empty'), ('empty', 'no MP4'),
    ('duplicate', 'duplicate MP4'), ('unknown', 'not an official VABench idx'),
    ('missing_prompt', 'no official JSON prompt'), ('duplicate_prompt', 'duplicate official prompt'),
])
def test_rejects_invalid_inputs_without_starting_models(inputs, monkeypatch, scenario, message):
    source, videos, output = inputs
    def unexpected(*args):
        pytest.fail('Invalid input should not extract audio')
    monkeypatch.setattr(prepare, 'extract_audio', unexpected)
    if scenario == 'source':
        (source / 'evaluate.py').unlink()
    elif scenario == 'videos':
        videos = videos / 'missing'
    elif scenario == 'output':
        output.mkdir()
        (output / 'existing').touch()
    elif scenario == 'empty':
        (videos / 'kitten curled up meowing.mp4').unlink()
    elif scenario == 'duplicate':
        # Sorted unknown/duplicate inputs must be detected before an extraction.
        folder = videos / 'a'
        folder.mkdir()
        (folder / 'kitten curled up meowing.mp4').touch()
        monkeypatch.setattr(prepare, 'extract_audio', lambda *args: None)
    elif scenario == 'unknown':
        (videos / 'aaa_unknown.mp4').touch()
    elif scenario in ('missing_prompt', 'duplicate_prompt'):
        path = source / 'json_files/json_text/scene.json'
        items = json.loads(path.read_text(encoding='utf-8'))
        path.write_text(json.dumps([] if scenario == 'missing_prompt' else items * 2), encoding='utf-8')
    with pytest.raises(SystemExit, match=message):
        prepare.main(_args((source, videos, output)))


@pytest.mark.parametrize('error,message', [
    (FileNotFoundError(), 'ffmpeg not found'),
    (subprocess.CalledProcessError(1, 'ffmpeg'), 'failed to extract audio'),
])
def test_ffmpeg_failure_is_actionable(inputs, monkeypatch, error, message):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(prepare.subprocess, 'run', fail)
    with pytest.raises(SystemExit, match=message):
        prepare.extract_audio(Path('input.mp4'), Path('audio.wav'))


def test_ffmpeg_uses_first_audio_pcm16_and_explicit_executable(monkeypatch):
    calls = []
    monkeypatch.setenv('VABENCH_FFMPEG', 'my-ffmpeg')
    monkeypatch.setattr(prepare.subprocess, 'run', lambda command, **kw: calls.append((command, kw)))
    prepare.extract_audio(Path('a video.mp4'), Path('audio.wav'))
    command, options = calls[0]
    assert command[0] == 'my-ffmpeg' and options == {'check': True}
    assert command[command.index('-map') + 1] == '0:a:0'
    assert command[command.index('-c:a') + 1] == 'pcm_s16le'
    assert '-ar' not in command and '-ac' not in command


def test_script_can_run_by_absolute_path_outside_the_repository(tmp_path):
    result = subprocess.run([sys.executable, str(Path(prepare.__file__).resolve()), '--help'],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0 and '--source-root' in result.stdout
