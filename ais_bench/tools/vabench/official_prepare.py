#!/usr/bin/env python3
"""Build an official VABench data_dir from MP4 files.

Only uses the official VABench source tree, Python stdlib and ffmpeg.
No AISBench installation or AISBench prepared directory is required.

Video rule: each MP4 stem must exactly equal an idx in
    <source-root>/mapping/final_idx_to_prompt.csv
The category and prompt metadata come from
    <source-root>/json_files/json_text/<category>.json
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def load_mapping(source_root):
    mapping = {}
    path = source_root / "mapping" / "final_idx_to_prompt.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            # The TSV stores JSON-escaped quotes; decode them exactly like
            # the official VABench loader does.
            mapping[row["idx"]] = json.loads('"' + row["prompt"] + '"')
    return mapping


def load_catalog(source_root):
    catalog = {}
    for path in sorted((source_root / "json_files" / "json_text").glob("*.json")):
        category = path.stem
        for item in json.loads(path.read_text(encoding="utf-8")):
            prompt = item["prompt_en"]
            if prompt in catalog:
                raise SystemExit(f"duplicate official prompt: {prompt!r}")
            catalog[prompt] = (category, item)
    return catalog


def extract_audio(video, wav):
    try:
        subprocess.run(
            [
                os.environ.get('VABENCH_FFMPEG', 'ffmpeg'), "-nostdin", "-v", "error", "-y",
                "-i", str(video),
                "-map", "0:a:0", "-vn", "-c:a", "pcm_s16le",
                str(wav),
            ],
            check=True,
        )
    except FileNotFoundError:
        raise SystemExit("ffmpeg not found in PATH")
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"failed to extract audio from {video} "
            f"(does the MP4 contain an audio stream?): {exc}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="path to VABench source")
    parser.add_argument("--videos", required=True, help="directory containing MP4 files")
    parser.add_argument("--output", required=True, help="new official data_dir to create")
    args = parser.parse_args(argv)

    source_root = Path(args.source_root).resolve()
    videos_root = Path(args.videos).resolve()
    output = Path(args.output).resolve()

    if not (source_root / "evaluate.py").is_file():
        raise SystemExit(f"not a VABench source tree: {source_root}")
    if not videos_root.is_dir():
        raise SystemExit(f"video directory not found: {videos_root}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory must be new or empty: {output}")

    mapping = load_mapping(source_root)
    catalog = load_catalog(source_root)

    mp4s = sorted(
        path for path in videos_root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".mp4"
    )
    if not mp4s:
        raise SystemExit(f"no MP4 files found under {videos_root}")

    grouped = {}
    seen = set()
    for video in mp4s:
        idx = video.stem
        if idx in seen:
            raise SystemExit(f"duplicate MP4 stem: {idx!r}")
        seen.add(idx)

        if idx not in mapping:
            raise SystemExit(
                f"MP4 stem is not an official VABench idx: {idx!r}\n"
                "Rename the MP4 so its filename (without .mp4) equals the "
                "idx column in mapping/final_idx_to_prompt.csv."
            )

        prompt = mapping[idx]
        if prompt not in catalog:
            raise SystemExit(f"no official JSON prompt found for idx {idx!r}")

        category, item = catalog[prompt]
        video_dir = output / "video" / category
        audio_dir = output / "audio" / category
        video_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)

        video_target = video_dir / f"{idx}.mp4"
        audio_target = audio_dir / f"{idx}.wav"
        shutil.copy2(video, video_target)
        extract_audio(video, audio_target)

        grouped.setdefault(category, []).append(
            dict(
                item,
                video_list=[str(video_target)],
                audio_list=[str(audio_target)],
            )
        )

    json_dir = output / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    for category, items in grouped.items():
        path = json_dir / f"{category}.json"
        path.write_text(
            json.dumps(items, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{category}: {len(items)} sample(s)")

    print(f"\nOfficial data_dir created: {output}")
    print(f"Total samples: {len(mp4s)}")
    print("\nNext: run evaluate.py from a working directory that contains:")
    print("  mapping_text -> <source-root>/mapping")


if __name__ == "__main__":
    main()
