#!/usr/bin/env python3
"""Validate required benchmark resources in release archives."""

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable, Set


REQUIRED_CMMLU_PATHS = frozenset({
    "ais_bench/benchmark/datasets/cmmlu.py",
    "ais_bench/benchmark/configs/datasets/cmmlu/cmmlu_gen_0_shot_cot_chat_prompt.py",
    "ais_bench/benchmark/configs/datasets/cmmlu/cmmlu_gen_5_shot_cot_chat_prompt.py",
    "ais_bench/benchmark/configs/datasets/cmmlu/cmmlu_ppl_0_shot_cot_chat_prompt.py",
    "ais_bench/benchmark/configs/summarizers/groups/cmmlu.py",
})


class ReleaseArtifactError(RuntimeError):
    """Raised when a release artifact is unsupported or incomplete."""


def _normalize_member_name(name: str) -> str:
    """Remove the versioned top-level directory used by source archives."""
    parts = PurePosixPath(name).parts
    if parts and parts[0].startswith("ais_bench_benchmark-"):
        parts = parts[1:]
    return PurePosixPath(*parts).as_posix()


def _archive_members(artifact: Path) -> Set[str]:
    artifact_name = artifact.name.lower()
    if artifact_name.endswith(".whl"):
        with zipfile.ZipFile(artifact) as archive:
            names: Iterable[str] = archive.namelist()
            return {_normalize_member_name(name) for name in names}
    if artifact_name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(artifact, mode="r:gz") as archive:
            return {
                _normalize_member_name(member.name)
                for member in archive.getmembers()
                if member.isfile()
            }
    raise ReleaseArtifactError(
        f"Unsupported release artifact '{artifact}'; expected a wheel or gzipped sdist"
    )


def verify_release_artifact(artifact: Path) -> None:
    """Ensure a built wheel or sdist contains complete CMMLU support."""
    artifact = Path(artifact)
    if not artifact.is_file():
        raise ReleaseArtifactError(f"Release artifact does not exist: {artifact}")

    missing = sorted(REQUIRED_CMMLU_PATHS - _archive_members(artifact))
    if missing:
        formatted = "\n  - ".join(missing)
        raise ReleaseArtifactError(
            f"Release artifact '{artifact}' is missing CMMLU resources:\n  - {formatted}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that release archives contain required benchmark resources."
    )
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()

    try:
        for artifact in args.artifacts:
            verify_release_artifact(artifact)
            print(f"Validated CMMLU resources in {artifact}")
    except (OSError, tarfile.TarError, zipfile.BadZipFile, ReleaseArtifactError) as exc:
        print(f"Release artifact validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
