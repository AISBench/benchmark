import json

from ais_bench.benchmark.datasets.omnidocbench.omnidocbench import OmniDocBenchDataset


def test_should_resolve_image_path_when_loading_dataset_given_relative_paths(
    tmp_path, monkeypatch
):
    # Given
    cache_root = tmp_path / "package"
    dataset_root = cache_root / "ais_bench" / "datasets" / "OmniDocBench"
    image_root = dataset_root / "images"
    image_root.mkdir(parents=True)
    (dataset_root / "OmniDocBench.json").write_text(
        json.dumps([{"page_info": {"image_path": "page.png"}}]),
        encoding="utf-8",
    )

    working_dir = tmp_path / "working_dir"
    working_dir.mkdir()
    monkeypatch.setenv("AIS_BENCH_DATASETS_CACHE", str(cache_root))
    monkeypatch.chdir(working_dir)

    # When
    dataset = OmniDocBenchDataset.load(
        "ais_bench/datasets/OmniDocBench/OmniDocBench.json",
        "ais_bench/datasets/OmniDocBench/images",
    )

    # Then
    assert str(image_root / "page.png") in dataset[0]["content"]