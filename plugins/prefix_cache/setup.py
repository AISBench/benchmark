from setuptools import find_packages, setup


setup(
    name="ais-bench-prefix-cache",
    version="0.1.0",
    description="Prefix Cache dataset and vLLM runtime plugin for AISBench",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "ais-bench-benchmark",
        "aiohttp",
        "datasets>=2.12.0,<=3.6.0",
        "transformers",
    ],
    entry_points={
        "ais_bench.benchmark_plugins": [
            "prefix_cache = ais_bench_prefix_cache",
        ],
        "console_scripts": [
            "ais-bench-prefix-cache = ais_bench_prefix_cache.cli:console_main",
        ],
    },
)
