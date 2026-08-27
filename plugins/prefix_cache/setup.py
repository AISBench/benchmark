from setuptools import find_packages, setup


setup(
    name="ais-bench-prefix-cache",
    version="0.1.2",
    description="Prefix Cache dataset generation and offline validation for AISBench",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "ais-bench-benchmark",
        "transformers",
    ],
    entry_points={
        "console_scripts": [
            "ais-bench-prefix-cache = ais_bench_prefix_cache.cli:console_main",
        ],
    },
)
