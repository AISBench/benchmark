from setuptools import setup, find_packages  # type: ignore

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

packages = find_packages(where=".", exclude=[])

setup(
    name='ais_bench_plugin_example',
    version='0.0.1',
    description='ais_bench_plugin_example',
    long_description=long_description,
    packages=packages,
    include_package_data=True,
    keywords='ais_bench_plugin_example',
    python_requires='>=3.8.0',
    entry_points={
        'ais_bench.benchmark_plugins': [
            'example_plugin = ais_bench_plugin_example_pkg',
        ],
    },
)