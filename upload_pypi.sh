#!/bin/bash
set -e

if [[ "$1" == "--testpypi-only" ]]; then
    TESTPYPI_ONLY=true
elif [[ "$1" == "--pypi-only" ]]; then
    PYPI_ONLY=true
elif [[ -n "$1" ]]; then
    echo "Unknown option: $1"
    echo "Usage: $0 [--testpypi-only | --pypi-only]"
    exit 1
fi

rm -rf dist/ build/ *.egg-info

echo "Installing build dependencies..."
pip install nltk==3.8 build twine

echo "Building package..."
python -m build --no-isolation --sdist
python -m build --no-isolation --wheel

VERSION=$(python setup.py --version 2>/dev/null || git describe --abbrev=0 --tags | sed 's/^v//')

echo "Version: $VERSION"

if [[ -n "${TESTPYPI_ONLY:-}" ]]; then
    read -p "Publish version $VERSION to TestPyPI only? (y/n) " -n 1 -r
elif [[ -n "${PYPI_ONLY:-}" ]]; then
    read -p "Publish version $VERSION directly to PyPI? (y/n) " -n 1 -r
else
    read -p "Publish version $VERSION to PyPI? (y/n) " -n 1 -r
fi
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

if [[ -z "${PYPI_ONLY:-}" ]]; then
    echo "Uploading to TestPyPI..."
    twine upload --repository testpypi dist/*

    if [[ -n "${TESTPYPI_ONLY:-}" ]]; then
        echo "Done (TestPyPI only)!"
        exit 0
    fi

    read -p "Test upload successful? Publish to PyPI? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Uploading to PyPI..."
twine upload dist/*

echo "Done!"