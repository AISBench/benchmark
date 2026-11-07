"""Lazy framework entry point; results/scoring also run in a light backend."""

__all__ = ['VABenchSummarizer']


def __getattr__(name):
    if name == 'VABenchSummarizer':
        from .summarizer import VABenchSummarizer
        return VABenchSummarizer
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
