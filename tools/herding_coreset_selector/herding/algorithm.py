"""Kernel Herding: greedy MMD minimization, RBF kernel."""

import numpy as np
import torch
from typing import Iterator


BANDWIDTH_SAMPLE_CAP = 1000
KERNEL_BATCH = 512
DEFAULT_BANDWIDTH = 1.0


def _rbf_kernel(X, Y, length_scale):
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)
    Y_sq = np.sum(Y ** 2, axis=1, keepdims=True)
    dist_sq = X_sq + Y_sq.T - 2.0 * (X @ Y.T)
    return np.exp(-dist_sq / (2.0 * length_scale ** 2))


def _median_heuristic(data):
    sq = np.sum(data ** 2, axis=1, keepdims=True)
    dist_sq = sq + sq.T - 2.0 * (data @ data.T)
    n = data.shape[0]
    dists = np.sqrt(dist_sq[np.triu_indices(n, k=1)])
    return float(np.median(dists))


def features_to_coreset_matrix(features_generator: Iterator[torch.Tensor]) -> np.ndarray:
    return np.stack([t.cpu().numpy() for t in features_generator], axis=0)


def coreset_indices(data, coreset_size, length_scale=None):
    n = data.shape[0]
    if coreset_size >= n:
        return list(range(n))

    if length_scale is None:
        num_samples = min(n, BANDWIDTH_SAMPLE_CAP)
        rng = np.random.default_rng(42)
        idx = rng.choice(n, num_samples, replace=False)
        length_scale = _median_heuristic(data[idx])
        if length_scale <= 0:
            length_scale = DEFAULT_BANDWIDTH

    # kernel mean
    K_sum = np.zeros(n, dtype=np.float64)
    for i in range(0, n, KERNEL_BATCH):
        end = min(i + KERNEL_BATCH, n)
        K_sum += _rbf_kernel(data[i:end], data, length_scale).sum(axis=0)
    K_mean = K_sum / n

    # greedy selection
    selected = []
    selected_mask = np.zeros(n, dtype=bool)
    K_selected_sum = np.zeros(n, dtype=np.float64)

    for t in range(coreset_size):
        score = K_mean - K_selected_sum / (t + 1)
        score[selected_mask] = -np.inf
        best_idx = int(np.argmax(score))
        selected.append(best_idx)
        selected_mask[best_idx] = True
        K_selected_sum += _rbf_kernel(
            data[best_idx:best_idx + 1], data, length_scale
        ).ravel()

    return selected
