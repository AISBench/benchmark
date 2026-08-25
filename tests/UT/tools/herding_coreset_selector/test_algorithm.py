import numpy as np
import torch

from herding import algorithm


def test_kernel_helpers():
    data = np.array([[0.0], [2.0], [4.0]])
    kernel = algorithm._rbf_kernel(data[:2], data[:2], length_scale=2.0)

    np.testing.assert_allclose(
        kernel,
        [[1.0, np.exp(-0.5)], [np.exp(-0.5), 1.0]],
    )
    assert algorithm._median_heuristic(data) == 2.0


def test_features_to_coreset_matrix():
    features = (torch.tensor(row) for row in ([1.0, 2.0], [3.0, 4.0]))
    np.testing.assert_array_equal(
        algorithm.features_to_coreset_matrix(features),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )


def test_coreset_indices_boundary_and_determinism():
    data = np.array([[0.0], [0.5], [2.0], [8.0], [9.0]])

    assert algorithm.coreset_indices(data, 5) == list(range(5))
    first = algorithm.coreset_indices(data, 3)
    assert first == algorithm.coreset_indices(data, 3)
    assert len(first) == len(set(first)) == 3


def test_coreset_indices_fallback_and_kernel_batches(monkeypatch):
    data = np.ones((5, 2))
    calls = []
    original_kernel = algorithm._rbf_kernel

    def record(left, right, length_scale):
        calls.append(left.shape[0])
        return original_kernel(left, right, length_scale)

    monkeypatch.setattr(algorithm, "KERNEL_BATCH", 2)
    monkeypatch.setattr(algorithm, "_median_heuristic", lambda _: 0.0)
    monkeypatch.setattr(algorithm, "_rbf_kernel", record)

    assert algorithm.coreset_indices(data, 2) == [0, 1]
    assert calls[:3] == [2, 2, 1]
