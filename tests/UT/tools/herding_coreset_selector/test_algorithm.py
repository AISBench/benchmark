import numpy as np
import torch

from herding import algorithm


def test_rbf_kernel_has_expected_values():
    data = np.array([[0.0], [2.0]])

    kernel = algorithm._rbf_kernel(data, data, length_scale=2.0)

    expected = np.array([[1.0, np.exp(-0.5)], [np.exp(-0.5), 1.0]])
    np.testing.assert_allclose(kernel, expected)


def test_median_heuristic_uses_pairwise_distances():
    data = np.array([[0.0], [2.0], [4.0]])

    assert algorithm._median_heuristic(data) == 2.0


def test_features_to_coreset_matrix_moves_tensors_to_cpu():
    features = (torch.tensor(values) for values in ([1.0, 2.0], [3.0, 4.0]))

    matrix = algorithm.features_to_coreset_matrix(features)

    np.testing.assert_array_equal(
        matrix,
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )


def test_coreset_indices_returns_all_indices_when_size_covers_dataset():
    data = np.array([[0.0], [1.0], [2.0]])

    assert algorithm.coreset_indices(data, coreset_size=3) == [0, 1, 2]
    assert algorithm.coreset_indices(data, coreset_size=10) == [0, 1, 2]


def test_coreset_indices_are_unique_and_deterministic():
    data = np.array([[0.0], [0.5], [2.0], [8.0], [9.0]])

    first = algorithm.coreset_indices(data, coreset_size=3)
    second = algorithm.coreset_indices(data, coreset_size=3)

    assert first == second
    assert len(first) == 3
    assert len(set(first)) == 3
    assert all(0 <= index < len(data) for index in first)


def test_coreset_indices_falls_back_for_zero_bandwidth(monkeypatch):
    data = np.ones((4, 2))
    monkeypatch.setattr(algorithm, "_median_heuristic", lambda _: 0.0)

    selected = algorithm.coreset_indices(data, coreset_size=2)

    assert selected == [0, 1]


def test_coreset_indices_processes_kernel_in_batches(monkeypatch):
    data = np.arange(10, dtype=float).reshape(5, 2)
    calls = []
    original_rbf_kernel = algorithm._rbf_kernel

    def recording_kernel(left, right, length_scale):
        calls.append(left.shape[0])
        return original_rbf_kernel(left, right, length_scale)

    monkeypatch.setattr(algorithm, "KERNEL_BATCH", 2)
    monkeypatch.setattr(algorithm, "_rbf_kernel", recording_kernel)

    selected = algorithm.coreset_indices(
        data,
        coreset_size=1,
        length_scale=1.0,
    )

    assert len(selected) == 1
    assert calls[:3] == [2, 2, 1]
