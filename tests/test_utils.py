"""Test utility functions of the project."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

import tune.utils as utils_module
from tune.utils import expected_ucb, latest_iterations


def test_latest_iterations():
    iterations = np.array([1.0, 2.0, 3.0, 3.0, 4.0])
    expected_indices = [0, 1, 3, 4]
    result = latest_iterations(iterations)
    assert len(result) == 1
    assert_allclose(result, (iterations[expected_indices],))
    array = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    result = latest_iterations(iterations, array)
    assert len(result) == 2
    assert_allclose(result[0], iterations[expected_indices])
    assert_allclose(result[1], array[expected_indices])

    # Test if inconsistent lengths cause an exception
    array = np.array([0.0, 0.1])
    with pytest.raises(ValueError):
        latest_iterations(iterations, array)

    # Test an empty input:
    iterations = np.array([])
    result = latest_iterations(iterations)
    assert len(result) == 1
    assert_allclose(result, (iterations,))


def test_expected_ucb_uses_float64_for_minimize(monkeypatch):
    class DummyRegressor:
        def __init__(self):
            self.seen_dtypes = []

        def predict(self, X, return_std=True):
            self.seen_dtypes.append(X.dtype)
            centered = X.astype(np.float32) - np.float32(0.2)
            mu = (centered**2).sum(axis=1).astype(np.float32)
            std = np.full_like(mu, 0.1, dtype=np.float32)
            return mu, std

    class DummySpace:
        def __init__(self):
            self.bounds = np.array(
                [(np.float32(0.0), np.float32(1.0))], dtype=np.float32
            )

        def transform(self, X):
            return np.asarray(X, dtype=np.float32)

        def inverse_transform(self, X):
            return X.astype(np.float64)

        def rvs(self, n_random_starts, random_state=None):
            rng = np.random.default_rng(random_state)
            samples = rng.uniform(
                0.0, 1.0, size=(n_random_starts, len(self.bounds))
            ).astype(np.float32)
            return samples

    reg = DummyRegressor()
    space = DummySpace()
    res = SimpleNamespace(
        models=[reg], space=space, x=np.array([0.5], dtype=np.float32)
    )

    captured = {}

    original_minimize = utils_module.minimize

    def spy_minimize(func, x0, bounds=None, **kwargs):
        captured["x0_dtype"] = np.asarray(x0).dtype
        captured["bounds"] = bounds
        return original_minimize(func, x0=x0, bounds=bounds, **kwargs)

    monkeypatch.setattr(utils_module, "minimize", spy_minimize)

    x_opt, fun = expected_ucb(res, n_random_starts=0)

    assert isinstance(x_opt, np.ndarray)
    assert x_opt.shape == (1,)
    assert x_opt.dtype == np.float64
    assert np.isfinite(fun)
    assert reg.seen_dtypes, "predict never called"
    assert all(dtype == np.float64 for dtype in reg.seen_dtypes)
    assert captured["x0_dtype"] == np.float64
    assert captured["bounds"] is not None
    assert np.asarray(captured["bounds"], dtype=np.float64).dtype == np.float64
