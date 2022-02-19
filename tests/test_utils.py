"""Test utility functions of the project."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tune.utils import latest_iterations


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
