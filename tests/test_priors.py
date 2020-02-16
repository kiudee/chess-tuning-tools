import numpy as np
from numpy.testing import assert_almost_equal
from tune.priors import roundflat


def test_roundflat():
    assert_almost_equal(roundflat(0.3), 0.0, decimal=0.1)

    assert roundflat(0.0) == -np.inf
    assert roundflat(-1.0) == -np.inf
