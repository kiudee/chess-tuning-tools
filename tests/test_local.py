from numpy.testing import assert_almost_equal
from tune.local import parse_experiment_result, reduce_ranges
from skopt.utils import normalize_dimensions


def test_parse_experiment_result():
    teststr = """Indexing opening suite...
    Started game 1 of 100 (lc0 vs sf)
    Finished game 1 (lc0 vs sf): 0-1 {Black mates}
    Score of lc0 vs sf: 0 - 1 - 0  [0.000] 1
    Started game 2 of 100 (sf vs lc0)
    Finished game 2 (sf vs lc0): 0-1 {Black mates}
    Score of lc0 vs sf: 1 - 1 - 0  [0.500] 2
    Elo difference: -31.4 +/- 57.1, LOS: 13.9 %, DrawRatio: 31.0 %
    Finished match
    """
    score, error = parse_experiment_result(teststr, n_dirichlet_samples=1000)
    assert_almost_equal(score, 0.0)
    assert_almost_equal(error, 0.06, decimal=2)


def test_reduce_ranges():
    space = normalize_dimensions([(0.0, 1.0), ("a", "b", "c")])
    x = ((0.0, "a"), (1.01, "a"), (0.5, "d"), (1.0, "c"))
    y = (0.0, 1.0, 2.0, 3.0)
    noise = (0.1, 0.2, 0.3, 0.4)
    reduction_needed, x_new, y_new, noise_new = reduce_ranges(x, y, noise, space)
    assert reduction_needed
    assert tuple(x_new) == ((0.0, "a"), (1.0, "c"))
    assert tuple(y_new) == (0.0, 3.0)
    assert tuple(noise_new) == (0.1, 0.4)
