from numpy.testing import assert_almost_equal
from skopt.utils import normalize_dimensions

from tune.local import parse_experiment_result, reduce_ranges


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

    # Test cutechess 1.2.0 output:
    teststr = """Started game 1 of 4 (engine1 vs engine2)
    Finished game 1 (engine1 vs engine2): 0-1 {Black mates}
    Score of engine1 vs engine2: 0 - 1 - 0  [0.000] 1
    Started game 2 of 4 (engine2 vs engine1)
    Finished game 2 (engine2 vs engine1): 1/2-1/2 {Draw by stalemate}
    Score of engine1 vs engine2: 0 - 1 - 1  [0.250] 2
    Started game 3 of 4 (engine1 vs engine2)
    Finished game 3 (engine1 vs engine2): 0-1 {Black mates}
    Score of engine1 vs engine2: 0 - 2 - 1  [0.167] 3
    Started game 4 of 4 (engine2 vs engine1)
    Finished game 4 (engine2 vs engine1): 0-1 {Black mates}
    Score of engine1 vs engine2: 1 - 2 - 1  [0.375] 4
    ...      engine1 playing White: 0 - 2 - 0  [0.000] 2
    ...      engine1 playing Black: 1 - 0 - 1  [0.750] 2
    ...      White vs Black: 0 - 3 - 1  [0.125] 4
    Elo difference: -88.7 +/- nan, LOS: 28.2 %, DrawRatio: 25.0 %
    Finished match
    """
    score, error = parse_experiment_result(teststr, n_dirichlet_samples=1000)
    assert_almost_equal(score, 1 / 9)
    assert_almost_equal(error, 0.04, decimal=2)


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
