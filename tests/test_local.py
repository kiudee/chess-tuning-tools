from numpy.testing import assert_almost_equal
from tune.local import parse_experiment_result


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
