import dill
import numpy as np
import pytest
from bask import Optimizer
from numpy.testing import assert_almost_equal
from skopt.utils import normalize_dimensions

from tune.local import (
    check_log_for_errors,
    initialize_data,
    initialize_optimizer,
    is_debug_log,
    parse_experiment_result,
    reduce_ranges,
    update_model,
)


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
    score, error, draw_rate = parse_experiment_result(
        teststr, n_dirichlet_samples=1000, random_state=0
    )
    assert_almost_equal(score, 0.0)
    assert_almost_equal(error, 0.887797821633887)
    assert_almost_equal(draw_rate, 1 / 4)

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
    score, error, draw_rate = parse_experiment_result(
        teststr, n_dirichlet_samples=1000, random_state=0
    )
    assert_almost_equal(score, 0.38764005203222596)
    assert_almost_equal(error, 0.6255020676255081)
    assert_almost_equal(draw_rate, 1.5 / 5)

    teststr = """Indexing opening suite...
    Started game 1 of 40 (engine1 vs engine2)
    Finished game 1 (engine1 vs engine2): 1/2-1/2 {Draw by 3-fold repetition}
    Score of engine1 vs engine2: 1 - 0 - 0  [0.500] 1
    Started game 2 of 40 (engine2 vs engine1)
    Finished game 2 (engine2 vs engine1): 1/2-1/2 {Draw by adjudication: SyzygyTB}
    Score of engine1 vs engine2: 2 - 0 - 0  [0.500] 2
    Started game 3 of 40 (engine1 vs engine2)
    Finished game 3 (engine1 vs engine2): 0-1 {Black wins by adjudication}
    Score of engine1 vs engine2: 3 - 0 - 0  [0.333] 3
    Started game 4 of 40 (engine2 vs engine1)
    Finished game 4 (engine2 vs engine1): 0-1 {Black wins by adjudication}
    Score of engine1 vs engine2: 4 - 0 - 0  [0.500] 4
    Started game 5 of 40 (engine1 vs engine2)
    Finished game 5 (engine1 vs engine2): 1-0 {White wins by adjudication}
    Score of engine1 vs engine2: 5 - 0 - 0  [0.600] 5
    Started game 6 of 40 (engine2 vs engine1)
    Finished game 6 (engine2 vs engine1): 1-0 {White wins by adjudication}
    Score of engine1 vs engine2: 6 - 0 - 0  [0.500] 6
    Started game 7 of 40 (engine1 vs engine2)
    Finished game 7 (engine1 vs engine2): 0-1 {Black wins by adjudication}
    Score of engine1 vs engine2: 7 - 0 - 0  [0.429] 7
    Started game 8 of 40 (engine2 vs engine1)
    Finished game 8 (engine2 vs engine1): 0-1 {Black wins by adjudication}
    Score of engine1 vs engine2: 8 - 0 - 0  [0.500] 8
    Started game 9 of 40 (engine1 vs engine2)
    Finished game 9 (engine1 vs engine2): 0-1 {Black wins by adjudication}
    Score of engine1 vs engine2: 9 - 0 - 0  [0.444] 9
    Started game 10 of 40 (engine2 vs engine1)
    Finished game 10 (engine2 vs engine1): 1/2-1/2 {Draw by adjudication}
    Score of engine1 vs engine2: 10 - 0 - 0  [0.450] 10
    """
    score, error, draw_rate = parse_experiment_result(
        teststr, n_dirichlet_samples=1000, random_state=0
    )
    assert_almost_equal(score, -2.7958800173440745)
    assert_almost_equal(error, 1.9952678343378125)
    assert_almost_equal(draw_rate, 1 / 8)

    # Test if the result is correct in case the order of finished games is not linear.
    # This can happen with concurrency > 1
    teststr = """Started game 1 of 4 (engine1 vs engine2)
    Started game 2 of 4 (engine2 vs engine1)
    Started game 3 of 4 (engine1 vs engine2)
    Started game 4 of 4 (engine2 vs engine1)
    Finished game 4 (engine2 vs engine1): 0-1 {Black mates}
    Score of engine1 vs engine2: 1 - 0 - 0  [0.375] 1
    Finished game 1 (engine1 vs engine2): 1/2-1/2 {Draw by stalemate}
    Score of engine1 vs engine2: 1 - 0 - 1  [0.000] 2
    Finished game 2 (engine2 vs engine1): 1-0 {White mates}
    Score of engine1 vs engine2: 1 - 1 - 1  [0.250] 3
    Finished game 3 (engine1 vs engine2): 0-1 {Black mates}
    Score of engine1 vs engine2: 1 - 2 - 1  [0.167] 4
    ...      engine1 playing White: 0 - 2 - 0  [0.000] 2
    ...      engine1 playing Black: 1 - 0 - 1  [0.750] 2
    ...      White vs Black: 0 - 3 - 1  [0.125] 4
    Elo difference: -88.7 +/- nan, LOS: 28.2 %, DrawRatio: 25.0 %
    Finished match
    """
    score, error, draw_rate = parse_experiment_result(
        teststr, n_dirichlet_samples=1000, random_state=0
    )
    assert_almost_equal(score, 0.38764005203222596)
    assert_almost_equal(error, 0.6255020676255081)
    assert_almost_equal(draw_rate, 1.5 / 5)


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


def test_initialize_data(tmp_path):
    # Test basic functionality without resume:
    X, y, noise, iteration, optima, performance = initialize_data(
        parameter_ranges=[(0.0, 1.0)],
        data_path=None,
        resume=False,
    )
    assert len(X) == 0
    assert len(y) == 0
    assert len(noise) == 0
    assert iteration == 0
    assert len(optima) == 0
    assert len(performance) == 0
    # Check if the created data structured are not exactly the same list:
    X.append(0)
    assert len(X) == 1
    assert len(y) == 0
    assert len(noise) == 0

    # Create a temporary file for testing:
    testfile = tmp_path / "data.npz"
    X_in = np.array([[0.0], [0.5], [1.0]])
    y_in = np.array([1.0, -1.0, 0.0])
    noise_in = np.array([0.3, 0.2, 0.5])
    optima_in = np.array([[0.3]])
    performance_in = np.array([[2.0, 30.0, 20.0]])
    iteration_in = np.array([5])
    np.savez_compressed(
        testfile, X_in, y_in, noise_in, optima_in, performance_in, iteration_in
    )

    # Check if resume=False is recognized correctly
    # (outputs should be empty despite data_path being given):
    X, _, _, _, _, _ = initialize_data(
        parameter_ranges=[(0.0, 1.0)],
        data_path=testfile,
        resume=False,
    )
    assert len(X) == 0

    # Check if we get the data back we saved with resume=True:
    X, y, noise, iteration, optima, performance = initialize_data(
        parameter_ranges=[(0.0, 1.0)],
        data_path=testfile,
        resume=True,
    )
    assert int(iteration) == 5
    assert np.allclose(X, X_in)
    assert np.allclose(y, y_in)
    assert np.allclose(noise, noise_in)
    assert np.allclose(optima, optima_in)
    assert np.allclose(performance, performance_in)

    # Check if we get the correct subset, if we reduce the parameter range:
    X, y, noise, iteration, _, _ = initialize_data(
        parameter_ranges=[(0.0, 0.5)],
        data_path=testfile,
        resume=True,
    )
    assert int(iteration) == 5
    assert np.allclose(X, np.array([[0.0], [0.5]]))
    assert np.allclose(y, np.array([1.0, -1.0]))
    assert np.allclose(noise, np.array([0.3, 0.2]))

    # Check if the ValueError is raised correctly:
    with pytest.raises(ValueError):
        _ = initialize_data(
            parameter_ranges=[(0.0, 1.0)] * 2,
            data_path=testfile,
            resume=True,
        )


def test_initialize_optimizer(tmp_path):
    # First test the minimal functionality without data and resume=False
    opt = initialize_optimizer(
        X=[],
        y=[],
        noise=[],
        parameter_ranges=[(0.0, 1.0)],
        resume=False,
    )
    assert len(opt.Xi) == 0

    # Provide some data to test resume functionality, but do not provide a path:
    opt = initialize_optimizer(
        X=[[0.0], [0.5], [1.0]],
        y=[1.0, -1.0, 1.0],
        noise=[0.1, 0.1, 0.1],
        n_initial_points=2,
        gp_initial_burnin=0,
        parameter_ranges=[(0.0, 1.0)],
        resume=True,
        # should only work with a given model_path, so should fall back:
        fast_resume=True,
        model_path=None,
    )
    assert len(opt.Xi) == 3
    assert hasattr(opt.gp, "chain_")

    # Save the optimizer from above to test fast resume:
    model_path = tmp_path / "model.pkl"
    with open(model_path, mode="wb") as f:
        dill.dump(opt, f)
    opt2 = initialize_optimizer(
        X=[[0.0], [0.5], [1.0]],
        y=[1.0, -1.0, 1.0],
        noise=[0.1, 0.1, 0.1],
        n_initial_points=2,
        gp_initial_burnin=0,
        parameter_ranges=[(0.0, 1.0)],
        resume=True,
        fast_resume=True,
        model_path=model_path,
    )
    assert np.allclose(opt2.Xi, opt.Xi)
    assert np.allclose(opt2.yi, opt.yi)
    assert np.allclose(opt2.noisei, opt.noisei)
    # Since fast_resume does not do a refit, these should be equal:
    assert np.allclose(opt2.gp.chain_, opt.gp.chain_)

    # Do a fast resume, but change the ranges. This is expected to raise a ValueError
    # exception, since at this point the data is assumed to be filtered already:
    with pytest.raises(ValueError):
        _ = initialize_optimizer(
            X=[[0.0], [0.5], [1.0]],
            y=[1.0, -1.0, 1.0],
            noise=[0.1, 0.1, 0.1],
            n_initial_points=2,
            gp_initial_burnin=0,
            parameter_ranges=[(0.0, 0.5)],
            resume=True,
            fast_resume=True,
            model_path=model_path,
        )


def test_update_model():
    opt = Optimizer(
        dimensions=[(0.0, 1.0)],
        n_points=10,
        random_state=0,
    )
    points = [[0.0], [1.0], [0.5]]
    scores = [-1.0, 1.0, 0.0]
    variances = [0.3, 0.2, 0.4]
    for p, s, v in zip(points, scores, variances):
        update_model(
            optimizer=opt,
            point=p,
            score=s,
            variance=v,
        )
    assert len(opt.Xi) == 3
    assert np.allclose(opt.Xi, points)
    assert np.allclose(opt.yi, scores)
    assert np.allclose(opt.noisei, variances)


@pytest.mark.parametrize(
    "log_line,expected",
    [
        (
            "3287 <engine1(0): info depth 1 seldepth 1 time 2 nodes 1 score cp -8502 "
            "tbhits 0 pv a7a8n",
            True,
        ),
        ("2018 >lc0(1): ucinewgame", True),
        ("Finished game 1 (engine1 vs engine2): 0-1 {White loses on time}", False),
        ("Finished game 3287 <engine1(0): ", False),
        ("...      White vs Black: 0 - 1 - 0  [0.000] 1", False),
    ],
)
def test_is_debug_log(log_line, expected):
    assert is_debug_log(log_line) == expected


def test_check_log_for_errors(caplog):

    # Test loss on time:
    teststr = """Started game 1 of 2 (SFDev vs Lc0.11198)
    Finished game 1 (SFDev vs Lc0.11198): 1-0 {Black loses on time}
    Score of SFDev vs Lc0.11198: 1 - 0 - 0  [1.000] 1
    Started game 2 of 2 (Lc0.11198 vs SFDev)
    Finished game 2 (Lc0.11198 vs SFDev): 0-1 {White loses on time}
    Score of SFDev vs Lc0.11198: 2 - 0 - 0  [1.000] 2
    Elo difference: inf +/- nan
    Finished match"""

    check_log_for_errors(teststr.split("\n"))
    assert "Engine Lc0.11198 lost on time as Black." in caplog.text

    # Test crash/termination of engine:
    teststr = """Terminating process of engine lc0(0)
    16994 >lc0(1): isready
    16994 <lc0(1): readyok
    ...      lc0 playing White: 0 - 1 - 0  [0.000] 1
    ...      White vs Black: 0 - 1 - 0  [0.000] 1
    Elo difference: -inf +/- nan, LOS: 15.9 %, DrawRatio: 0.0 %
    Finished match
    Finished game 1 (lc0 vs lc0): 0-1 {White's connection stalls}
    Score of lc0 vs lc0: 0 - 1 - 0  [0.000] 1
    ...      lc0 playing White: 0 - 1 - 0  [0.000] 1
    ...      White vs Black: 0 - 1 - 0  [0.000] 1
    Elo difference: -inf +/- nan, LOS: 15.9 %, DrawRatio: 0.0 %
    Finished match
    16995 >lc0(1): quit"""
    check_log_for_errors(teststr.split("\n"))
    assert (
        "lc0's connection stalled as White. Game result is unreliable." in caplog.text
    )

    # Test correct forward of error:
    teststr = "797 <lc0(0): error The cuda backend requires a network file."
    check_log_for_errors([teststr])
    assert (
        "cutechess-cli error: The cuda backend requires a network file." in caplog.text
    )

    # Test unknown UCI option:
    teststr = "424 <lc0(1): error Unknown option: UnknownOption"
    check_log_for_errors([teststr])
    assert (
        "UCI option UnknownOption was unknown to the engine. Check if the spelling "
        "is correct." in caplog.text
    )
