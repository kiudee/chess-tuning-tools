import re
import pathlib
import subprocess

import numpy as np
from scipy.stats import dirichlet
from skopt.space import Categorical, Integer, Real

from tune.utils import TimeControl


__all__ = ["run_match", "parse_experiment_result", "reduce_ranges"]


def parse_experiment_result(
    outstr, prior_counts=None, n_dirichlet_samples=1000000, **kwargs
):
    """Parse cutechess-cli result output to extract mean score and error.

    Here we use a simple pentanomial model to exploit paired openings.
    We distinguish the outcomes WW, WD, WL/DD, LD and LL and apply the
    following scoring (note, that the optimizer always minimizes the score):

    +------+------+-------+-----+-----+
    | WW   | WD   | WL/DD | LD  | LL  |
    +======+======+=======+=====+=====+
    | -1.0 | -0.5 | 0.0   | 0.5 | 1.0 |
    +------+------+-------+-----+-----+

    Note: It is important that the match output was produced using
    cutechess-cli using paired openings, otherwise the returned score is
    useless.

    Parameters
    ----------
    output : string (utf-8)
        Match output of cutechess-cli. It assumes the output was coming from
        a head-to-head match with paired openings.
    prior_counts : list-like float or int, default=None
        Pseudo counts to use for WW, WD, WL/DD, LD and LL in the
        pentanomial model.
    n_dirichlet_samples : int, default = 1 000 000
        Number of samples to draw from the Dirichlet distribution in order to
        estimate the standard error of the score.
    Returns
    -------
    score : float (in [-1, 1])
        Expected (negative) score of the first player (the lower the stronger)
    error : float
        Estimated standard error of the score. Estimated by repeated draws
        from a Dirichlet distribution.
    """
    wdl_strings = re.findall(r"[0-9]+\s-\s[0-9]+\s-\s[0-9]+", outstr)
    array = np.array(
        [np.array([int(y) for y in re.findall(r"[0-9]+", x)]) for x in wdl_strings]
    )
    diffs = np.diff(array, axis=0, prepend=np.array([[0, 0, 0]]))

    counts = {"WW": 0, "WD": 0, "WL/DD": 0, "LD": 0, "LL": 0}
    for i in range(0, len(diffs) - 1, 2):
        match = diffs[i] + diffs[i + 1]
        if match[0] == 2:
            counts["WW"] += 1
        elif match[0] == 1:
            if match[1] == 1:
                counts["WL/DD"] += 1
            else:
                counts["WD"] += 1
        elif match[1] == 1:
            counts["LD"] += 1
        elif match[2] == 2:
            counts["WL/DD"] += 1
        else:
            counts["LL"] += 1
    counts_array = np.array(list(counts.values()))
    if prior_counts is None:
        prior_counts = np.array([0.14, 0.19, 0.34, 0.19, 0.14]) * 2.5
    elif len(prior_counts) != 5:
        raise ValueError("Argument prior_counts should contain 5 elements.")
    dist = dirichlet(alpha=counts_array + prior_counts)
    scores = [-1.0, -0.5, 0.0, 0.5, 1.0]
    score = dist.mean().dot(scores)
    error = dist.rvs(n_dirichlet_samples).dot(scores).var()
    return score, error


def _construct_engine_conf(id, engine_npm=None, engine_tc=None):
    result = ["-engine", f"conf=engine{id}"]
    if engine_npm is not None:
        result.extend(("tc=inf", f"nodes={engine_npm}"))
        return result
    if isinstance(engine_tc, str):
        engine_tc = TimeControl.from_string(engine_tc)
    result.append(f"tc={str(engine_tc)}")
    return result


def run_match(
    rounds=5,
    engine1_tc=None,
    engine2_tc=None,
    engine1_npm=None,
    engine2_npm=None,
    opening_file=None,
    adjudicate_draws=True,
    draw_movenumber=1,
    draw_movecount=10,
    draw_score=8,
    adjudicate_resign=True,
    resign_movecount=3,
    resign_score=550,
    adjudicate_tb=False,
    tb_path=None,
    concurrency=1,
    output_as_string=True,
    **kwargs,
):
    """Run a cutechess-cli match of two engines with paired random openings.

    Parameters
    ----------
    rounds : int, default=1
        Number of rounds to play in the match (each round consists of 2 games).
    engine1_tc : str or TimeControl object, default=None
        Time control to use for the first engine. If str, it can be a
        non-increment time control like "10" (10 seconds) or an increment
        time control like "5+1.5" (5 seconds total with 1.5 seconds increment).
        If None, it is assumed that engine1_npm is provided.
    engine2_tc : str or TimeControl object, default=None
        See engine1_tc.
    engine1_npm : str or int, default=None
        Number of nodes per move the engine is allowed to search.
        If None, it is assumed that engine1_tc is provided.
    engine2_npm : str or int, default=None
        See engine1_npm.
    opening_file : str, default=None
        Path to the file containing the openings. Can be .epd or .pgn.
        Make sure that the file explicitly has the .epd or .pgn suffix, as it
        is used to detect the format.
    adjudicate_draws : bool, default=True
        Specify, if cutechess-cli is allowed to adjudicate draws, if the
        scores of both engines drop below draw_score for draw_movecount number
        of moves. Only kicks in after draw_movenumber moves have been played.
    draw_movenumber : int, default=1
        Number of moves to play after the opening, before draw adjudication is
        allowed.
    draw_movecount : int, default=10
        Number of moves below the threshold draw_score, without captures and
        pawn moves, before the game is adjudicated as draw.
    draw_score : int, default=8
        Score threshold of the engines in centipawns. If the score of both
        engines drops below this value for draw_movecount consecutive moves,
        and there are no captures and pawn moves, the game is adjudicated as
        draw.
    adjudicate_resign : bool, default=True
        Specify, if cutechess-cli is allowed to adjudicate wins/losses based on
        the engine scores. If one engine’s score drops below -resign_score for
        resign_movecount many moves, the game is considered a loss for this
        engine.
    resign_movecount : int, default=3
        Number of consecutive moves one engine has to output a score below
        the resign_score threshold for the game to be considered a loss for this
        engine.
    resign_score : int, default=550
        Resign score threshold in centipawns. The score of the engine has to
        stay below -resign_score for at least resign_movecount moves for it to
        be adjudicated as a loss.
    adjudicate_tb : bool, default=False
        Allow cutechess-cli to adjudicate games based on Syzygy tablebases.
        If true, tb_path has to be set.
    tb_path : str, default=None
        Path to the
    concurrency : int, default=1
        Number of games to run in parallel. Be careful when running time control
        games, since the engines can negatively impact each other when running
        in parallel.
    output_as_string : bool, default=True
        If True, only return the cutechess-cli output as string.
        If False, the complete subprocess.CompletedProcess object will be
        returned for debugging purposes.

    Returns
    -------
    out : str or subprocess.CompletedProcess object
        Results of the cutechess-cli match as string.
        If output_as_string was set to False, returns the
        CompletedProcess object for debugging purposes.
    """
    string_array = ["cutechess-cli"]
    string_array.extend(("-concurrency", str(concurrency)))

    if (engine1_npm is None and engine1_tc is None) or (
        engine2_npm is None and engine2_tc is None
    ):
        raise ValueError("A valid time control or nodes configuration is required.")
    string_array.extend(_construct_engine_conf(1, engine1_npm, engine1_tc))
    string_array.extend(_construct_engine_conf(2, engine2_npm, engine2_tc))

    if opening_file is None:
        raise ValueError("Providing an opening file is required.")
    opening_path = pathlib.Path(opening_file)
    if not opening_path.exists():
        raise FileNotFoundError(
            f"Opening file the following path was not found: {opening_path}"
        )
    opening_format = opening_path.suffix
    if opening_format not in {".epd", ".pgn"}:
        raise ValueError(
            "Unable to determine opening format. "
            "Make sure to add .epd or .pgn to your filename."
        )
    string_array.extend(
        (
            "-openings",
            f"file={str(opening_path)}",
            f"format={opening_format[1:]}",
            "order=random",
        )
    )

    if adjudicate_draws:
        string_array.extend(
            (
                "-draw",
                f"movenumber={draw_movenumber}",
                f"movecount={draw_movecount}",
                f"score={draw_score}",
            )
        )
    if adjudicate_resign:
        string_array.extend(
            ("-resign", f"movecount={resign_movecount}", f"score={resign_score}")
        )
    if adjudicate_tb:
        if tb_path is None:
            raise ValueError("No path to tablebases provided.")
        tb_path_object = pathlib.Path(tb_path)
        if not tb_path_object.exists():
            raise FileNotFoundError(
                f"No folder found at the following path: {str(tb_path_object)}"
            )
        string_array.extend(("-tb", str(tb_path_object)))

    string_array.extend(("-rounds", f"{rounds}"))
    string_array.extend(("-games", "2"))
    string_array.append("-repeat")
    string_array.append("-recover")
    string_array.extend(("-pgnout", "out.pgn"))

    out = subprocess.run(string_array, capture_output=True)
    if output_as_string:
        return out.stdout.decode("utf-8"), out.stderr.decode("utf-8")
    return out


def reduce_ranges(X, y, noise, space):
    X_new = []
    y_new = []
    noise_new = []
    reduction_needed = False
    for row, yval, nval in zip(X, y, noise):
        include_row = True
        for dim, value in zip(space.dimensions, row):
            if isinstance(dim, Integer) or isinstance(dim, Real):
                lb, ub = dim.bounds
                if value < lb or value > ub:
                    include_row = False
            elif isinstance(dim, Categorical):
                if value not in dim.bounds:
                    include_row = False
            else:
                raise ValueError(f"Parameter type {type(dim)} unknown.")
        if include_row:
            X_new.append(row)
            y_new.append(yval)
            noise_new.append(nval)
        else:
            reduction_needed = True
    return reduction_needed, X_new, y_new, noise_new
