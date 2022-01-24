import logging
import pathlib
import re
import subprocess
import sys
import time
from datetime import datetime
from logging import Logger
from typing import Callable, List, Optional, Sequence, Tuple, Union

import dill
import matplotlib.pyplot as plt
import numpy as np
from bask import Optimizer
from matplotlib.transforms import Bbox
from numpy.random import RandomState
from scipy.optimize import OptimizeResult
from scipy.stats import dirichlet
from skopt.space import Categorical, Dimension, Integer, Real, Space
from skopt.utils import normalize_dimensions

from tune.plots import plot_objective, plot_objective_1d
from tune.summary import confidence_intervals
from tune.utils import TimeControl, confidence_to_mult, expected_ucb

__all__ = [
    "counts_to_penta",
    "initialize_optimizer",
    "run_match",
    "is_debug_log",
    "check_log_for_errors",
    "parse_experiment_result",
    "print_results",
    "plot_results",
    "reduce_ranges",
    "update_model",
    "elo_to_prob",
    "prob_to_elo",
    "setup_logger",
]

LOGGER = "ChessTuner"


def elo_to_prob(elo, k=4.0):
    """Convert an Elo score (logit space) to a probability.

    Parameters
    ----------
    elo : float
        A real-valued Elo score.
    k : float, optional (default=4.0)
        Scale of the logistic distribution.

    Returns
    -------
    float
        Win probability

    Raises
    ------
    ValueError
        if k <= 0

    """
    if k <= 0:
        raise ValueError("k must be positive")
    return 1 / (1 + np.power(10, -elo / k))


def prob_to_elo(p, k=4.0):
    """Convert a win probability to an Elo score (logit space).

    Parameters
    ----------
    p : float
        The win probability of the player.
    k : float, optional (default=4.0)
        Scale of the logistic distribution.

    Returns
    -------
    float
        Elo score of the player

    Raises
    ------
    ValueError
        if k <= 0

    """
    if k <= 0:
        raise ValueError("k must be positive")
    return k * np.log10(-p / (p - 1))


def counts_to_penta(
    counts: np.ndarray,
    prior_counts: Optional[np.ndarray] = None,
    n_dirichlet_samples: int = 1000000,
    score_scale: float = 4.0,
    random_state: Union[int, RandomState, None] = None,
    **kwargs,
) -> Tuple[float, float]:
    """Compute mean Elo score and variance of the pentanomial model for a count array.

    Parameters
    ----------
    counts : np.ndarray
        Array of counts for WW, WD, WL/DD, LD and LL
    prior_counts : np.ndarray or None, default=None
        Pseudo counts to use for WW, WD, WL/DD, LD and LL in the
        pentanomial model.
    n_dirichlet_samples : int, default = 1 000 000
        Number of samples to draw from the Dirichlet distribution in order to
        estimate the standard error of the score.
    score_scale : float, optional (default=4.0)
        Scale of the logistic distribution used to calculate the score. Has to be a
        positive real number
    random_state : int, RandomState instance or None, optional (default: None)
        The generator used to initialize the centers. If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.
    kwargs : dict
        Additional keyword arguments
    Returns
    -------
    tuple (float, float)
        Mean Elo score and corresponding variance
    """
    if prior_counts is None:
        prior_counts = np.array([0.14, 0.19, 0.34, 0.19, 0.14]) * 2.5
    elif len(prior_counts) != 5:
        raise ValueError("Argument prior_counts should contain 5 elements.")
    dist = dirichlet(alpha=counts + prior_counts)
    scores = [0.0, 0.25, 0.5, 0.75, 1.0]
    score = prob_to_elo(dist.mean().dot(scores), k=score_scale)
    error = prob_to_elo(
        dist.rvs(n_dirichlet_samples, random_state=random_state).dot(scores),
        k=score_scale,
    ).var()
    return score, error


def setup_logger(verbose: int = 0, logfile: str = "log.txt") -> Logger:
    """Setup logger with correct verbosity and file handler.

    Parameters
    ----------
    verbose : int
        Verbosity level. If verbose = 0, use INFO level, otherwise DEBUG.
    logfile : str
        Desired path to the logfile.

    Returns
    -------
    Logger
        Logger to be used for logging.
    """
    log_level = logging.DEBUG if verbose > 0 else logging.INFO
    log_format = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    logger = logging.getLogger(LOGGER)
    logger.setLevel(log_level)
    logger.propagate = False

    file_logger = logging.FileHandler(logfile)
    file_logger.setFormatter(log_format)
    logger.addHandler(file_logger)
    console_logger = logging.StreamHandler(sys.stdout)
    console_logger.setFormatter(log_format)
    logger.addHandler(console_logger)
    return logger


def reduce_ranges(
    X: Sequence[list], y: Sequence[float], noise: Sequence[float], space: Space
) -> Tuple[bool, List[list], List[float], List[float]]:
    """Return all data points consistent with the new restricted space.

    Parameters
    ----------
    X : Sequence of lists
        Contains n_points many lists, each representing one configuration.
    y : Sequence of floats
        Contains n_points many scores, one for each configuration.
    noise : Sequence of floats
        Contains n_points many variances, one for each score.
    space : skopt.space.Space
        Space object specifying the new optimization space.

    Returns
    -------
    Tuple (bool, list, list, list)
        Returns a boolean indicating if a reduction of the dataset was needed and the
        corresponding new X, y and noise lists.
    """
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


def initialize_data(
    parameter_ranges: Sequence[Union[Sequence, Dimension]],
    data_path: Optional[str] = None,
    resume: bool = True,
) -> Tuple[list, list, list, int]:
    """Initialize data structures needed for tuning. Either empty or resumed from disk.

    Parameters
    ----------
    parameter_ranges : Sequence of Dimension objects or tuples
        Parameter range specifications as expected by scikit-optimize.
    data_path : str or None, default=None
        Path to the file containing the data structures used for resuming.
        If None, no resuming will be performed.
    resume : bool, default=True
        If True, fill the data structures with the the data from the given data_path.
        Otherwise return empty data structures.

    Returns
    -------
    tuple consisting of list, list, list and int
        Returns the initialized data structures X, y, noise and iteration number.

    Raises
    ------
    ValueError
        If the number of specified parameters is not matching the existing number of
        parameters in the data.
    """
    logger = logging.getLogger()
    X = []
    y = []
    noise = []
    iteration = 0
    if data_path is not None and resume:
        space = normalize_dimensions(parameter_ranges)
        path = pathlib.Path(data_path)
        if path.exists():
            with np.load(path) as importa:
                X = importa["arr_0"].tolist()
                y = importa["arr_1"].tolist()
                noise = importa["arr_2"].tolist()
            if len(X[0]) != space.n_dims:
                raise ValueError(
                    f"Number of parameters ({len(X[0])}) are not matching "
                    f"the number of dimensions ({space.n_dims})."
                )
            reduction_needed, X_reduced, y_reduced, noise_reduced = reduce_ranges(
                X, y, noise, space
            )
            if reduction_needed:
                backup_path = path.parent / (
                    path.stem + f"_backup_{int(time.time())}" + path.suffix
                )
                logger.warning(
                    f"The parameter ranges are smaller than the existing data. "
                    f"Some points will have to be discarded. "
                    f"The original {len(X)} data points will be saved to "
                    f"{backup_path}"
                )
                np.savez_compressed(
                    backup_path, np.array(X), np.array(y), np.array(noise)
                )
                X = X_reduced
                y = y_reduced
                noise = noise_reduced
            iteration = len(X)
    return X, y, noise, iteration


def setup_random_state(seed: int) -> np.random.RandomState:
    """Return a seeded RandomState object.

    Parameters
    ----------
    seed : int
        Random seed to be used to seed the RandomState.

    Returns
    -------
    numpy.random.RandomState
        RandomState to be used to generate random numbers.
    """
    ss = np.random.SeedSequence(seed)
    return np.random.RandomState(np.random.MT19937(ss.spawn(1)[0]))


def initialize_optimizer(
    X: Sequence[list],
    y: Sequence[float],
    noise: Sequence[float],
    parameter_ranges: Sequence[Union[Sequence, Dimension]],
    random_seed: int = 0,
    warp_inputs: bool = True,
    n_points: int = 500,
    n_initial_points: int = 16,
    acq_function: str = "mes",
    acq_function_samples: int = 1,
    resume: bool = True,
    fast_resume: bool = True,
    model_path: Optional[str] = None,
    gp_initial_burnin: int = 100,
    gp_initial_samples: int = 300,
    gp_priors: Optional[List[Callable[[float], float]]] = None,
) -> Optimizer:
    """Create an Optimizer object and if needed resume and/or reinitialize.

    Parameters
    ----------
    X : Sequence of lists
        Contains n_points many lists, each representing one configuration.
    y : Sequence of floats
        Contains n_points many scores, one for each configuration.
    noise : Sequence of floats
        Contains n_points many variances, one for each score.
    parameter_ranges : Sequence of Dimension objects or tuples
        Parameter range specifications as expected by scikit-optimize.
    random_seed : int, default=0
        Random seed for the optimizer.
    warp_inputs : bool, default=True
        If True, the optimizer will internally warp the input space for a better model
        fit. Can negatively impact running time and required burnin samples.
    n_points : int, default=500
        Number of points to evaluate the acquisition function on.
    n_initial_points : int, default=16
        Number of points to pick quasi-randomly to initialize the the model, before
        using the acquisition function.
    acq_function : str, default="mes"
        Acquisition function to use.
    acq_function_samples : int, default=1
        Number of hyperposterior samples to average the acquisition function over.
    resume : bool, default=True
        If True, resume optimization from existing data. If False, start with a
        completely fresh optimizer.
    fast_resume : bool, default=True
        If True, restore the optimizer from disk, avoiding costly reinitialization.
        If False, reinitialize the optimizer from the existing data.
    model_path : str or None, default=None
        Path to the file containing the existing optimizer to be used for fast resume
        functionality.
    gp_initial_burnin : int, default=100
        Number of burnin samples to use for reinitialization.
    gp_initial_samples : int, default=300
        Number of samples to use for reinitialization.
    gp_priors : list of callables, default=None
        List of priors to be used for the kernel hyperparameters. Specified in the
        following order:
        - signal magnitude prior
        - lengthscale prior (x number of parameters)
        - noise magnitude prior

    Returns
    -------
    bask.Optimizer
        Optimizer object to be used in the main tuning loop.
    """
    logger = logging.getLogger(LOGGER)
    # Create random generator:
    random_state = setup_random_state(random_seed)

    gp_kwargs = dict(
        # TODO: Due to a bug in scikit-learn 0.23.2, we set normalize_y=False:
        normalize_y=True,
        warp_inputs=warp_inputs,
    )
    opt = Optimizer(
        dimensions=parameter_ranges,
        n_points=n_points,
        n_initial_points=n_initial_points,
        # gp_kernel=kernel,  # TODO: Let user pass in different kernels
        gp_kwargs=gp_kwargs,
        gp_priors=gp_priors,
        acq_func=acq_function,
        acq_func_kwargs=dict(alpha=1.96, n_thompson=500),
        random_state=random_state,
    )

    if not resume:
        return opt

    reinitialize = True
    if model_path is not None and fast_resume:
        path = pathlib.Path(model_path)
        if path.exists():
            with open(model_path, mode="rb") as model_file:
                old_opt = dill.load(model_file)
                logger.info(f"Resuming from existing optimizer in {model_path}.")
            if opt.space == old_opt.space:
                old_opt.acq_func = opt.acq_func
                old_opt.acq_func_kwargs = opt.acq_func_kwargs
                opt = old_opt
                reinitialize = False
            else:
                logger.info(
                    "Parameter ranges have been changed and the "
                    "existing optimizer instance is no longer "
                    "valid. Reinitializing now."
                )
            if gp_priors is not None:
                opt.gp_priors = gp_priors

    if reinitialize and len(X) > 0:
        logger.info(
            f"Importing {len(X)} existing datapoints. " f"This could take a while..."
        )
        opt.tell(
            X,
            y,
            noise_vector=noise,
            gp_burnin=gp_initial_burnin,
            gp_samples=gp_initial_samples,
            n_samples=acq_function_samples,
            progress=True,
        )
        logger.info("Importing finished.")
    return opt


def print_results(
    optimizer: Optimizer,
    result_object: OptimizeResult,
    parameter_names: Sequence[str],
    confidence: float = 0.9,
) -> None:
    """ Log the current results of the optimizer.

    Parameters
    ----------
    optimizer : bask.Optimizer
        Fitted Optimizer object.
    result_object : scipy.optimize.OptimizeResult
        Result object containing the data and the last fitted model.
    parameter_names : Sequence of str
        Names of the parameters to use for printing.
    confidence : float, default=0.9
        Confidence used for the confidence intervals.
    """
    logger = logging.getLogger(LOGGER)
    try:
        best_point, best_value = expected_ucb(result_object, alpha=0.0)
        best_point_dict = dict(zip(parameter_names, best_point))
        with optimizer.gp.noise_set_to_zero():
            _, best_std = optimizer.gp.predict(
                optimizer.space.transform([best_point]), return_std=True
            )
        logger.info(f"Current optimum:\n{best_point_dict}")
        logger.info(
            f"Estimated Elo: {np.around(-best_value * 100, 4)} +- "
            f"{np.around(best_std * 100, 4).item()}"
        )
        confidence_mult = confidence_to_mult(confidence)
        lower_bound = np.around(
            -best_value * 100 - confidence_mult * best_std * 100, 4
        ).item()
        upper_bound = np.around(
            -best_value * 100 + confidence_mult * best_std * 100, 4
        ).item()
        logger.info(
            f"{confidence * 100}% confidence interval of the Elo value: "
            f"({lower_bound}, "
            f"{upper_bound})"
        )
        confidence_out = confidence_intervals(
            optimizer=optimizer,
            param_names=parameter_names,
            hdi_prob=confidence,
            opt_samples=1000,
            multimodal=False,
        )
        logger.info(
            f"{confidence * 100}% confidence intervals of the parameters:"
            f"\n{confidence_out}"
        )
    except ValueError:
        logger.info(
            "Computing current optimum was not successful. "
            "This can happen in rare cases and running the "
            "tuner again usually works."
        )


def plot_results(
    optimizer: Optimizer,
    result_object: OptimizeResult,
    plot_path: str,
    parameter_names: Sequence[str],
    confidence: float = 0.9,
) -> None:
    """Plot the current results of the optimizer.

    Parameters
    ----------
    optimizer : bask.Optimizer
        Fitted Optimizer object.
    result_object : scipy.optimize.OptimizeResult
        Result object containing the data and the last fitted model.
    plot_path : str
        Path to the directory to which the plots should be saved.
    parameter_names : Sequence of str
        Names of the parameters to use for plotting.
    confidence : float
        The confidence level of the normal distribution to plot in the 1d plot.
    """
    logger = logging.getLogger(LOGGER)
    logger.debug("Starting to compute the next plot.")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    dark_gray = "#36393f"
    save_params = dict()
    if optimizer.space.n_dims == 1:
        fig, ax = plot_objective_1d(
            result=result_object,
            parameter_name=parameter_names[0],
            confidence=confidence,
        )
        save_params["bbox_inches"] = Bbox([[0.5, -0.2], [9.25, 5.5]])
    else:
        plt.style.use("dark_background")
        fig, ax = plt.subplots(
            nrows=optimizer.space.n_dims,
            ncols=optimizer.space.n_dims,
            figsize=(3 * optimizer.space.n_dims, 3 * optimizer.space.n_dims),
        )
        for i in range(optimizer.space.n_dims):
            for j in range(optimizer.space.n_dims):
                ax[i, j].set_facecolor(dark_gray)
        fig.patch.set_facecolor(dark_gray)
        plot_objective(result_object, dimensions=parameter_names, fig=fig, ax=ax)
    plotpath = pathlib.Path(plot_path)
    plotpath.mkdir(parents=True, exist_ok=True)
    full_plotpath = plotpath / f"{timestr}-{len(optimizer.Xi)}.png"
    dpi = 150 if optimizer.space.n_dims == 1 else 300
    plt.savefig(full_plotpath, dpi=dpi, facecolor=dark_gray, **save_params)
    logger.info(f"Saving a plot to {full_plotpath}.")
    plt.close(fig)


def run_match(
    rounds=10,
    engine1_tc=None,
    engine2_tc=None,
    engine1_st=None,
    engine2_st=None,
    engine1_npm=None,
    engine2_npm=None,
    engine1_ponder=False,
    engine2_ponder=False,
    timemargin=None,
    opening_file=None,
    adjudicate_draws=False,
    draw_movenumber=1,
    draw_movecount=10,
    draw_score=8,
    adjudicate_resign=False,
    resign_movecount=3,
    resign_score=550,
    adjudicate_tb=False,
    tb_path=None,
    concurrency=1,
    debug_mode=False,
    **kwargs,
):
    """Run a cutechess-cli match of two engines with paired random openings.

    Parameters
    ----------
    rounds : int, default=10
        Number of rounds to play in the match (each round consists of 2 games).
    engine1_tc : str or TimeControl object, default=None
        Time control to use for the first engine. If str, it can be a
        non-increment time control like "10" (10 seconds) or an increment
        time control like "5+1.5" (5 seconds total with 1.5 seconds increment).
        If None, it is assumed that engine1_npm or engine1_st is provided.
    engine2_tc : str or TimeControl object, default=None
        See engine1_tc.
    engine1_st : str or int, default=None
        Time limit in seconds for each move.
        If None, it is assumed that engine1_tc or engine1_npm is provided.
    engine2_st : str or TimeControl object, default=None
        See engine1_tc.
    engine1_npm : str or int, default=None
        Number of nodes per move the engine is allowed to search.
        If None, it is assumed that engine1_tc or engine1_st is provided.
    engine2_npm : str or int, default=None
        See engine1_npm.
    engine1_ponder : bool, default=False
        If True, allow engine1 to ponder.
    engine2_ponder : bool, default=False
        See engine1_ponder.
    timemargin : str or int, default=None
        Allowed number of milliseconds the engines are allowed to go over the time
        limit. If None, the margin is 0.
    opening_file : str, default=None
        Path to the file containing the openings. Can be .epd or .pgn.
        Make sure that the file explicitly has the .epd or .pgn suffix, as it
        is used to detect the format.
    adjudicate_draws : bool, default=False
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
    adjudicate_resign : bool, default=False
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
        Path to the folder containing the Syzygy tablebases.
    concurrency : int, default=1
        Number of games to run in parallel. Be careful when running time control
        games, since the engines can negatively impact each other when running
        in parallel.

    Yields
    -------
    out : str
        Results of the cutechess-cli match streamed as str.
    """
    string_array = ["cutechess-cli"]
    string_array.extend(("-concurrency", str(concurrency)))

    if (engine1_npm is None and engine1_tc is None and engine1_st is None) or (
        engine2_npm is None and engine2_tc is None and engine2_st is None
    ):
        raise ValueError("A valid time control or nodes configuration is required.")
    string_array.extend(
        _construct_engine_conf(
            id=1,
            engine_npm=engine1_npm,
            engine_tc=engine1_tc,
            engine_st=engine1_st,
            engine_ponder=engine1_ponder,
            timemargin=timemargin,
        )
    )
    string_array.extend(
        _construct_engine_conf(
            id=2,
            engine_npm=engine2_npm,
            engine_tc=engine2_tc,
            engine_st=engine2_st,
            engine_ponder=engine2_ponder,
            timemargin=timemargin,
        )
    )

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
    string_array.append("-debug")
    string_array.extend(("-pgnout", "out.pgn"))

    with subprocess.Popen(
        string_array, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as popen:
        for line in iter(popen.stdout.readline, ""):
            yield line


def is_debug_log(cutechess_line: str,) -> bool:
    """Check if the provided cutechess log line is a debug mode line.

    Parameters
    ----------
    cutechess_line : str
        One line from a cutechess log.

    Returns
    -------
    bool
        True, if the given line is a debug mode line, False otherwise.
    """
    if re.match(r"[0-9]+ [<>]", cutechess_line) is not None:
        return True
    return False


def check_log_for_errors(cutechess_output: List[str],) -> None:
    """Parse the log output produced by cutechess-cli and scan for important errors.

    Parameters
    ----------
    cutechess_output : list of str
        String containing the log output produced by cutechess-cli.
    """
    logger = logging.getLogger(LOGGER)
    for line in cutechess_output:

        # Check for forwarded errors:
        pattern = r"[0-9]+ [<>].+: error (.+)"
        match = re.search(pattern=pattern, string=line)
        if match is not None:
            logger.warning(f"cutechess-cli error: {match.group(1)}")

        # Check for unknown UCI option
        pattern = r"Unknown (?:option|command): (.+)"
        match = re.search(pattern=pattern, string=line)
        if match is not None:
            logger.error(
                f"UCI option {match.group(1)} was unknown to the engine. "
                f"Check if the spelling is correct."
            )
            continue

        # Check for loss on time
        pattern = (
            r"Finished game [0-9]+ \((.+) vs (.+)\): [0-9]-[0-9] {(\S+) "
            r"(?:loses on time)}"
        )
        match = re.search(pattern=pattern, string=line)
        if match is not None:
            engine = match.group(1) if match.group(3) == "White" else match.group(2)
            logger.warning(f"Engine {engine} lost on time as {match.group(3)}.")
            continue

        # Check for connection stall:
        pattern = (
            r"Finished game [0-9]+ \((.+) vs (.+)\): [0-9]-[0-9] {(\S+)'s "
            r"(?:connection stalls)}"
        )
        match = re.search(pattern=pattern, string=line)
        if match is not None:
            engine = match.group(1) if match.group(3) == "White" else match.group(2)
            logger.error(
                f"{engine}'s connection stalled as {match.group(3)}. "
                f"Game result is unreliable."
            )


def parse_experiment_result(
    outstr,
    prior_counts=None,
    n_dirichlet_samples=1000000,
    score_scale=4.0,
    random_state=None,
    **kwargs,
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
    score_scale : float, optional (default=4.0)
        Scale of the logistic distribution used to calculate the score. Has to be a
        positive real number
    random_state : int, RandomState instance or None, optional (default: None)
        The generator used to initialize the centers. If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.
    Returns
    -------
    score : float (in [-1, 1])
        Expected (negative) score of the first player (the lower the stronger)
    error : float
        Estimated standard error of the score. Estimated by repeated draws
        from a Dirichlet distribution.
    """
    wdl_strings = re.findall(r"Score of.*:\s*([0-9]+\s-\s[0-9]+\s-\s[0-9]+)", outstr)
    array = np.array(
        [np.array([int(y) for y in re.findall(r"[0-9]+", x)]) for x in wdl_strings]
    )
    diffs = np.diff(array, axis=0, prepend=np.array([[0, 0, 0]]))

    # Parse order of finished games to be able to compute the correct pentanomial scores
    finished = np.array(
        [int(x) - 1 for x in re.findall(r"Finished game ([0-9]+)", outstr)]
    )
    diffs = diffs[np.argsort(finished)]

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
    return counts_to_penta(
        counts=counts_array,
        prior_counts=prior_counts,
        n_dirichlet_samples=n_dirichlet_samples,
        score_scale=score_scale,
        random_state=random_state,
        **kwargs,
    )


def update_model(
    optimizer: Optimizer,
    point: list,
    score: float,
    variance: float,
    acq_function_samples: int = 1,
    gp_burnin: int = 5,
    gp_samples: int = 300,
    gp_initial_burnin: int = 100,
    gp_initial_samples: int = 300,
) -> None:
    """Update the optimizer model with the newest data.

    Parameters
    ----------
    optimizer : bask.Optimizer
        Optimizer object which is to be updated.
    point : list
        Latest configuration which was tested.
    score : float
        Elo score the configuration achieved.
    variance : float
        Variance of the Elo score of the configuration.
    acq_function_samples : int, default=1
        Number of hyperposterior samples to average the acquisition function over.
    gp_burnin : int, default=5
        Number of burnin iterations to use before keeping samples for the model.
    gp_samples : int, default=300
        Number of samples to collect for the model.
    gp_initial_burnin : int, default=100
        Number of burnin iterations to use for the first initial model fit.
    gp_initial_samples : int, default=300
        Number of samples to collect
    """
    logger = logging.getLogger(LOGGER)
    while True:
        try:
            now = datetime.now()
            # We fetch kwargs manually here to avoid collisions:
            n_samples = acq_function_samples
            gp_burnin = gp_burnin
            gp_samples = gp_samples
            if optimizer.gp.chain_ is None:
                gp_burnin = gp_initial_burnin
                gp_samples = gp_initial_samples
            optimizer.tell(
                x=point,
                y=score,
                noise_vector=variance,
                n_samples=n_samples,
                gp_samples=gp_samples,
                gp_burnin=gp_burnin,
            )
            later = datetime.now()
            difference = (later - now).total_seconds()
            logger.info(f"GP sampling finished ({difference}s)")
            logger.debug(f"GP kernel: {optimizer.gp.kernel_}")
        except ValueError:
            logger.warning(
                "Error encountered during fitting. Trying to sample chain a bit. "
                "If this problem persists, restart the tuner to reinitialize."
            )
            optimizer.gp.sample(n_burnin=11, priors=optimizer.gp_priors)
        else:
            break


def _construct_engine_conf(
    id,
    engine_npm=None,
    engine_tc=None,
    engine_st=None,
    engine_ponder=False,
    timemargin=None,
):
    result = ["-engine", f"conf=engine{id}"]
    if engine_npm is not None:
        result.extend(("tc=inf", f"nodes={engine_npm}"))
        return result
    if engine_st is not None:
        result.append(f"st={str(engine_st)}")
        if timemargin is not None:
            result.append(f"timemargin={str(timemargin)}")
        if engine_ponder:
            result.append("ponder")
        return result
    if isinstance(engine_tc, str):
        engine_tc = TimeControl.from_string(engine_tc)
    result.append(f"tc={str(engine_tc)}")
    if timemargin is not None:
        result.append(f"timemargin={str(timemargin)}")
    if engine_ponder:
        result.append("ponder")
    return result
