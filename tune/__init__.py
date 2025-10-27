"""Top-level package for Chess Tuning Tools."""

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version

__author__ = """Karlson Pfannschmidt"""
__email__ = "kiudee@mail.upb.de"
__version__ = version("chess-tuning-tools")

from tune.io import InitStrings, load_tuning_config, parse_ranges
from tune.local import (
    elo_to_prob,
    parse_experiment_result,
    prob_to_elo,
    reduce_ranges,
    run_match,
)
from tune.plots import (
    partial_dependence,
    plot_objective,
    plot_optima,
    plot_performance,
)
from tune.priors import roundflat
from tune.utils import (
    TimeControl,
    TimeControlBag,
    expected_ucb,
    parse_timecontrol,
)

__all__ = [
    "elo_to_prob",
    "expected_ucb",
    "InitStrings",
    "load_tuning_config",
    "parse_experiment_result",
    "parse_ranges",
    "parse_timecontrol",
    "partial_dependence",
    "plot_objective",
    "plot_optima",
    "plot_performance",
    "prob_to_elo",
    "reduce_ranges",
    "roundflat",
    "run_match",
    "TimeControl",
    "TimeControlBag",
]
