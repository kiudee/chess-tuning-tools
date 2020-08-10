"""Top-level package for Chess Tuning Tools."""

__author__ = """Karlson Pfannschmidt"""
__email__ = "kiudee@mail.upb.de"
__version__ = "0.5.0-beta.2"

from tune.io import InitStrings, load_tuning_config, parse_ranges
from tune.local import parse_experiment_result, reduce_ranges, run_match
from tune.plots import partial_dependence, plot_objective
from tune.priors import roundflat
from tune.utils import TimeControl, TimeControlBag, expected_ucb, parse_timecontrol

__all__ = [
    "InitStrings",
    "load_tuning_config",
    "parse_ranges",
    "parse_experiment_result",
    "run_match",
    "reduce_ranges",
    "partial_dependence",
    "plot_objective",
    "roundflat",
    "expected_ucb",
    "parse_timecontrol",
    "TimeControl",
    "TimeControlBag",
]
