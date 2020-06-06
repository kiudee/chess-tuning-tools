import re
import subprocess

import numpy as np
from scipy.stats import dirichlet


__all__ = ["run_experiment", "parse_experiment_result"]


def parse_experiment_result(outstr, prior_counts=None, n_samples=1000000):
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
    n_samples : int, default = 1 000 000
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
    error = dist.rvs(n_samples).dot(scores).var()
    return score, error

