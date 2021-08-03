import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import invgamma
from scipy.stats._distn_infrastructure import rv_frozen  # noqa

__all__ = ["make_invgamma_prior", "roundflat"]


def roundflat(x, a_low=2.0, a_high=8.0, d_low=0.005, d_high=1.2):
    """Return the log probability of the round flat prior.

    The round flat prior is completely uninformative inside the interval bounds
    ``d_low`` and ``d_high`` while smoothly going to -inf for values outside.
    ``a_low`` and ``a_high`` specify how quickly the density falls at the boundaries.

    Args:
        x (float): A parameter value in [0, inf) for which to compute the log
            probability
        a_low (float): Steepness of the prior at the boundary ``d_low``.
        a_high (float): Steepness of the prior at the boundary ``d_high``.
        d_low (float): Lower boundary for which the log probability is -2.
        d_high (float): Upper boundary for which the log probability is -2.

    Returns:
        The log probability for x.
    """
    if x <= 0:
        return -np.inf
    return -2 * ((x / d_low) ** (-2 * a_low) + (x / d_high) ** (2 * a_high))


def make_invgamma_prior(
    lower_bound: float = 0.1, upper_bound: float = 0.5
) -> rv_frozen:
    """Create an inverse gamma distribution prior with 98% density inside the bounds.

    Not all combinations of (lower_bound, upper_bound) are feasible and some of them
    could result in a RuntimeError.

    Parameters
    ----------
    lower_bound : float, default=0.1
        Lower bound at which 1 % of the cumulative density is reached.
    upper_bound : float, default=0.5
        Upper bound at which 99 % of the cumulative density is reached.

    Returns
    -------
    scipy.stats._distn_infrastructure.rv_frozen
        The frozen distribution with shape parameters already set.

    Raises
    ------
    ValueError
        Either if any of the bounds is 0 or negative, or if the upper bound is equal or
        smaller than the lower bound.
    """
    if lower_bound <= 0 or upper_bound <= 0:
        raise ValueError("The bounds cannot be equal to or smaller than 0.")
    if lower_bound >= upper_bound:
        raise ValueError(
            "Lower bound needs to be strictly smaller than the upper " "bound."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (a_out, scale_out), pcov = curve_fit(
            lambda xdata, a, scale: invgamma.ppf(xdata, a=a, scale=scale),
            [0.01, 0.99],
            [lower_bound, upper_bound],
        )
    return invgamma(a=a_out, scale=scale_out)
