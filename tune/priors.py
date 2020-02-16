import numpy as np

__all__ = ["roundflat"]


def roundflat(x, a_low=2.0, a_high=8.0, d_low=0.005, d_high=1.2):
    """Return the log probability of the round flat prior.

    The round flat prior is completely uninformative inside the interval bounds
    ``d_low`` and ``d_high`` while smoothly going to -inf for values outside.
    ``a_low`` and ``a_high`` specify how quickly the density falls at the boundaries.

    Args:
        x (float): A parameter value in [0, inf) for which to compute the log probability
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
