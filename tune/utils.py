import numpy as np
from scipy.optimize import minimize


def expected_ucb(res, n_random_starts=100, alpha=1.96, random_state=None):
    """ Compute the expected (pessimistic) optimum of the optimization result.

    This will compute `gp_mean + alpha * gp_se` and find the optimum of that function.

    Parameters
    ----------
    * `res`  [`OptimizeResult`, scipy object]:
        The optimization result returned by a `bask` minimizer.
    * `n_random_starts` [int, default=100]:
        The number of random starts for the minimization of the surrogate model.
    * `alpha` [float, default=1.96]:
        Number of standard errors to add to each point.
    * `random_state` [int, RandomState instance, or None (default)]:
        Set random state to something other than None for reproducible
        results.
    Returns
    -------
    * `x` [list]: location of the minimum.
    * `fun` [float]: the surrogate function value at the minimum.
    """

    def func(x):
        reg = res.models[-1]
        mu, std = reg.predict(x.reshape(1, -1), return_std=True)
        return (mu + alpha * std)[0]

    xs = [res.space.transform([res.x]).tolist()]
    if n_random_starts > 0:
        xs.extend(
            res.space.transform(
                res.space.rvs(n_random_starts, random_state=random_state)
            ).tolist()
        )

    best_x = None
    best_fun = np.inf

    for x0 in xs:
        r = minimize(func, x0=x0, bounds=[(0.0, 1.0)] * len(res.space.bounds))

        if r.fun < best_fun:
            best_x = r.x
            best_fun = r.fun
    return res.space.inverse_transform(best_x[None, :])[0], best_fun
