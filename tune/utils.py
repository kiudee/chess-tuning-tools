from collections import namedtuple
from decimal import Decimal
import itertools

import numpy as np
from scipy.optimize import minimize


__all__ = ["expected_ucb", "parse_timecontrol", "TimeControl", "TimeControlBag"]


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


def parse_timecontrol(tc_string):
    if "+" in tc_string:
        return tuple([Decimal(x) for x in tc_string.split("+")])
    return (Decimal(tc_string),)


TC = namedtuple("TimeControl", ["time", "increment"],)


class TimeControl(TC):
    @classmethod
    def from_string(cls, tc_string):
        tc = parse_timecontrol(tc_string)
        inc = Decimal("0.0") if len(tc) == 1 else tc[1]
        return cls(time=Decimal(tc[0]), increment=inc,)

    def __str__(self):
        if abs(self.increment) < 1e-10:
            return f"{self.time}"
        return f"{self.time}+{self.increment}"


def _latin_1d(n):
    return (np.random.uniform(0, 1, size=n) + np.arange(n)) / n


def _probabilistic_round(x):
    return int(np.floor(x + np.random.uniform()))


class TimeControlBag(object):
    def __init__(self, tcs, bag_size=10, p=None):
        self.bag = []
        self.tcs = tcs
        if p is not None:
            self.p = p
        else:
            self.p = np.ones(len(tcs)) / len(tcs)
        self.bag_size = bag_size

    def next_tc(self):
        if self.bag is None or len(self.bag) == 0:
            out = []
            for p in self.p:
                out.append(_probabilistic_round(p * self.bag_size))
            tmp_bag = []
            for o, tc in zip(out, self.tcs):
                tmp_bag.extend(itertools.product(_latin_1d(o), [tc]))
            sorted_bag = sorted(tmp_bag)
            self.bag = [x[1] for x in sorted_bag]
        return self.bag.pop()
