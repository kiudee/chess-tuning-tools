from skopt.plots import _format_scatter_plot_axes
from tune.utils import expected_ucb

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator


def _evenly_sample(dim, n_points):
    """Return `n_points` evenly spaced points from a Dimension.
    Parameters
    ----------
    dim : `Dimension`
        The Dimension to sample from.  Can be categorical; evenly-spaced
        category indices are chosen in order without replacement (result
        may be smaller than `n_points`).
    n_points : int
        The number of points to sample from `dim`.
    Returns
    -------
    xi : np.array
        The sampled points in the Dimension.  For Categorical
        dimensions, returns the index of the value in
        `dim.categories`.
    xi_transformed : np.array
        The transformed values of `xi`, for feeding to a model.
    """
    cats = np.array(getattr(dim, 'categories', []), dtype=object)
    if len(cats):  # Sample categoricals while maintaining order
        xi = np.linspace(0, len(cats) - 1, min(len(cats), n_points),
                         dtype=int)
        xi_transformed = dim.transform(cats[xi])
    else:
        bounds = dim.bounds
        # XXX use linspace(*bounds, n_points) after python2 support ends
        if dim.prior == "log-uniform":
            xi = np.logspace(np.log10(bounds[0]), np.log10(bounds[1]), n_points)
        else:
            xi = np.linspace(bounds[0], bounds[1], n_points)
        xi_transformed = dim.transform(xi)
    return xi, xi_transformed


def partial_dependence(space, model, i, j=None, sample_points=None,
                       n_samples=250, n_points=40, x_eval=None):
    """Calculate the partial dependence for dimensions `i` and `j` with
    respect to the objective value, as approximated by `model`.
    The partial dependence plot shows how the value of the dimensions
    `i` and `j` influence the `model` predictions after "averaging out"
    the influence of all other dimensions.
    When `x_eval` is not `None`, the given values are used instead of
    random samples. In this case, `n_samples` will be ignored.
    Parameters
    ----------
    space : `Space`
        The parameter space over which the minimization was performed.
    model
        Surrogate model for the objective function.
    i : int
        The first dimension for which to calculate the partial dependence.
    j : int, default=None
        The second dimension for which to calculate the partial dependence.
        To calculate the 1D partial dependence on `i` alone set `j=None`.
    sample_points : np.array, shape=(n_points, n_dims), default=None
        Only used when `x_eval=None`, i.e in case partial dependence should
        be calculated.
        Randomly sampled and transformed points to use when averaging
        the model function at each of the `n_points` when using partial
        dependence.
    n_samples : int, default=100
        Number of random samples to use for averaging the model function
        at each of the `n_points` when using partial dependence. Only used
        when `sample_points=None` and `x_eval=None`.
    n_points : int, default=40
        Number of points at which to evaluate the partial dependence
        along each dimension `i` and `j`.
    x_eval : list, default=None
        `x_eval` is a list of parameter values or None. In case `x_eval`
        is not None, the parsed dependence will be calculated using these
        values.
        Otherwise, random selected samples will be used.
    Returns
    -------
    For 1D partial dependence:
    xi : np.array
        The points at which the partial dependence was evaluated.
    yi : np.array
        The value of the model at each point `xi`.
    For 2D partial dependence:
    xi : np.array, shape=n_points
        The points at which the partial dependence was evaluated.
    yi : np.array, shape=n_points
        The points at which the partial dependence was evaluated.
    zi : np.array, shape=(n_points, n_points)
        The value of the model at each point `(xi, yi)`.
    For Categorical variables, the `xi` (and `yi` for 2D) returned are
    the indices of the variable in `Dimension.categories`.
    """
    # The idea is to step through one dimension, evaluating the model with
    # that dimension fixed and averaging either over random values or over
    # the given ones in x_val in all other dimensions.
    # (Or step through 2 dimensions when i and j are given.)
    # Categorical dimensions make this interesting, because they are one-
    # hot-encoded, so there is a one-to-many mapping of input dimensions
    # to transformed (model) dimensions.

    # If we haven't parsed an x_eval list we use random sampled values instead
    if x_eval is None and sample_points is None:
        sample_points = space.transform(space.rvs(n_samples=n_samples))
    elif sample_points is None:
        sample_points = space.transform([x_eval])

    # dim_locs[i] is the (column index of the) start of dim i in
    # sample_points.
    # This is usefull when we are using one hot encoding, i.e using
    # categorical values
    dim_locs = np.cumsum([0] + [d.transformed_size for d in space.dimensions])

    if j is None:
        # We sample evenly instead of randomly. This is necessary when using
        # categorical values
        xi, xi_transformed = _evenly_sample(space.dimensions[i], n_points)
        yi = []
        for x_ in xi_transformed:
            rvs_ = np.array(sample_points)  # copy
            # We replace the values in the dimension that we want to keep
            # fixed
            rvs_[:, dim_locs[i]:dim_locs[i + 1]] = x_
            # In case of `x_eval=None` rvs conists of random samples.
            # Calculating the mean of these samples is how partial dependence
            # is implemented.
            yi.append(np.mean(model.predict(rvs_)))

        return xi, yi

    else:
        xi, xi_transformed = _evenly_sample(space.dimensions[j], n_points)
        yi, yi_transformed = _evenly_sample(space.dimensions[i], n_points)

        zi = []
        for x_ in xi_transformed:
            row = []
            for y_ in yi_transformed:
                rvs_ = np.array(sample_points)  # copy
                rvs_[:, dim_locs[j]:dim_locs[j + 1]] = x_
                rvs_[:, dim_locs[i]:dim_locs[i + 1]] = y_
                row.append(np.mean(model.predict(rvs_)))
            zi.append(row)

        return xi, yi, np.array(zi).T


def plot_objective(
    result,
    levels=20,
    n_points=200,
    n_samples=30,
    size=3,
    zscale="linear",
    dimensions=None,
    n_random_restarts=100,
    alpha=0.25,
    colors=None,
    fig=None,
    ax=None,
):
    """Pairwise partial dependence plot of the objective function.
    The diagonal shows the partial dependence for dimension `i` with
    respect to the objective function. The off-diagonal shows the
    partial dependence for dimensions `i` and `j` with
    respect to the objective function. The objective function is
    approximated by `result.model.`
    Pairwise scatter plots of the points at which the objective
    function was directly evaluated are shown on the off-diagonal.
    A red point indicates the found minimum.
    Note: search spaces that contain `Categorical` dimensions are
          currently not supported by this function.
    Parameters
    ----------
    * `result` [`OptimizeResult`]
        The result for which to create the scatter plot matrix.
    * `levels` [int, default=10]
        Number of levels to draw on the contour plot, passed directly
        to `plt.contour()`.
    * `n_points` [int, default=40]
        Number of points at which to evaluate the partial dependence
        along each dimension.
    * `n_samples` [int, default=250]
        Number of random samples to use for averaging the model function
        at each of the `n_points`.
    * `size` [float, default=2]
        Height (in inches) of each facet.
    * `zscale` [str, default='linear']
        Scale to use for the z axis of the contour plots. Either 'linear'
        or 'log'.
    * `dimensions` [list of str, default=None] Labels of the dimension
        variables. `None` defaults to `space.dimensions[i].name`, or
        if also `None` to `['X_0', 'X_1', ..]`.
    * `n_random_restarts` [int, default=100]
        Number of restarts to try to find the global optimum.
    * `alpha` [float, default=0.5]
        Transparency of the sampled points.
    * `colors` [list of tuples, default=None]
        Colors to use for the optima.
    * `fig` [Matplotlib figure, default=None]
        Figure to use for plotting. If None, it will create one.
    * `ax` [k x k axes, default=None]
        Axes on which to plot the marginals. If None, it will create appropriate
        axes.
    Returns
    -------
    * `ax`: [`Axes`]:
        The matplotlib axes.
    """
    if colors is None:
        colors = plt.cm.get_cmap("Set3").colors
    space = result.space
    samples = np.asarray(result.x_iters)
    rvs_transformed = space.transform(space.rvs(n_samples=n_samples))

    if zscale == "log":
        locator = LogLocator()
    elif zscale == "linear":
        locator = None
    else:
        raise ValueError(
            "Valid values for zscale are 'linear' and 'log'," " not '%s'." % zscale
        )
    if fig is None:
        fig, ax = plt.subplots(
            space.n_dims,
            space.n_dims,
            figsize=(size * space.n_dims, size * space.n_dims),
        )

    fig.subplots_adjust(
        left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.1, wspace=0.1
    )
    failures = 0
    while True:
        try:
            with result.models[-1].noise_set_to_zero():
                min_x = expected_ucb(
                    result, alpha=0.0, n_random_starts=n_random_restarts
                )[0]
                min_ucb = expected_ucb(result, n_random_starts=n_random_restarts)[0]
        except:
            failures += 1
            if failures == 10:
                break
            continue
        else:
            break

    for i in range(space.n_dims):
        for j in range(space.n_dims):
            if i == j:
                xi, yi = partial_dependence(
                    space,
                    result.models[-1],
                    i,
                    j=None,
                    sample_points=rvs_transformed,
                    n_points=n_points,
                )
                yi_min, yi_max = np.min(yi), np.max(yi)
                ax[i, i].plot(xi, yi, color=colors[1])
                if failures != 10:
                    ax[i, i].axvline(min_x[i], linestyle="--", color=colors[3], lw=1)
                    ax[i, i].axvline(min_ucb[i], linestyle="--", color=colors[5], lw=1)
                    ax[i, i].text(
                        min_x[i],
                        yi_min + 0.9 * (yi_max - yi_min),
                        f"{np.around(min_x[i], 4)}",
                        color=colors[3],
                    )
                    ax[i, i].text(
                        min_ucb[i],
                        yi_min + 0.7 * (yi_max - yi_min),
                        f"{np.around(min_ucb[i], 4)}",
                        color=colors[5],
                    )

            # lower triangle
            elif i > j:
                xi, yi, zi = partial_dependence(
                    space, result.models[-1], i, j, rvs_transformed, n_points
                )
                ax[i, j].contourf(xi, yi, zi, levels, locator=locator, cmap="viridis_r")
                ax[i, j].scatter(
                    samples[:, j], samples[:, i], c="k", s=10, lw=0.0, alpha=alpha
                )
                if failures != 10:
                    ax[i, j].scatter(min_x[j], min_x[i], c=["r"], s=20, lw=0.0)
                    ax[i, j].scatter(
                        min_ucb[j], min_ucb[i], c=["xkcd:orange"], s=20, lw=0.0
                    )

    return _format_scatter_plot_axes(
        ax, space, ylabel="Partial dependence", dim_labels=dimensions
    )
