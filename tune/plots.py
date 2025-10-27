import itertools
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import LogLocator
from scipy.optimize import OptimizeResult
from skopt.plots import _format_scatter_plot_axes
from skopt.space import Space

from tune.utils import confidence_to_mult, expected_ucb, latest_iterations

__all__ = [
    "partial_dependence",
    "plot_objective",
    "plot_objective_1d",
    "plot_optima",
    "plot_performance",
]


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
    cats = np.array(getattr(dim, "categories", []), dtype=object)
    if len(cats):  # Sample categoricals while maintaining order
        xi = np.linspace(0, len(cats) - 1, min(len(cats), n_points), dtype=int)
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


def partial_dependence(
    space,
    model,
    i,
    j=None,
    sample_points=None,
    n_samples=250,
    n_points=40,
    x_eval=None,
):
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
            rvs_[:, dim_locs[i] : dim_locs[i + 1]] = x_
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
                rvs_[:, dim_locs[j] : dim_locs[j + 1]] = x_
                rvs_[:, dim_locs[i] : dim_locs[i + 1]] = y_
                row.append(np.mean(model.predict(rvs_)))
            zi.append(row)

        return xi, yi, np.array(zi).T


def plot_objective_1d(
    result: OptimizeResult,
    parameter_name: Optional[str] = None,
    n_points: int = 500,
    n_random_restarts: int = 100,
    confidence: float = 0.9,
    figsize: Tuple[float, float] = (10, 6),
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    colors: Optional[Sequence[Union[tuple, str]]] = None,
) -> Tuple[Figure, Axes]:
    """Plot the 1D objective function.

    Parameters
    ----------
    result : OptimizeResult
        The current optimization result.
    parameter_name : str, optional
        The name of the parameter to plot. If None, no x-axis label is shown.
    n_points : int (default=500)
        The number of points to use for prediction of the Gaussian process.
    n_random_restarts : int (default=100)
        The number of random restarts to employ to find the optima.
    confidence : float (default=0.9)
        The confidence interval to plot around the mean prediction.
    figsize : tuple (default=(10, 6))
        The size of the figure.
    fig : Figure, optional
        The figure to use. If None, a new figure is created.
    ax : Axes, optional
        The axes to use. If None, new axes are created.
    colors : Sequence of colors, optional
        The colors to use for different elements in the plot.
        Can be tuples or strings.

    Returns
    -------
    fig : Figure
        The figure.
    ax : Axes
        The axes.

    """
    if colors is None:
        colors = plt.cm.get_cmap("Set3").colors

    if fig is None:
        plt.style.use("dark_background")
        gs_kw = {
            "width_ratios": (1,),
            "height_ratios": [5, 1],
            "hspace": 0.05,
        }
        fig, ax = plt.subplots(
            figsize=figsize, nrows=2, gridspec_kw=gs_kw, sharex=True
        )
        for a in ax:
            a.set_facecolor("#36393f")
        fig.patch.set_facecolor("#36393f")
    gp = result.models[-1]

    # Compute the optima of the objective function:
    failures = 0
    while True:
        try:
            with gp.noise_set_to_zero():
                min_x = expected_ucb(
                    result, alpha=0.0, n_random_starts=n_random_restarts
                )[0]
                min_ucb = expected_ucb(
                    result, n_random_starts=n_random_restarts
                )[0]
        except ValueError:
            failures += 1
            if failures == 10:
                break
            continue
        else:
            break

    # Regardless of the range of the parameter to be plotted, the model always operates
    # in [0, 1]:
    x_gp = np.linspace(0, 1, num=n_points)
    x_orig = np.array(result.space.inverse_transform(x_gp[:, None])).flatten()
    with gp.noise_set_to_zero():
        y, y_err = gp.predict(x_gp[:, None], return_std=True)
    y = -y * 100
    y_err = y_err * 100
    confidence_mult = confidence_to_mult(confidence)

    (mean_plot,) = ax[0].plot(x_orig, y, zorder=4, color=colors[0])
    err_plot = ax[0].fill_between(
        x_orig,
        y - y_err * confidence_mult,
        y + y_err * confidence_mult,
        alpha=0.3,
        zorder=0,
        color=colors[0],
    )
    opt_plot = ax[0].axvline(x=min_x, zorder=3, color=colors[3])
    pess_plot = ax[0].axvline(x=min_ucb, zorder=2, color=colors[5])
    if parameter_name is not None:
        ax[1].set_xlabel(parameter_name)
    dim = result.space.dimensions[0]
    ax[0].set_xlim(dim.low, dim.high)
    match_plot = ax[1].scatter(
        x=result.x_iters,
        y=-result.func_vals * 100,
        zorder=1,
        marker=".",
        s=0.6,
        color=colors[0],
    )
    ax[0].set_ylabel("Elo")
    ax[1].set_ylabel("Elo")
    fig.legend(
        (mean_plot, err_plot, opt_plot, pess_plot, match_plot),
        (
            "Mean",
            f"{confidence:.0%} CI",
            "Optimum",
            "Conservative Optimum",
            "Matches",
        ),
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.03),
        frameon=False,
    )

    return fig, ax


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
    margin=0.65,
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
    * `alpha` [float, default=0.25]
        Transparency of the sampled points.
    * `margin` [float, default=0.65]
        Margin in inches around the plot.
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
            "Valid values for zscale are 'linear' and 'log', not '%s'." % zscale
        )
    if fig is None:
        fig, ax = plt.subplots(
            space.n_dims,
            space.n_dims,
            figsize=(size * space.n_dims, size * space.n_dims),
        )
    width, height = fig.get_size_inches()

    fig.subplots_adjust(
        left=margin / width,
        right=1 - margin / width,
        bottom=margin / height,
        top=1 - margin / height,
        hspace=0.1,
        wspace=0.1,
    )
    failures = 0
    while True:
        try:
            with result.models[-1].noise_set_to_zero():
                min_x = expected_ucb(
                    result, alpha=0.0, n_random_starts=n_random_restarts
                )[0]
                min_ucb = expected_ucb(
                    result, n_random_starts=n_random_restarts
                )[0]
        except ValueError:
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
                    ax[i, i].axvline(
                        min_x[i], linestyle="--", color=colors[3], lw=1
                    )
                    ax[i, i].axvline(
                        min_ucb[i], linestyle="--", color=colors[5], lw=1
                    )
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
                ax[i, j].contourf(
                    xi, yi, zi, levels, locator=locator, cmap="viridis_r"
                )
                ax[i, j].scatter(
                    samples[:, j],
                    samples[:, i],
                    c="k",
                    s=10,
                    lw=0.0,
                    alpha=alpha,
                )
                if failures != 10:
                    ax[i, j].scatter(min_x[j], min_x[i], c=["r"], s=20, lw=0.0)
                    ax[i, j].scatter(
                        min_ucb[j], min_ucb[i], c=["xkcd:orange"], s=20, lw=0.0
                    )
    # Get all dimensions.
    plot_dims = []
    for row in range(space.n_dims):
        if space.dimensions[row].is_constant:
            continue
        plot_dims.append((row, space.dimensions[row]))
    return _format_scatter_plot_axes(
        ax,
        space,
        ylabel="Partial dependence",
        plot_dims=plot_dims,
        dim_labels=dimensions,
    )


def plot_optima(
    iterations: np.ndarray,
    optima: np.ndarray,
    space: Optional[Space] = None,
    parameter_names: Optional[Sequence[str]] = None,
    plot_width: float = 8,
    aspect_ratio: float = 0.4,
    fig: Optional[Figure] = None,
    ax: Optional[Union[Axes, np.ndarray]] = None,
    colors: Optional[Sequence[Union[tuple, str]]] = None,
) -> Tuple[Figure, np.ndarray]:
    """Plot the optima found by the tuning algorithm.

    Parameters
    ----------
    iterations : np.ndarray
        The iterations at which the optima were found.
    optima : np.ndarray
        The optima found recorded at the given iterations.
    space : Space, optional
        The optimization space for the parameters. If provided, it will be used to
        scale the y-axes and to apply log-scaling, if the parameter is optimized on
        a log-scale.
    parameter_names : Sequence[str], optional
        The names of the parameters. If not provided, no y-axis labels will be shown.
    plot_width : int, optional
        The width of each plot in inches. The total width of the plot will be larger
        depending on the number of parameters and how they are arranged.
    aspect_ratio : float, optional
        The aspect ratio of the subplots. The default is 0.4, which means that the
        height of each subplot will be 40% of the width.
    fig : Figure, optional
        The figure to plot on. If not provided, a new figure in the style of
        chess-tuning-tools will be created.
    ax : np.ndarray or Axes, optional
        The axes to plot on. If not provided, new axes will be created.
        If provided, the axes will be filled. Thus, the number of axes should be at
        least as large as the number of parameters.
    colors : Sequence[Union[tuple, str]], optional
        The colors to use for the plots. If not provided, the color scheme 'Set3' of
        matplotlib will be used.

    Returns
    -------
    Figure
        The figure containing the plots.
    np.ndarray
        A two-dimensional array containing the axes.

    Raises
    ------
    ValueError
        - if the number of parameters does not match the number of parameter names
        - if the number of axes is smaller than the number of parameters
        - if the number of iterations is not matching the number of optima
        - if a fig, but no ax is passed
    """
    if optima.shape[0] != len(iterations):
        raise ValueError("Iteration array does not match optima array.")
    iterations, optima = latest_iterations(iterations, optima)
    n_points, n_parameters = optima.shape
    if parameter_names is not None and len(parameter_names) != n_parameters:
        raise ValueError(
            "Number of parameter names does not match the number of parameters."
        )
    if colors is None:
        colors = plt.cm.get_cmap("Set3").colors
    n_colors = len(colors)
    if fig is None:
        plt.style.use("dark_background")
        n_cols = int(np.floor(np.sqrt(n_parameters)))
        n_rows = int(np.ceil(n_parameters / n_cols))
        figsize = (n_cols * plot_width, aspect_ratio * plot_width * n_rows)
        fig, ax = plt.subplots(
            figsize=figsize,
            nrows=n_rows,
            ncols=n_cols,
            sharex=True,
        )

        margin_left = 1.0
        margin_right = 0.1
        margin_bottom = 0.5
        margin_top = 0.4
        wspace = 1
        hspace = 0.3
        plt.subplots_adjust(
            left=margin_left / figsize[0],
            right=1 - margin_right / figsize[0],
            bottom=margin_bottom / figsize[1],
            top=1 - margin_top / figsize[1],
            hspace=n_rows * hspace / figsize[1],
            wspace=n_cols * wspace / figsize[0],
        )
        ax = np.atleast_2d(ax).reshape(n_rows, n_cols)
        for a in ax.reshape(-1):
            a.set_facecolor("#36393f")
            a.grid(which="major", color="#ffffff", alpha=0.1)
        fig.patch.set_facecolor("#36393f")
        fig.suptitle(
            "Predicted best parameters over time",
            y=1 - 0.5 * margin_top / figsize[1],
            va="center",
        )
    else:
        if ax is None:
            raise ValueError("Axes must be specified if a figure is provided.")
        if not hasattr(ax, "__len__"):
            n_rows = n_cols = 1
        elif ax.ndim == 1:
            n_rows = len(ax)
            n_cols = 1
        else:
            n_rows, n_cols = ax.shape
        if n_rows * n_cols < n_parameters:
            raise ValueError("Not enough axes to plot all parameters.")
        ax = np.atleast_2d(ax).reshape(n_rows, n_cols)

    for i, (j, k) in enumerate(itertools.product(range(n_rows), range(n_cols))):
        a = ax[j, k]
        if i >= n_parameters:
            fig.delaxes(a)
            continue
        # If the axis is the last one in the current column, then set the xlabel:
        if (j + 1) * n_cols + k + 1 > n_parameters:
            a.set_xlabel("Iteration")
            # Since hspace=0, we have to fix the xaxis label and tick labels here:
            a.xaxis.set_label_coords(0.5, -0.12)
            a.xaxis.set_tick_params(labelbottom=True)

        a.plot(
            iterations,
            optima[:, i],
            color=colors[i % n_colors],
            zorder=10,
            linewidth=1.3,
        )
        a.axhline(
            y=optima[-1, i],
            color=colors[i % n_colors],
            zorder=9,
            linewidth=0.5,
            ls="--",
            alpha=0.6,
        )
        # If the user supplied an optimization space, we can use that information to
        # scale the yaxis and apply log-scaling, where necessary:
        s = f"{optima[-1, i]:.2f}"
        if space is not None:
            dim = space.dimensions[i]
            a.set_ylim(*dim.bounds)
            if dim.prior == "log-uniform":
                a.set_yscale("log")
                s = np.format_float_scientific(optima[-1, i], precision=2)

        # Label the horizontal line of the current optimal value:
        # First convert the y-value to normalized axes coordinates:
        point = a.get_xlim()[0], optima[-1, i]
        transformed_point = a.transAxes.inverted().transform(
            a.transData.transform(point)
        )
        a.text(
            x=transformed_point[0] + 0.01,
            y=transformed_point[1] - 0.02,
            s=s,
            bbox={
                "facecolor": "#36393f",
                "edgecolor": "None",
                "alpha": 0.5,
                "boxstyle": "square,pad=0.1",
            },
            transform=a.transAxes,
            horizontalalignment="left",
            verticalalignment="top",
            color=colors[i % n_colors],
            zorder=11,
        )

        if parameter_names is not None:
            a.set_ylabel(parameter_names[i])
    return fig, ax


def plot_performance(
    performance: np.ndarray,
    confidence: float = 0.9,
    plot_width: float = 8,
    aspect_ratio: float = 0.7,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    colors: Optional[Sequence[Union[tuple, str]]] = None,
) -> Tuple[Figure, np.ndarray]:
    """Plot the estimated Elo of the Optima predicted by the tuning algorithm.

    Parameters
    ----------
    performance : np.ndarray, shape=(n_iterations, 3)
        Array containing the iteration numbers, the estimated Elo of the predicted
        optimum, and the estimated standard error of the estimated Elo.
    confidence : float, optional (default=0.9)
        The confidence interval to plot around the estimated Elo.
    plot_width : int, optional (default=8)
        The width of each plot in inches. The total width of the plot will be larger
        depending on the number of parameters and how they are arranged.
    aspect_ratio : float, optional (default=0.7)
        The aspect ratio of the subplots. The default is 0.4, which means that the
        height of each subplot will be 40% of the width.
    fig : Figure, optional
        The figure to plot on. If not provided, a new figure in the style of
        chess-tuning-tools will be created.
    ax : np.ndarray or Axes, optional
        The axes to plot on. If not provided, new axes will be created.
        If provided, the axes will be filled. Thus, the number of axes should be at
        least as large as the number of parameters.
    colors : Sequence[Union[tuple, str]], optional
        The colors to use for the plots. If not provided, the color scheme 'Set3' of
        matplotlib will be used.

    Returns
    -------
    Figure
        The figure containing the plots.
    np.ndarray
        A two-dimensional array containing the axes.

    Raises
    ------
    ValueError
        - if the number of parameters does not match the number of parameter names
        - if the number of axes is smaller than the number of parameters
        - if the number of iterations is not matching the number of optima
        - if a fig, but no ax is passed
    """
    iterations, elo, elo_std = latest_iterations(*performance.T)
    if colors is None:
        colors = plt.cm.get_cmap("Set3").colors
    if fig is None:
        plt.style.use("dark_background")
        figsize = (plot_width, aspect_ratio * plot_width)
        fig, ax = plt.subplots(figsize=figsize)

        margin_left = 0.8
        margin_right = 0.1
        margin_bottom = 0.7
        margin_top = 0.3
        plt.subplots_adjust(
            left=margin_left / figsize[0],
            right=1 - margin_right / figsize[0],
            bottom=margin_bottom / figsize[1],
            top=1 - margin_top / figsize[1],
        )
        ax.set_facecolor("#36393f")
        ax.grid(which="major", color="#ffffff", alpha=0.1)
        fig.patch.set_facecolor("#36393f")
        ax.set_title("Elo of the predicted best parameters over time")
    elif ax is None:
        raise ValueError("Axes must be specified if a figure is provided.")

    ax.plot(
        iterations,
        elo,
        color=colors[0],
        zorder=10,
        linewidth=1.3,
        label="Predicted Elo",
    )
    confidence_mult = confidence_to_mult(confidence)
    ax.fill_between(
        iterations,
        elo - confidence_mult * elo_std,
        elo + confidence_mult * elo_std,
        color=colors[0],
        linewidth=0,
        zorder=9,
        alpha=0.25,
        label=f"{confidence:.0%} confidence interval",
    )
    ax.axhline(
        y=elo[-1],
        linestyle="--",
        zorder=8,
        color=colors[0],
        label="Last prediction",
        linewidth=1,
        alpha=0.3,
    )
    ax.legend(
        loc="upper center", frameon=False, bbox_to_anchor=(0.5, -0.08), ncol=3
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Elo")
    ax.set_xlim(min(iterations), max(iterations))

    return fig, ax
