import numpy as np

__all__ = ["confidence_intervals"]


def _round_interval(interval, threshold=0.01, max_precision=32):
    diff = interval[1] - interval[0]
    if diff == 0:
        return tuple(interval)
    for i in range(max_precision):
        rounded = np.around(interval, decimals=i)
        diff_i = np.diff(rounded).item()
        rel_error = abs(diff_i - diff) / diff
        if (
            rel_error <= threshold
            and len(np.unique(rounded)) == 2
            and rounded[0] < rounded[1]
        ):
            return tuple(np.around(interval, decimals=i))


def _round_all_intervals(intervals, threshold=0.01, max_precision=32):
    result = []
    for dim in intervals:
        sub_result = []
        if hasattr(dim[0], "__len__"):
            for sub in dim:
                sub_result.append(
                    _round_interval(
                        sub, threshold=threshold, max_precision=max_precision
                    )
                )
        else:
            sub_result.append(
                _round_interval(dim, threshold=threshold, max_precision=max_precision)
            )
        result.append(sub_result)
    return result


def confidence_intervals(
    optimizer,
    param_names=None,
    hdi_prob=0.95,
    multimodal=True,
    opt_samples=200,
    space_samples=500,
    only_mean=True,
    random_state=None,
    max_precision=32,
    threshold=0.01,
):
    if param_names is None:
        param_names = [
            "Parameter {}".format(i) for i in range(len(optimizer.space.dimensions))
        ]
    intervals = optimizer.optimum_intervals(
        hdi_prob=hdi_prob,
        multimodal=multimodal,
        opt_samples=opt_samples,
        space_samples=space_samples,
        only_mean=only_mean,
        random_state=random_state,
    )
    rounded = _round_all_intervals(
        intervals, max_precision=max_precision, threshold=threshold
    )
    max_param_length = max(max((len(x) for x in param_names)), 9)
    max_lb = max(max(len(str(row[0])) for sub in rounded for row in sub), 11)
    max_ub = max(max(len(str(row[1])) for sub in rounded for row in sub), 11)
    format_string = "{:<{}}  {:>{}}  {:>{}}\n"
    output = format_string.format(
        "Parameter", max_param_length, "Lower bound", max_lb, "Upper bound", max_ub
    )
    output += "{:-^{}}\n".format("", max_param_length + max_lb + max_ub + 4)
    for sub, name in zip(rounded, param_names):
        for i, interval in enumerate(sub):
            if i == 0:
                name_out = name
            else:
                name_out = ""
            output += format_string.format(
                name_out, max_param_length, interval[0], max_lb, interval[1], max_ub
            )
    return output
