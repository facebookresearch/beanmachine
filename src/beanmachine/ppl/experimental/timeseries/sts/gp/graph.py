# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import math
from typing import List, Optional, OrderedDict, Tuple

import matplotlib.pyplot as plt
import torch
from gpytorch.distributions import MultivariateNormal
from sts.data import get_mvn_stats


def _decouple_plot_fmt_attrs(plot_kwargs):
    plot_kwargs = plot_kwargs.copy()
    plot_fmt = {
        "color": plot_kwargs.pop("color", "black"),
        "linestyle": plot_kwargs.pop("linestyle", None),
        "linewidth": plot_kwargs.pop("linewidth", 1),
        "marker": plot_kwargs.pop("marker", None),
        "markerfacecolor": plot_kwargs.pop("markerfacecolor", None),
        "markersize": plot_kwargs.pop("markersize", None),
    }
    return plot_fmt, plot_kwargs


def plot_predictions(
    x: torch.Tensor,
    y: MultivariateNormal,
    transform: Optional[torch.distributions.Transform] = None,
    axis: plt.Axes = None,
    **plot_kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the predictions from the time-series model.

    :param x: the x axis for the plot.
    :param y: the multivariate normal distribution representing the prediction
        from the model.
    :param transform: optional transform to apply to rescale the
        predictions from the model.
    :param axis: optional axis
    """
    fig, ax = None, axis
    if axis is None:
        fig, ax = plt.subplots(1, 1)
    plot_fmt, plot_kwargs = _decouple_plot_fmt_attrs(plot_kwargs)
    mean, _, ci = get_mvn_stats(y, transform)
    ax.plot(x, mean, label="predicted", **plot_fmt)
    ax.fill_between(x, ci[0], ci[1], color="gray", alpha=0.4)
    ax.set(**plot_kwargs)
    return fig, ax


def plot_components(
    x: torch.Tensor,
    components: Tuple[OrderedDict[str, MultivariateNormal]],
    transform: Optional[torch.distributions.Transform] = None,
    ncols: int = 2,
    y: Optional[MultivariateNormal] = None,
    plot_mean: Optional[bool] = False,
    **plot_kwargs,
) -> List[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot the decomposition of the time series components.

    :param x: the x axis for the plot
    :param components: a dict of predictions for the individual components of
        the model keyed by the name of the component.
    :param transform: optional transform to apply to rescale the
        predictions from the model.
    :param y: the multivariate normal distribution representing the predictions
        from the model.
    :param bool plot_mean: Whether to plot the components of the mean function.
        Defaults to False.
    :param ncols: number of columns in the subplots, defaults to 2.
    """
    figs = []
    if y is not None:
        figs.append(plot_predictions(x, y, transform, **plot_kwargs))

    mean_components = components[0]
    cov_components = components[1]
    if plot_mean:
        num_mean = len(mean_components)
        components = list(mean_components.items()) + list(cov_components.items())
        num_components = num_mean + len(cov_components)
    else:
        num_mean = 0
        components = list(cov_components.items())
        num_components = len(cov_components)

    nrows = math.ceil(num_components / float(ncols))
    fig, ax = plt.subplots(nrows, ncols, squeeze=False)
    plot_fmt, plot_attrs = _decouple_plot_fmt_attrs(plot_kwargs)
    for i in range(nrows):
        for j in range(ncols):
            nplot = i * ncols + j
            if nplot == num_components:
                break
            ax_c = ax[i][j]
            name, comp = components[nplot]
            if nplot < num_mean:
                ax_c.plot(x, comp.detach(), **plot_fmt)
                ax_c.set(title=name, **plot_attrs)
            else:
                plot_predictions(x, comp, transform, ax_c, title=name, **plot_kwargs)
    figs.append((fig, ax))
    return figs
