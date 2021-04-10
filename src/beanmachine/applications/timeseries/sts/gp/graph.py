import math
from typing import List, Optional, OrderedDict, Tuple

import matplotlib.pyplot as plt
import torch
from gpytorch.distributions import MultivariateNormal
from sts.data import get_mvn_stats


def plot_predictions(
    x: torch.Tensor,
    y: MultivariateNormal,
    transform: Optional[torch.distributions.Transform] = None,
    axis: plt.Axes = None,
    **plot_kwargs
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
    plot_kwargs = plot_kwargs.copy()
    plot_fmt = {
        "color": plot_kwargs.pop("color", "black"),
        "linestyle": plot_kwargs.pop("linestyle", None),
        "linewidth": plot_kwargs.pop("linewidth", 1),
        "marker": plot_kwargs.pop("marker", None),
        "markerfacecolor": plot_kwargs.pop("markerfacecolor", None),
        "markersize": plot_kwargs.pop("markersize", None),
    }
    mean, _, ci = get_mvn_stats(y, transform)
    ax.plot(x, mean, label="predicted", **plot_fmt)
    ax.fill_between(x, ci[0], ci[1], color="gray", alpha=0.4)
    ax.set(**plot_kwargs)
    return fig, ax


def plot_components(
    x: torch.Tensor,
    components: OrderedDict[str, MultivariateNormal],
    transform: Optional[torch.distributions.Transform] = None,
    ncols: int = 2,
    y: Optional[MultivariateNormal] = None,
    **plot_kwargs
) -> List[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot the decomposition of the time series components.

    :param x: the x axis for the plot
    :param components: a dict of predictions for the individual components of
        the model keyed by the name of the component.
    :param transform: optional transform to apply to rescale the
        predictions from the model.
    :param ncols: number of columns in the subplots, defaults to 2.
    """
    figs = []
    if y is not None:
        figs.append(plot_predictions(x, y, transform, **plot_kwargs))
    num_components = len(components)
    nrows = math.ceil(num_components / float(ncols))
    fig, ax = plt.subplots(nrows, ncols)
    components = list(components.items())
    for i in range(nrows):
        for j in range(ncols):
            nplot = i * ncols + j
            if nplot == num_components:
                break
            ax_c = ax[i][j] if nrows > 1 else ax[j]
            name, mvn = components[nplot]
            plot_predictions(x, mvn, transform, ax_c, title=name, **plot_kwargs)
    figs.append((fig, ax))
    return figs
