# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Basic plotting methods used in the tutorials."""
from itertools import cycle
from typing import Any, Dict, List, Optional, Tuple, Union

import arviz as az
import numpy as np
from beanmachine.ppl import RVIdentifier
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from bokeh.layouts import LayoutDOM
from bokeh.models import (
    Circle,
    ColumnDataSource,
    Div,
    FixedTicker,
    HoverTool,
    Legend,
    Line,
)
from bokeh.palettes import Colorblind
from bokeh.plotting import figure, gridplot
from bokeh.plotting.figure import Figure


def style(plot: Figure) -> None:
    """Style the given plot.

    Args:
        plot (Figure): Bokeh `Figure` object to be styled.

    Returns:
        None: Nothing is returned, but the figure is now styled.
    """
    plot.set_from_json(name="outline_line_color", json="black")
    plot.grid.grid_line_alpha = 0.2
    plot.grid.grid_line_color = "grey"
    plot.grid.grid_line_width = 0.2
    plot.xaxis.minor_tick_line_color = "grey"
    plot.yaxis.minor_tick_line_color = "grey"


def choose_palette(n: int) -> Tuple[str]:
    """Choose an appropriate colorblind palette, given ``n``.

    Args:
        n (int): The size of the number of glyphs for a plot.

    Returns:
        palette (Tuple[str]): A tuple of color strings from Bokeh's colorblind palette.
    """
    palette_indices = [key for key in Colorblind.keys() if n <= key]
    if not palette_indices:
        palette_index = max(Colorblind.keys())
    else:
        palette_index = min(palette_indices)
    palette = Colorblind[palette_index]
    return palette


def bar_plot(
    plot_source: ColumnDataSource,
    orientation: Optional[str] = "vertical",
    figure_kwargs: Union[Dict[str, Any], None] = None,
    plot_kwargs: Union[Dict[str, Any], None] = None,
    tooltips: Union[List[Tuple[str, str]], None] = None,
) -> Figure:
    """Interactive Bokeh bar plot.

    Args:
        plot_source (ColumnDataSource): Bokeh object that contains data for the plot.
        orientation (Optional[str]): Optional orientation for the figure. Can be one of
            either: "vertical" (default) or "horizontal".
        figure_kwargs (Union[Dict[str, Any], None]): Figure arguments that change the
            style of the figure.
        plot_kwargs (Union[Dict[str, Any], None]): Plot arguments that change the style
            of the glyphs.
        tooltips (Union[List[Tuple[str, str]], None]): Hover tooltips.

    Returns:
        Figure: A Bokeh `Figure` object you can display in a notebook.
    """
    if figure_kwargs is None:
        figure_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}

    x = np.array(plot_source.data["x"])
    y = np.array(plot_source.data["y"])
    tick_labels = plot_source.data["tick_labels"]
    padding = 0.2
    range_ = []
    if orientation == "vertical":
        y_range_start = 1 - padding
        y_range_end = (1 + padding) * y.max()
        log_bounds = [y_range_start, y_range_end]
        minimum = (1 - padding) * y.min()
        maximum = (1 + padding) * y.max()
        no_log_bounds = [minimum, maximum]
        range_ = (
            log_bounds
            if figure_kwargs.get("y_axis_type", None) is not None
            else no_log_bounds
        )
    elif orientation == "horizontal":
        x_range_start = 1 - padding
        x_range_end = (1 + padding) * x.max()
        log_bounds = [x_range_start, x_range_end]
        minimum = (1 - padding) * x.min()
        maximum = (1 + padding) * x.max()
        no_log_bounds = [minimum, maximum]
        range_ = (
            log_bounds
            if figure_kwargs.get("x_axis_type", None) is not None
            else no_log_bounds
        )

    # Define default plot and figure keyword arguments.
    fig_kwargs = {
        "plot_width": 700,
        "plot_height": 500,
        "y_range" if orientation == "vertical" else "x_range": range_,
    }
    if figure_kwargs:
        fig_kwargs.update(figure_kwargs)
    plt_kwargs = {
        "fill_color": "steelblue",
        "fill_alpha": 0.7,
        "line_color": "white",
        "line_width": 1,
        "line_alpha": 0.7,
        "hover_fill_color": "orange",
        "hover_fill_alpha": 1,
        "hover_line_color": "black",
        "hover_line_width": 2,
        "hover_line_alpha": 1,
    }
    if plot_kwargs:
        plt_kwargs.update(plot_kwargs)

    # Create the plot.
    plot = figure(**fig_kwargs)

    # Bind data to the plot.
    glyph = plot.quad(
        left="left",
        top="top",
        right="right",
        bottom="bottom",
        source=plot_source,
        **plt_kwargs,
    )
    if tooltips is not None:
        tips = HoverTool(renderers=[glyph], tooltips=tooltips)
        plot.add_tools(tips)

    # Style the plot.
    style(plot)
    plot.xaxis.major_label_orientation = np.pi / 4
    if orientation == "vertical":
        plot.xaxis.ticker = FixedTicker(ticks=list(range(len(tick_labels))))
        plot.xaxis.major_label_overrides = dict(zip(range(len(x)), tick_labels))
        plot.xaxis.minor_tick_line_color = None
    if orientation == "horizontal":
        plot.yaxis.ticker = FixedTicker(ticks=list(range(len(tick_labels))))
        plot.yaxis.major_label_overrides = dict(zip(range(len(y)), tick_labels))
        plot.yaxis.minor_tick_line_color = None

    return plot


def histogram_plot(
    data,
    n_bins: Union[int, None] = None,
    figure_kwargs: Union[Dict[str, Any], None] = None,
    plot_kwargs: Union[Dict[str, Any], None] = None,
) -> Figure:
    if figure_kwargs is None:
        figure_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}
    if n_bins is None:
        n_bins = int(np.ceil(2 * np.log2(len(data)) + 1))

    top, density, bins = az.stats.density_utils.histogram(data=data, bins=n_bins)
    bottom = np.zeros(len(top))
    left = bins[:-1].tolist()
    right = bins[1:].tolist()
    label = [f"{item[0]:.3f} - {item[1]:.3f}" for item in zip(bins[:-1], bins[1:])]
    cds = ColumnDataSource(
        {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "label": label,
        }
    )
    fig = figure(
        plot_width=800,
        plot_height=500,
        y_axis_label="Counts",
        **figure_kwargs,
    )
    glyph = fig.quad(
        left="left",
        top="top",
        right="right",
        bottom="bottom",
        source=cds,
        fill_color="steelblue",
        line_color="white",
        fill_alpha=0.7,
        hover_fill_color="orange",
        hover_line_color="black",
        hover_alpha=1.0,
        **plot_kwargs,
    )
    tips = HoverTool(
        renderers=[glyph],
        tooltips=[("Counts", "@top"), ("Bin", "@label")],
    )
    fig.add_tools(tips)
    style(fig)
    return fig


def scatter_plot(  # noqa flake8 C901 too complex
    plot_sources: Union[ColumnDataSource, List[ColumnDataSource]],
    figure_kwargs: Union[Dict[str, Any], None] = None,
    plot_kwargs: Union[Dict[str, Any], None] = None,
    tooltips: Union[List[List[Tuple[str, str]]], List[Tuple[str, str]], None] = None,
    legend_items: Union[str, List[str], None] = None,
) -> Figure:
    """Create a scatter plot using Bokeh.

    Args:
        plot_sources (Union[ColumnDataSource, List[ColumnDataSource]]): Bokeh
            ``ColumnDataSource`` object(s).
        figure_kwargs (Union[Dict[str, Any], None]): (optional, default is None) Figure
            arguments that change the style of the figure.
        plot_kwargs (Union[Dict[str, Any], None]): (optional, default is None) Plot
            arguments that change the style of the glyphs of the figure.
        tooltips (Union[List[List[Tuple[str, str]]], List[Tuple[str, str]], None]):
            (optional, default is None) Hover tooltips.
        legend_items (Union[str, List[str], None]): (optional, default is None) Labels
            for the scatter items.

    Returns:
        plot (Figure): Bokeh figure you can visualize in a notebook.
    """
    if figure_kwargs is None:
        figure_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}
    if not isinstance(plot_sources, list):
        plot_sources = [plot_sources]
    if not isinstance(tooltips[0], list):
        if isinstance(tooltips[0], tuple):
            tooltips = [tooltips]
    if legend_items:
        if not isinstance(legend_items, list):
            legend_items = [legend_items]
    palette = choose_palette(len(plot_sources))
    colors = cycle(palette)

    # Define default plot and figure keyword arguments.
    fig_kwargs = {
        "plot_width": 700,
        "plot_height": 500,
    }
    if figure_kwargs:
        fig_kwargs.update(figure_kwargs)
    plt_kwargs = {
        "size": 10,
        "fill_alpha": 0.7,
        "line_color": "white",
        "line_width": 1,
        "line_alpha": 0.7,
        "hover_fill_color": "orange",
        "hover_fill_alpha": 1,
        "hover_line_color": "black",
        "hover_line_width": 2,
        "hover_line_alpha": 1,
    }
    if plot_kwargs:
        plt_kwargs.update(plot_kwargs)

    # Create the plot.
    plot = figure(**fig_kwargs)

    for i, plot_source in enumerate(plot_sources):
        if plot_kwargs:
            if "fill_color" in plot_kwargs:
                color = plot_kwargs["fill_color"]
        else:
            color = next(colors)
        plot_kwargs.update({"fill_color": color})
        if legend_items:
            glyph = plot.circle(
                x="x",
                y="y",
                source=plot_source,
                legend_label=legend_items[i],
                **plt_kwargs,
            )
        else:
            glyph = plot.circle(
                x="x",
                y="y",
                source=plot_source,
                **plt_kwargs,
            )
        if tooltips is not None:
            tips = HoverTool(renderers=[glyph], tooltips=tooltips[i])
            plot.add_tools(tips)

    # Style the plot.
    style(plot)

    return plot


def line_plot(
    plot_sources: Union[ColumnDataSource, List[ColumnDataSource]],
    labels: Union[List[str], None] = None,
    figure_kwargs: Union[Dict[str, Any], None] = None,
    tooltips: Union[List[List[Tuple[str, str]]], None] = None,
    plot_kwargs: Union[Dict[str, Any], None] = None,
) -> Figure:
    """Create a line plot using Bokeh.

    Args:
        plot_sources (Union[ColumnDataSource, List[ColumnDataSource]]): List of Bokeh
            `ColumnDataSource` objects or a single `ColumnDataSource`.
        labels (Union[List[str], None]): Labels for the legend. If none are given, then
            no legend will be generated.
        figure_kwargs (Union[Dict[str, Any], None]): Figure arguments that change the
            style of the figure.
        tooltips (Union[List[List[Tuple[str, str]]], None]): Hover tooltips.
        plot_kwargs (Union[Dict[str, Any], None]): Plot arguments that change the style
            of the glyphs.

    Returns:
        Figure:
    """
    if not isinstance(plot_sources, list):
        plot_sources = [plot_sources]
    if figure_kwargs is None:
        figure_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}
    palette = choose_palette(len(plot_sources))
    colors = cycle(palette)

    # Define default plot and figure keyword arguments.
    fig_kwargs = {
        "plot_width": 700,
        "plot_height": 500,
    }
    if figure_kwargs:
        fig_kwargs.update(figure_kwargs)

    plot = figure(**fig_kwargs)

    for i, plot_source in enumerate(plot_sources):
        color = next(colors)
        if labels is not None:
            plot_kwargs.update({"legend_label": labels[i]})
        locals()[f"glyph_{i}"] = plot.line(
            x="x",
            y="y",
            source=plot_source,
            color=color,
            **plot_kwargs,
        )
        if tooltips:
            plot.add_tools(
                HoverTool(
                    renderers=[locals()[f"glyph_{i}"]],
                    tooltips=tooltips[i],
                )
            )

    # Style the plot.
    style(plot)

    return plot


def plot_marginal(
    queries: List[RVIdentifier],
    samples: MonteCarloSamples,
    true_values: Union[List[float], List[None], None] = None,
    n_bins: Union[int, None] = None,
    bandwidth: Union[float, str] = "experimental",
    figure_kwargs: Union[Dict[str, Any], None] = None,
    joint_plot_title: Union[str, None] = None,
) -> LayoutDOM:
    """Marginal plot using Bokeh.

    - If one RV is given, then a single marginal plot will be shown.
    - If two RVs are given, then a joint plot is shown along with marginal densities.

    Args:
        queries (List[RVIdentifier]): Bean Machine `RVIdentifier` objects.
        samples (MonteCarloSamples): Bean Machine `MonteCarloSamples` object use to
            query data for the given queries.
        true_values (Union[List[float], List[None], None]): If you are creating
            simulated data then you can plot the true values if they are supplied.
        n_bins (Union[int, None]): The number of bins to use when generating the
            marginal plots. If no value is supplied, then twice the Sturges value will
            be used. See https://en.wikipedia.org/wiki/Histogram#Sturges'_formula
            or https://doi.org/10.1080%2F01621459.1926.10502161.
        bandwidth (Union[float, str]): Bandwidth to use for calculating the KDE, the
            default is `experimental`.
        figure_kwargs (Union[Dict[str, Any], None]): Figure keyword arguments supplied
            to the central figure.
        joint_plot_title (Union[str, None]): The title to display if two query objects
            has been given.

    Returns:
        LayoutDOM: An interactive Bokeh object that can be displayed in a notebook.
    """
    if len(queries) > 2:
        raise NotImplementedError("Can only handle two random variables at this time.")

    if true_values is None:
        true_values = [None] * len(queries)
    if figure_kwargs is None:
        figure_kwargs = {}

    layout = LayoutDOM()

    # Create an empty figure
    plot_width = 500
    plot_height = 500
    min_border = 0
    central_fig = figure(
        plot_width=plot_width,
        plot_height=plot_height,
        outline_line_color="black",
        min_border=min_border,
        title="",
        x_axis_label="",
        y_axis_label="",
        **figure_kwargs,
    )
    central_fig.add_layout(Legend(), "below")
    central_fig.grid.visible = False

    # Prepare data for the figure(s)
    figure_data = {}
    scaled_density = np.empty(0)
    for query in queries:
        data = samples[query]
        kde = az.stats.density_utils.kde(data.flatten().numpy(), bw=bandwidth)
        support = kde[0]
        density = kde[1]
        normalized_density = density / density.max()
        n_bins = (
            int(np.ceil(2 * np.log2(data.shape[1])) + 1) if n_bins is None else n_bins
        )
        histogram, _, bins = az.stats.density_utils.histogram(data, n_bins)
        scaled_density = normalized_density * histogram.max()
        density_cds = ColumnDataSource(
            {
                "x": support.tolist(),
                "scaled": scaled_density.tolist(),
                "normalized": normalized_density.tolist(),
            }
        )
        labels = [f"{item[0]:.3f}–{item[1]:.3f}" for item in zip(bins[:-1], bins[1:])]
        histogram_cds = ColumnDataSource(
            {
                "left": bins[:-1].tolist(),
                "top": histogram.tolist(),
                "right": bins[1:].tolist(),
                "bottom": [0] * len(histogram),
                "label": labels,
            }
        )
        figure_data[str(query).replace("()", "")] = {
            "histogram": histogram_cds,
            "density": density_cds,
        }

    if len(queries) == 1:
        query = queries[0]

        # Label the figure
        central_fig.xaxis.axis_label = str(query).replace("()", "")
        central_fig.yaxis.visible = False
        central_fig.set_from_json(name="outline_line_color", json="black")

        # Bind data to the figure
        histogram_glyph = central_fig.quad(
            left="left",
            top="top",
            right="right",
            bottom="bottom",
            source=figure_data[str(query).replace("()", "")]["histogram"],
            fill_color="steelblue",
            line_color="white",
            fill_alpha=0.6,
            hover_fill_color="orange",
            hover_line_color="black",
            hover_alpha=1,
            legend_label="Histogram",
        )
        density_glyph = central_fig.line(
            x="x",
            y="scaled",
            source=figure_data[str(query).replace("()", "")]["density"],
            line_color="brown",
            line_width=2,
            line_alpha=0.6,
            hover_line_color="magenta",
            hover_line_alpha=1,
            legend_label="Density",
        )

        # Add tooltips to the figure
        histogram_tips = HoverTool(
            renderers=[histogram_glyph],
            tooltips=[(f"{str(query).replace('()', '')}", "@label")],
        )
        central_fig.add_tools(histogram_tips)
        density_tips = HoverTool(
            renderers=[density_glyph],
            tooltips=[(f"{str(query).replace('()', '')}", "@x{0.000}")],
        )
        central_fig.add_tools(density_tips)

        true_value = true_values[0]
        true_cds = ColumnDataSource(
            {
                "x": [true_value] * 100,
                "y": np.linspace(0, scaled_density.max(), 100).tolist(),
            }
        )
        true_glyph = central_fig.line(
            x="x",
            y="y",
            source=true_cds,
            line_color="magenta",
            line_width=2,
            line_alpha=1,
            legend_label="True value",
        )
        true_tips = HoverTool(
            renderers=[true_glyph],
            tooltips=[("True value", "@x{0.000}")],
        )
        central_fig.add_tools(true_tips)
        mean_cds = ColumnDataSource(
            {
                "x": [samples[query].mean().item()] * 100,
                "y": np.linspace(0, scaled_density.max(), 100).tolist(),
            }
        )
        mean_glyph = central_fig.line(
            x="x",
            y="y",
            source=mean_cds,
            line_color="black",
            line_width=2,
            line_alpha=1,
            legend_label="Posterior mean value",
        )
        mean_tips = HoverTool(
            renderers=[mean_glyph],
            tooltips=[("Posterior mean value", "@x{0.000}")],
        )
        central_fig.add_tools(mean_tips)
        layout = gridplot([[central_fig]])

    if len(queries) == 2:
        title_div = None
        if joint_plot_title is not None:
            title_div = Div(text=f"<h3>{joint_plot_title}</h3>")
        # Prepare the 2D data
        v0 = samples[queries[0]].flatten().numpy()
        v1 = samples[queries[1]].flatten().numpy()
        density, xmin, xmax, ymin, ymax = az.stats.density_utils._fast_kde_2d(v0, v1)
        simulated_mean_cds = ColumnDataSource(
            {
                # Simulated mean
                f"{str(queries[1]).replace('()', '')}_x": np.linspace(
                    xmin,
                    xmax,
                    100,
                ).tolist(),
                f"{str(queries[1]).replace('()', '')}_y": [v1.mean()] * 100,
                f"{str(queries[0]).replace('()', '')}_x": [v0.mean()] * 100,
                f"{str(queries[0]).replace('()', '')}_y": np.linspace(
                    ymin,
                    ymax,
                    100,
                ).tolist(),
                # True
                f"{str(queries[1]).replace('()', '')}_true_x": np.linspace(
                    xmin,
                    xmax,
                    100,
                ).tolist(),
                f"{str(queries[1]).replace('()', '')}_true_y": [true_values[1]] * 100,
                f"{str(queries[0]).replace('()', '')}_true_x": [true_values[0]] * 100,
                f"{str(queries[0]).replace('()', '')}_true_y": np.linspace(
                    ymin,
                    ymax,
                    100,
                ).tolist(),
            }
        )

        # Style the central figure
        central_fig.x_range.start = xmin
        central_fig.x_range.end = xmax
        central_fig.x_range.max_interval = xmax
        central_fig.y_range.start = ymin
        central_fig.y_range.end = ymax
        central_fig.y_range.max_interval = ymax
        central_fig.set_from_json(name="match_aspect", json=True)
        central_fig.set_from_json(name="background_fill_color", json="#440154")
        central_fig.background_fill_alpha = 0.5
        central_fig.xaxis.axis_label = f"{str(queries[0]).replace('()', '')}"
        central_fig.yaxis.axis_label = f"{str(queries[1]).replace('()', '')}"

        # Create empty figures
        v0_fig = figure(
            plot_width=plot_width,
            plot_height=100,
            outline_line_color=None,
            x_range=central_fig.x_range,
            x_axis_location=None,
            min_border=min_border,
        )
        v0_fig.yaxis.visible = False
        v0_fig.xaxis.visible = False
        v0_fig.grid.visible = False
        v1_fig = figure(
            plot_width=100,
            plot_height=plot_height,
            outline_line_color=None,
            y_range=central_fig.y_range,
            y_axis_location=None,
            min_border=min_border,
        )
        v1_fig.yaxis.visible = False
        v1_fig.xaxis.visible = False
        v1_fig.grid.visible = False

        # Bind density data to the marginal plots
        v0_density_glyph = v0_fig.line(
            x="x",
            y="normalized",
            source=figure_data[str(queries[0]).replace("()", "")]["density"],
            line_color="steelblue",
            line_width=2,
            line_alpha=1,
        )
        v0_density_tips = HoverTool(
            renderers=[v0_density_glyph],
            tooltips=[(f"{str(queries[0]).replace('()', '')}", "@x{0.000}")],
        )
        v0_fig.add_tools(v0_density_tips)
        v0_mean_cds = ColumnDataSource(
            {"x": [v0.mean()] * 100, "y": np.linspace(0, 1, 100).tolist()}
        )
        v0_mean_glyph = v0_fig.line(
            x="x",
            y="y",
            source=v0_mean_cds,
            line_color="magenta",
            line_width=2,
            alpha=1,
        )
        v0_mean_tips = HoverTool(
            renderers=[v0_mean_glyph],
            tooltips=[(f"{str(queries[0]).replace('()', '')} mean", "@x{0.000}")],
        )
        v0_fig.add_tools(v0_mean_tips)

        v0_true_cds = ColumnDataSource(
            {"x": [true_values[0]] * 100, "y": np.linspace(0, 1, 100).tolist()}
        )
        v0_true_glyph = v0_fig.line(
            x="x",
            y="y",
            source=v0_true_cds,
            line_color="steelblue",
            line_width=2,
            alpha=1,
        )
        v0_true_tips = HoverTool(
            renderers=[v0_true_glyph],
            tooltips=[(f"{str(queries[0]).replace('()', '')} true", "@x{0.000}")],
        )
        v0_fig.add_tools(v0_true_tips)

        v1_true_cds = ColumnDataSource(
            {"y": [true_values[1]] * 100, "x": np.linspace(0, 1, 100).tolist()}
        )
        v1_true_glyph = v1_fig.line(
            x="x",
            y="y",
            source=v1_true_cds,
            line_color="steelblue",
            line_width=2,
            alpha=1,
        )
        v1_true_tips = HoverTool(
            renderers=[v1_true_glyph],
            tooltips=[(f"{str(queries[1]).replace('()', '')} true", "@y{0.000}")],
        )
        v1_fig.add_tools(v1_true_tips)

        v1_density_glyph = v1_fig.line(
            x="normalized",
            y="x",
            source=figure_data[str(queries[1]).replace("()", "")]["density"],
            line_color="steelblue",
            line_width=2,
            line_alpha=1,
        )
        v1_density_tips = HoverTool(
            renderers=[v1_density_glyph],
            tooltips=[(f"{str(queries[1]).replace('()', '')}", "@normalized{0.000}")],
        )
        v1_fig.add_tools(v1_density_tips)
        v1_mean_cds = ColumnDataSource(
            {"x": [v1.mean()] * 100, "y": np.linspace(0, 1, 100).tolist()}
        )
        v1_mean_glyph = v1_fig.line(
            x="y",
            y="x",
            source=v1_mean_cds,
            line_color="magenta",
            line_width=2,
            alpha=1,
        )
        v1_mean_tips = HoverTool(
            renderers=[v1_mean_glyph],
            tooltips=[(f"{str(queries[1]).replace('()', '')} mean", "@x{0.000}")],
        )
        v1_fig.add_tools(v1_mean_tips)

        central_fig.image(
            image=[density.T],
            x=xmin,
            y=ymin,
            dw=xmax - xmin,
            dh=ymax - ymin,
            palette="Viridis256",
        )
        v0_mean_joint_glyph = central_fig.line(
            x=f"{str(queries[0]).replace('()', '')}_x",
            y=f"{str(queries[0]).replace('()', '')}_y",
            source=simulated_mean_cds,
            line_color="magenta",
            line_width=2,
            line_alpha=0.5,
            legend_label="Posterior marginal mean",
        )
        v0_mean_joint_tips = HoverTool(
            renderers=[v0_mean_joint_glyph],
            tooltips=[
                (
                    f"{str(queries[0]).replace('()', '')} mean",
                    f"@{str(queries[0]).replace('()', '')}_x",
                )
            ],
        )
        central_fig.add_tools(v0_mean_joint_tips)

        v0_true_joint_glyph = central_fig.line(
            x=f"{str(queries[0]).replace('()', '')}_true_x",
            y=f"{str(queries[0]).replace('()', '')}_true_y",
            source=simulated_mean_cds,
            line_color="steelblue",
            line_width=2,
            line_alpha=0.5,
            legend_label="True value",
        )
        v0_true_joint_tips = HoverTool(
            renderers=[v0_true_joint_glyph],
            tooltips=[
                (
                    f"{str(queries[0]).replace('()', '')} true",
                    f"@{str(queries[0]).replace('()', '')}_true_x",
                )
            ],
        )
        central_fig.add_tools(v0_true_joint_tips)
        v1_true_joint_glyph = central_fig.line(
            x=f"{str(queries[1]).replace('()', '')}_true_x",
            y=f"{str(queries[1]).replace('()', '')}_true_y",
            source=simulated_mean_cds,
            line_color="steelblue",
            line_width=2,
            line_alpha=0.5,
            legend_label="True value",
        )
        v1_true_joint_tips = HoverTool(
            renderers=[v1_true_joint_glyph],
            tooltips=[
                (
                    f"{str(queries[1]).replace('()', '')} true",
                    f"@{str(queries[1]).replace('()', '')}_true_x",
                )
            ],
        )
        central_fig.add_tools(v1_true_joint_tips)

        v1_mean_joint_glyph = central_fig.line(
            x=f"{str(queries[1]).replace('()', '')}_x",
            y=f"{str(queries[1]).replace('()', '')}_y",
            source=simulated_mean_cds,
            line_color="magenta",
            line_width=2,
            line_alpha=0.5,
            legend_label="Posterior marginal mean",
        )
        v1_mean_joint_tips = HoverTool(
            renderers=[v1_mean_joint_glyph],
            tooltips=[
                (
                    f"{str(queries[1]).replace('()', '')} mean",
                    f"@{str(queries[1]).replace('()', '')}_y",
                )
            ],
        )
        central_fig.add_tools(v1_mean_joint_tips)
        mean_cds = ColumnDataSource({"x": [v0.mean()], "y": [v1.mean()]})
        mean_glyph = central_fig.circle(
            x="x",
            y="y",
            source=mean_cds,
            size=10,
            fill_color="magenta",
            line_color="white",
            fill_alpha=1,
            legend_label="Posterior marginal mean",
            hover_fill_color="orange",
            hover_line_color="black",
            hover_alpha=1,
        )
        mean_tips = HoverTool(
            renderers=[mean_glyph],
            tooltips=[
                (f"{str(queries[1]).replace('()', '')} mean", "@y{0.000}"),
                (f"{str(queries[0]).replace('()', '')} mean", "@x{0.000}"),
            ],
        )
        central_fig.add_tools(mean_tips)
        true_cds = ColumnDataSource({"x": [true_values[0]], "y": [true_values[1]]})
        true_glyph = central_fig.circle(
            x="x",
            y="y",
            source=true_cds,
            size=10,
            fill_color="steelblue",
            line_color="white",
            fill_alpha=1,
            legend_label="True value",
            hover_fill_color="orange",
            hover_line_color="black",
            hover_alpha=1,
        )
        true_tips = HoverTool(
            renderers=[true_glyph],
            tooltips=[
                (f"{str(queries[1]).replace('()', '')} true", "@y{0.000}"),
                (f"{str(queries[0]).replace('()', '')} true", "@x{0.000}"),
            ],
        )
        central_fig.add_tools(true_tips)
        if title_div is not None:
            layout = gridplot(
                [[title_div, None], [v0_fig, None], [central_fig, v1_fig]]
            )
        else:
            layout = gridplot([[v0_fig, None], [central_fig, v1_fig]])

    return layout


def marginal_2d(
    x,
    y,
    x_label: str,
    y_label: str,
    title: str,
    true_x: Union[float, None] = None,
    true_y: Union[float, None] = None,
    figure_kwargs: Union[Dict[str, Any], None] = None,
    bandwidth: Union[float, str] = "experimental",
    n_bins: Union[int, None] = None,
):
    # NOTE: This is duplicated from the plot_marginal, but it uses non Bean Machine
    #       objects to generate the plot. This replication was done for quickly getting
    #       a figure to work for the tutorials.

    if figure_kwargs is None:
        figure_kwargs = {}

    # Prepare the 1D data for the figure each feature.
    figure_data = {}
    # x-axis
    x = np.array(x)
    kde_x = az.stats.density_utils.kde(x, bw=bandwidth)
    support_x = kde_x[0]
    density_x = kde_x[1]
    normalized_density_x = density_x / density_x.max()
    n_bins_x = int(np.ceil(2 * np.log2(len(x)) + 1)) if n_bins is None else n_bins
    histogram_x, _, bins_x = az.stats.density_utils.histogram(x, n_bins_x)
    scaled_density_x = normalized_density_x * histogram_x.max()
    labels_x = [f"{item[0]:.3f}–{item[1]:.3f}" for item in zip(bins_x[:-1], bins_x[1:])]
    density_cds_x = ColumnDataSource(
        {
            "x": support_x.tolist(),
            "scaled": scaled_density_x.tolist(),
            "normalized": normalized_density_x.tolist(),
        }
    )
    histogram_cds_x = ColumnDataSource(
        {
            "left": bins_x[:-1].tolist(),
            "top": histogram_x.tolist(),
            "right": bins_x[1:].tolist(),
            "bottom": [0] * len(histogram_x),
            "label": labels_x,
        }
    )
    figure_data[x_label] = {
        "histogram": histogram_cds_x,
        "density": density_cds_x,
    }
    # y-axis
    y = np.array(y)
    kde_y = az.stats.density_utils.kde(y, bw=bandwidth)
    support_y = kde_y[0]
    density_y = kde_y[1]
    normalized_density_y = density_y / density_y.max()
    n_bins_y = int(np.ceil(2 * np.log2(len(y)) + 1)) if n_bins is None else n_bins
    histogram_y, _, bins_y = az.stats.density_utils.histogram(y, n_bins_y)
    scaled_density_y = normalized_density_y * histogram_y.max()
    labels_y = [f"{item[0]:.3f}–{item[1]:.3f}" for item in zip(bins_y[:-1], bins_y[1:])]
    density_cds_y = ColumnDataSource(
        {
            "x": support_y.tolist(),
            "scaled": scaled_density_y.tolist(),
            "normalized": normalized_density_y.tolist(),
        }
    )
    histogram_cds_y = ColumnDataSource(
        {
            "left": bins_y[:-1].tolist(),
            "top": histogram_y.tolist(),
            "right": bins_y[1:].tolist(),
            "bottom": [0] * len(histogram_y),
            "label": labels_y,
        }
    )
    figure_data[y_label] = {
        "histogram": histogram_cds_y,
        "density": density_cds_y,
    }
    # Prepare the 2D data for the figure.
    density_2d, xmin, xmax, ymin, ymax = az.stats.density_utils._fast_kde_2d(x, y)
    simulated_mean_cds = ColumnDataSource(
        {
            # Simulated mean
            "yaxis_x": np.linspace(xmin, xmax, 100).tolist(),
            "yaxis_y": [y.mean()] * 100,
            "xaxis_x": [x.mean()] * 100,
            "xaxis_y": np.linspace(ymin, ymax, 100).tolist(),
            # True
            "yaxis_true_x": np.linspace(xmin, xmax, 100).tolist(),
            "yaxis_true_y": [true_y] * 100,
            "xaxis_true_x": [true_x] * 100,
            "xaxis_true_y": np.linspace(ymin, ymax, 100).tolist(),
        }
    )

    # Figure title
    title_div = None
    if title is not None:
        title_div = Div(text=f"<h3>{title}</h3>")

    # Create central figure.
    plot_width = 500
    plot_height = 500
    min_border = 0
    central_fig = figure(
        plot_width=plot_width,
        plot_height=plot_height,
        outline_line_color="black",
        min_border=min_border,
        title="",
        x_axis_label=x_label,
        y_axis_label=y_label,
        **figure_kwargs,
    )
    central_fig.add_layout(Legend(), "below")
    central_fig.grid.visible = False

    # Style the central figure
    central_fig.x_range.start = xmin
    central_fig.x_range.end = xmax
    central_fig.x_range.max_interval = xmax
    central_fig.y_range.start = ymin
    central_fig.y_range.end = ymax
    central_fig.y_range.max_interval = ymax
    central_fig.set_from_json(name="match_aspect", json=True)
    central_fig.set_from_json(name="background_fill_color", json="#440154")
    central_fig.background_fill_alpha = 0.5
    central_fig.xaxis.axis_label = x_label
    central_fig.yaxis.axis_label = y_label

    # Create the marginal figures.
    # x-axis
    x_fig = figure(
        plot_width=plot_width,
        plot_height=100,
        outline_line_color=None,
        x_range=central_fig.x_range,
        x_axis_location=None,
        min_border=min_border,
    )
    x_fig.yaxis.visible = False
    x_fig.xaxis.visible = False
    x_fig.grid.visible = False
    # y-axis
    y_fig = figure(
        plot_width=100,
        plot_height=plot_height,
        outline_line_color=None,
        y_range=central_fig.y_range,
        y_axis_location=None,
        min_border=min_border,
    )
    y_fig.yaxis.visible = False
    y_fig.xaxis.visible = False
    y_fig.grid.visible = False

    # Bind density data to the marginal plots
    # x-axis
    x_density_glyph = x_fig.line(
        x="x",
        y="normalized",
        source=figure_data[x_label]["density"],
        line_color="steelblue",
        line_width=2,
        line_alpha=1,
    )
    x_density_tips = HoverTool(
        renderers=[x_density_glyph],
        tooltips=[(x_label, "@x{0.000}")],
    )
    x_fig.add_tools(x_density_tips)
    x_mean_cds = ColumnDataSource(
        {"x": [x.mean()] * 100, "y": np.linspace(0, 1, 100).tolist()}
    )
    x_mean_glyph = x_fig.line(
        x="x",
        y="y",
        source=x_mean_cds,
        line_color="magenta",
        line_width=2,
        alpha=1,
    )
    x_mean_tips = HoverTool(
        renderers=[x_mean_glyph],
        tooltips=[(f"{x_label} mean", "@x{0.000}")],
    )
    x_fig.add_tools(x_mean_tips)
    if true_x is not None:
        x_true_cds = ColumnDataSource(
            {"x": [true_x] * 100, "y": np.linspace(0, 1, 100).tolist()}
        )
        x_true_glyph = x_fig.line(
            x="x",
            y="y",
            source=x_true_cds,
            line_color="steelblue",
            line_width=2,
            alpha=1,
        )
        x_true_tips = HoverTool(
            renderers=[x_true_glyph],
            tooltips=[(f"{x_label} true", "@x{0.000}")],
        )
        x_fig.add_tools(x_true_tips)
    # y-axis
    if true_y is not None:
        y_true_cds = ColumnDataSource(
            {"y": [true_y] * 100, "x": np.linspace(0, 1, 100).tolist()}
        )
        y_true_glyph = y_fig.line(
            x="x",
            y="y",
            source=y_true_cds,
            line_color="steelblue",
            line_width=2,
            alpha=1,
        )
        y_true_tips = HoverTool(
            renderers=[y_true_glyph],
            tooltips=[(f"{y_label} true", "@y{0.000}")],
        )
        y_fig.add_tools(y_true_tips)
    y_density_glyph = y_fig.line(
        x="normalized",
        y="x",
        source=figure_data[y_label]["density"],
        line_color="steelblue",
        line_width=2,
        line_alpha=1,
    )
    y_density_tips = HoverTool(
        renderers=[y_density_glyph],
        tooltips=[(y_label, "@normalized{0.000}")],
    )
    y_fig.add_tools(y_density_tips)
    y_mean_cds = ColumnDataSource(
        {"x": [y.mean()] * 100, "y": np.linspace(0, 1, 100).tolist()}
    )
    y_mean_glyph = y_fig.line(
        x="y",
        y="x",
        source=y_mean_cds,
        line_color="magenta",
        line_width=2,
        alpha=1,
    )
    y_mean_tips = HoverTool(
        renderers=[y_mean_glyph],
        tooltips=[(f"{y_label} mean", "@x{0.000}")],
    )
    y_fig.add_tools(y_mean_tips)
    # joint figure
    central_fig.image(
        image=[density_2d.T],
        x=xmin,
        y=ymin,
        dw=xmax - xmin,
        dh=ymax - ymin,
        palette="Viridis256",
    )
    x_mean_joint_glyph = central_fig.line(
        x="xaxis_x",
        y="xaxis_y",
        source=simulated_mean_cds,
        line_color="magenta",
        line_width=2,
        line_alpha=0.5,
        legend_label="Posterior marginal mean",
    )
    x_mean_joint_tips = HoverTool(
        renderers=[x_mean_joint_glyph],
        tooltips=[(f"{x_label} mean", "@xaxis_x")],
    )
    central_fig.add_tools(x_mean_joint_tips)
    if true_x is not None:
        x_true_joint_glyph = central_fig.line(
            x="xaxis_true_x",
            y="xaxis_true_y",
            source=simulated_mean_cds,
            line_color="steelblue",
            line_width=2,
            line_alpha=0.5,
            legend_label="True value",
        )
        x_true_joint_tips = HoverTool(
            renderers=[x_true_joint_glyph],
            tooltips=[(f"{x_label} true", "@xaxis_true_x")],
        )
        central_fig.add_tools(x_true_joint_tips)
    if true_y is not None:
        y_true_joint_glyph = central_fig.line(
            x="yaxis_true_x",
            y="yaxis_true_y",
            source=simulated_mean_cds,
            line_color="steelblue",
            line_width=2,
            line_alpha=0.5,
            legend_label="True value",
        )
        y_true_joint_tips = HoverTool(
            renderers=[y_true_joint_glyph],
            tooltips=[(f"{y_label} true", "@yaxis_true_y")],
        )
        central_fig.add_tools(y_true_joint_tips)
    y_mean_joint_glyph = central_fig.line(
        x="yaxis_x",
        y="yaxis_y",
        source=simulated_mean_cds,
        line_color="magenta",
        line_width=2,
        line_alpha=0.5,
        legend_label="Posterior marginal mean",
    )
    y_mean_joint_tips = HoverTool(
        renderers=[y_mean_joint_glyph],
        tooltips=[(f"{y_label} mean", "@yaxis_y")],
    )
    central_fig.add_tools(y_mean_joint_tips)
    mean_cds = ColumnDataSource({"x": [x.mean()], "y": [y.mean()]})
    mean_glyph = central_fig.circle(
        x="x",
        y="y",
        source=mean_cds,
        size=10,
        fill_color="magenta",
        line_color="white",
        fill_alpha=1,
        legend_label="Posterior marginal mean",
        hover_fill_color="orange",
        hover_line_color="black",
        hover_alpha=1,
    )
    mean_tips = HoverTool(
        renderers=[mean_glyph],
        tooltips=[
            (f"{y_label} mean", "@y{0.000}"),
            (f"{x_label} mean", "@x{0.000}"),
        ],
    )
    central_fig.add_tools(mean_tips)
    if true_x is not None and true_y is not None:
        true_cds = ColumnDataSource({"x": [true_x], "y": [true_y]})
        true_glyph = central_fig.circle(
            x="x",
            y="y",
            source=true_cds,
            size=10,
            fill_color="steelblue",
            line_color="white",
            fill_alpha=1,
            legend_label="True value",
            hover_fill_color="orange",
            hover_line_color="black",
            hover_alpha=1,
        )
        true_tips = HoverTool(
            renderers=[true_glyph],
            tooltips=[
                (f"{y_label} true", "@y{0.000}"),
                (f"{x_label} true", "@x{0.000}"),
            ],
        )
        central_fig.add_tools(true_tips)
    if title_div is not None:
        layout = gridplot([[title_div, None], [x_fig, None], [central_fig, y_fig]])
    else:
        layout = gridplot([[x_fig, None], [central_fig, y_fig]])

    return layout


def plot_diagnostics(
    samples: MonteCarloSamples,
    ordering: Union[None, List[str]] = None,
    plot_posterior: bool = False,
) -> List[Figure]:
    """
    Plot model diagnostics.

    :param samples: Bean Machine inference object.
    :type samples: MonteCarloSamples
    :param ordering: Define an ordering for how the plots are displayed.
    :type ordering: List[str]
    :return: Bokeh figures with visual diagnostics.
    :rtype: List[Figure]
    """
    COLORS = ["#2a2eec", "#fa7c17", "#328c06", "#c10c90"]

    # Prepare the data for the figure.
    samples_xr = samples.to_xarray()
    data = {str(key): value.values for key, value in samples_xr.data_vars.items()}

    if ordering is not None:
        diagnostics_data = {}
        for key in ordering:
            key = str(key)
            diagnostics_data[key] = data[key]
    else:
        diagnostics_data = data

    # Cycle through each query and create the diagnostics plots using arviz.
    diagnostics_plots = []
    for key, value in diagnostics_data.items():
        posterior_plot = None
        if plot_posterior:
            posterior_plot = az.plot_posterior({key: value}, show=False)[0][0]
            posterior_plot.plot_width = 300
            posterior_plot.plot_height = 300
            posterior_plot.grid.grid_line_alpha = 0.2
            posterior_plot.grid.grid_line_color = "gray"
            posterior_plot.grid.grid_line_width = 0.3
            posterior_plot.yaxis.minor_tick_line_color = None
            posterior_plot.outline_line_color = "black"

        tr_plot = az.plot_trace(az.from_dict({key: value}), show=False)[0][1]
        line_index = 0
        circle_index = 0
        for renderer in tr_plot.renderers:
            glyph = renderer._property_values["glyph"]
            if isinstance(glyph, Line):
                glyph.line_color = COLORS[line_index]
                glyph.line_dash = "solid"
                glyph.line_width = 2
                glyph.line_alpha = 0.6
                line_index += 1
            if isinstance(renderer._property_values["glyph"], Circle):
                glyph.fill_color = COLORS[circle_index]
                glyph.line_color = COLORS[circle_index]
                glyph.fill_alpha = 0.6
                circle_index += 1
        tr_plot.plot_width = 300
        tr_plot.plot_height = 300
        tr_plot.grid.grid_line_alpha = 0.2
        tr_plot.grid.grid_line_color = "gray"
        tr_plot.grid.grid_line_width = 0.3
        tr_plot.yaxis.minor_tick_line_color = None
        tr_plot.outline_line_color = "black"
        tr_plot.title.text = f"{tr_plot.title.text} trace plot"

        ac_plot = az.plot_autocorr({key: value}, show=False)[0].tolist()
        for i, p in enumerate(ac_plot):
            for renderer in p.renderers:
                glyph = renderer._property_values["glyph"]
                glyph.line_color = COLORS[i]
            p.plot_width = 300
            p.plot_height = 300
            p.grid.grid_line_alpha = 0.2
            p.grid.grid_line_color = "gray"
            p.grid.grid_line_width = 0.3
            p.yaxis.minor_tick_line_color = None
            p.outline_line_color = "black"
            p.title.text = f"{p.title.text.split()[0]}\nautocorrelation chain {i}"
        if plot_posterior:
            ps = [posterior_plot, tr_plot, *ac_plot]
        else:
            ps = [tr_plot, *ac_plot]
        diagnostics_plots.append(ps)

    return diagnostics_plots
