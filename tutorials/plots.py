# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from bokeh.models import ColumnDataSource, FixedTicker, HoverTool
from bokeh.plotting import figure
from bokeh.plotting.figure import Figure


def style(plot: Figure) -> None:
    """
    Style the given plot.

    :param plot: Figure object to style.
    """
    plot.outline_line_color = "black"
    plot.grid.grid_line_alpha = 0.2
    plot.grid.grid_line_color = "grey"
    plot.grid.grid_line_width = 0.2


def bar_plot(
    plot_source: ColumnDataSource,
    orientation: Optional[str] = "vertical",
    figure_kwargs: Optional[Union[Dict[str, Any], None]] = None,
    plot_kwargs: Optional[Union[Dict[str, Any], None]] = None,
    tooltips: Optional[Union[None, List[Tuple[str, str]]]] = None,
) -> Figure:
    """
    Create and style a bar plot using the given `plot_source`.

    :param plot_source: Bokeh `ColumnDataSource` object.
    :param orientation: Can be one of "vertical" or "horizontal".
    :param figure_kwargs: Figure options to pass to the Bokeh figure object.
    :param plot_kwargs: Plot options to pass to the Bokeh figure object.
    :param tooltips: A list of tuples Bokeh can use for hover tips.
    :returns plot: The bar plot using Bokeh as the backend.
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


def scatter_plot(
    plot_source: ColumnDataSource,
    figure_kwargs: Optional[Union[Dict[str, Any], None]] = None,
    plot_kwargs: Optional[Union[Dict[str, Any], None]] = None,
    tooltips: Optional[Union[None, List[Tuple[str, str]]]] = None,
) -> Figure:
    """
    Style and create a scatter plot using the given `plot_source`.

    :param plot_source: Bokeh `ColumnDataSource` object.
    :param figure_kwargs: Figure options to pass to the Bokeh figure object.
    :param plot_kwargs: Plot options to pass to the Bokeh figure object.
    :param tooltips: A list of tuples Bokeh can use for hover tips.
    :returns plot: The bar plot using Bokeh as the backend.
    """
    if figure_kwargs is None:
        figure_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}

    # Define default plot and figure keyword arguments.
    fig_kwargs = {
        "plot_width": 700,
        "plot_height": 500,
    }
    if figure_kwargs:
        fig_kwargs.update(figure_kwargs)
    plt_kwargs = {
        "size": 10,
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

    glyph = plot.circle(
        x="x",
        y="y",
        source=plot_source,
        **plt_kwargs,
    )
    if tooltips is not None:
        tips = HoverTool(renderers=[glyph], tooltips=tooltips)
        plot.add_tools(tips)

    # Style the plot.
    style(plot)

    return plot


def line_plot(
    plot_sources: List[ColumnDataSource],
    labels: List[str],
    colors: List[str],
    figure_kwargs: Optional[Union[Dict[str, Any], None]] = None,
    tooltips: Optional[Union[None, List[List[Tuple[str, str]]]]] = None,
) -> Figure:
    """
    Style and create a line plot using the given `plot_source`.

    :param plot_source: Bokeh `ColumnDataSource` object.
    :param labels: Labels to use in the legend.
    :param colors: Colors to use for multiple lines.
    :param figure_kwargs: Figure options to pass to the Bokeh figure object.
    :param tooltips: A list of tuples Bokeh can use for hover tips.
    :returns plot: The bar plot using Bokeh as the backend.
    """
    if figure_kwargs is None:
        figure_kwargs = {}

    # Define default plot and figure keyword arguments.
    fig_kwargs = {
        "plot_width": 700,
        "plot_height": 500,
    }
    if figure_kwargs:
        fig_kwargs.update(figure_kwargs)

    plot = figure(**fig_kwargs)

    for i, plot_source in enumerate(plot_sources):
        locals()[f"glyph_{i}"] = plot.line(
            x="x",
            y="y",
            source=plot_source,
            color=colors[i],
            legend_label=labels[i],
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
