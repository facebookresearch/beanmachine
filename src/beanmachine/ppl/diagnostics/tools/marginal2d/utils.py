# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Methods used to generate the diagnostic tool."""
from __future__ import annotations

from typing import List

import numpy as np
from beanmachine.ppl.diagnostics.tools.marginal2d import typing
from beanmachine.ppl.diagnostics.tools.utils.plotting_utils import (
    choose_palette,
    create_toolbar,
    filter_renderers,
)
from bokeh.models.annotations import Band
from bokeh.models.glyphs import Circle, Line
from bokeh.models.layouts import Column, GridBox, Row
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.markups import Div
from bokeh.models.widgets.panels import Panel, Tabs
from bokeh.models.widgets.sliders import Slider
from bokeh.plotting import figure


MARGINAL1D_PLOT_WIDTH = 500
MARGINAL1D_PLOT_HEIGHT = 100
MARGINAL2D_PLOT_WIDTH = MARGINAL1D_PLOT_WIDTH
MARGINAL2D_PLOT_HEIGHT = MARGINAL2D_PLOT_WIDTH
# Define what the empty data object looks like in order to make the browser handle all
# computations.
EMPTY_DATA = {
    "x": {
        "distribution": {"x": [], "y": [], "bandwidth": np.NaN},
        "hdi": {"base": [], "lower": [], "upper": []},
        "stats": {"x": [], "y": [], "text": []},
        "labels": {
            "x": [],
            "y": [],
            "text": [],
            "text_align": [],
            "x_offset": [],
            "y_offset": [],
        },
    },
    "y": {
        "distribution": {"x": [], "y": [], "bandwidth": np.NaN},
        "hdi": {
            "lower": {"base": [], "lower": [], "upper": []},
            "upper": {"base": [], "lower": [], "upper": []},
        },
        "stats": {"x": [], "y": [], "text": []},
        "labels": {
            "x": [],
            "y": [],
            "text": [],
            "text_align": [],
            "x_offset": [],
            "y_offset": [],
        },
    },
    "xy": {
        "distribution": {"x": [], "y": []},
        "hdi": {
            "x": {"lower": {"x": [], "y": []}, "upper": {"x": [], "y": []}},
            "y": {"lower": {"x": [], "y": []}, "upper": {"x": [], "y": []}},
        },
        "stats": {"x": [], "y": [], "text": []},
        "labels": {
            "x": [],
            "y": [],
            "text": [],
            "text_align": [],
            "x_offset": [],
            "y_offset": [],
        },
    },
}


def create_sources() -> typing.Sources:
    """
    Create Bokeh sources that will be bound to glyphs.

    Returns:
        typing.Sources: A dictionary of Bokeh ColumnDataSource objects.
    """
    output = {
        "x": {
            "distribution": ColumnDataSource({"x": [], "y": []}),
            "hdi": ColumnDataSource({"base": [], "lower": [], "upper": []}),
            "stats": ColumnDataSource({"x": [], "y": [], "text": []}),
        },
        "y": {
            "distribution": ColumnDataSource({"x": [], "y": []}),
            "hdi": {
                "lower": ColumnDataSource({"base": [], "lower": [], "upper": []}),
                "upper": ColumnDataSource({"base": [], "lower": [], "upper": []}),
            },
            "stats": ColumnDataSource({"x": [], "y": [], "text": []}),
        },
        "xy": {
            "distribution": ColumnDataSource({"x": [], "y": []}),
            "hdi": {
                "x": {
                    "lower": ColumnDataSource({"x": [], "y": []}),
                    "upper": ColumnDataSource({"x": [], "y": []}),
                },
                "y": {
                    "lower": ColumnDataSource({"x": [], "y": []}),
                    "upper": ColumnDataSource({"x": [], "y": []}),
                },
            },
            "stats": ColumnDataSource({"x": [], "y": [], "text": []}),
        },
    }
    return output


def create_figures(rv_name_x: str, rv_name_y: str) -> typing.Figures:
    """
    Create the Bokeh figures used for the tool.

    Args:
        rv_name_x (str): The name of the random variable data in the x-direction.
        rv_name_y (str): The name of the random variable data in the y-direction.

    Returns:
        typing.Figures: A dictionary of Bokeh Figure objects.
    """
    # x figure
    x = figure(
        outline_line_color=None,
        min_border=None,
        width=MARGINAL1D_PLOT_WIDTH,
        height=MARGINAL1D_PLOT_HEIGHT,
        name="x",
        y_axis_label=rv_name_y,
    )
    # NOTE: The extra steps we are taking for customizing both the x and y figures stem
    #       from the fact that the plots will shift to fit in their allotted space if no
    #       axis exists. In order to prevent these visual shifts, we keep the axes
    #       around and manipulate them so you cannot see them. Unfortunately we must
    #       keep the major tick labels around and color them white, otherwise the
    #       problem will persist.
    x.grid.visible = False
    x.xaxis.visible = False
    x.x_range.range_padding = 0
    x.y_range.range_padding = 0
    x.yaxis.axis_label_text_color = None
    x.yaxis.axis_line_color = None
    x.yaxis.major_label_text_color = "white"
    x.yaxis.major_tick_line_color = None
    x.yaxis.minor_tick_line_color = None

    # y figure
    y = figure(
        outline_line_color=None,
        min_border=None,
        width=MARGINAL1D_PLOT_HEIGHT,
        height=MARGINAL1D_PLOT_WIDTH,
        name="y",
        x_axis_label=rv_name_x,
    )
    y.grid.visible = False
    y.yaxis.visible = False
    y.x_range.range_padding = 0
    y.y_range.range_padding = 0
    y.xaxis.axis_label_text_color = None
    y.xaxis.axis_line_color = None
    y.xaxis.major_label_text_color = "white"
    y.xaxis.major_tick_line_color = None
    y.xaxis.minor_tick_line_color = None

    # xy figure
    xy = figure(
        outline_line_color="black",
        min_border=None,
        width=MARGINAL2D_PLOT_WIDTH,
        height=MARGINAL2D_PLOT_HEIGHT,
        x_axis_label=rv_name_x,
        y_axis_label=rv_name_y,
        x_range=x.x_range,
        y_range=y.y_range,
        name="xy",
    )
    xy.grid.visible = False
    xy.x_range.range_padding = 0
    xy.y_range.range_padding = 0

    output = {"x": x, "y": y, "xy": xy}
    return output


def create_glyphs() -> typing.Glyphs:
    """
    Create the glyphs used for the figures of the tool.

    Returns:
        typing.Glyphs: A dictionary of Bokeh Glyphs objects.
    """
    palette = choose_palette(4)
    glyph_color = palette[0]
    hover_glyph_color = palette[1]
    mean_color = palette[3]
    output = {
        "x": {
            "distribution": {
                "glyph": Line(
                    x="x",
                    y="y",
                    line_alpha=0.7,
                    line_color=glyph_color,
                    line_width=2.0,
                    name="xDistribution",
                ),
                "hover_glyph": Line(
                    x="x",
                    y="y",
                    line_alpha=1.0,
                    line_color=hover_glyph_color,
                    line_width=2.0,
                ),
            },
            "stats": {
                "glyph": Circle(
                    x="x",
                    y="y",
                    size=10,
                    fill_alpha=1.0,
                    fill_color=glyph_color,
                    line_color="white",
                    name="xStats",
                ),
                "hover_glyph": Circle(
                    x="x",
                    y="y",
                    size=10,
                    fill_alpha=1.0,
                    fill_color=hover_glyph_color,
                    line_color="black",
                ),
            },
        },
        "y": {
            "distribution": {
                "glyph": Line(
                    x="x",
                    y="y",
                    line_alpha=0.7,
                    line_color=glyph_color,
                    line_width=2.0,
                    name="yDistribution",
                ),
                "hover_glyph": Line(
                    x="x",
                    y="y",
                    line_alpha=1.0,
                    line_color=hover_glyph_color,
                    line_width=2.0,
                ),
            },
            "stats": {
                "glyph": Circle(
                    x="x",
                    y="y",
                    size=10,
                    fill_alpha=1.0,
                    fill_color=glyph_color,
                    line_color="white",
                    name="yStats",
                ),
                "hover_glyph": Circle(
                    x="x",
                    y="y",
                    size=10,
                    fill_alpha=1.0,
                    fill_color=hover_glyph_color,
                    line_color="black",
                ),
            },
        },
        "xy": {
            "distribution": {
                "glyph": Circle(
                    x="x",
                    y="y",
                    size=5,
                    fill_alpha=0.4,
                    line_color="white",
                    fill_color=glyph_color,
                    name="xyDistribution",
                ),
                "hover_glyph": Circle(
                    x="x",
                    y="y",
                    size=5,
                    fill_alpha=1.0,
                    line_color="black",
                    fill_color=hover_glyph_color,
                ),
            },
            "hdi": {
                "x": {
                    "lower": {
                        "glyph": Line(
                            x="x",
                            y="y",
                            line_alpha=0.7,
                            line_color="black",
                            line_width=2.0,
                            name="xyLowerXHDI",
                        ),
                        "hover_glyph": Line(
                            x="x",
                            y="y",
                            line_alpha=1.0,
                            line_color="black",
                            line_width=2.0,
                        ),
                    },
                    "upper": {
                        "glyph": Line(
                            x="x",
                            y="y",
                            line_alpha=0.7,
                            line_color="black",
                            line_width=2.0,
                            name="xyUpperXHDI",
                        ),
                        "hover_glyph": Line(
                            x="x",
                            y="y",
                            line_alpha=1.0,
                            line_color="black",
                            line_width=2.0,
                        ),
                    },
                },
                "y": {
                    "lower": {
                        "glyph": Line(
                            x="x",
                            y="y",
                            line_alpha=0.7,
                            line_color="black",
                            line_width=2.0,
                            name="xyLowerYHDI",
                        ),
                        "hover_glyph": Line(
                            x="x",
                            y="y",
                            line_alpha=1.0,
                            line_color="black",
                            line_width=2.0,
                        ),
                    },
                    "upper": {
                        "glyph": Line(
                            x="x",
                            y="y",
                            line_alpha=0.7,
                            line_color="black",
                            line_width=2.0,
                            name="xyUpperYHDI",
                        ),
                        "hover_glyph": Line(
                            x="x",
                            y="y",
                            line_alpha=1.0,
                            line_color="black",
                            line_width=2.0,
                        ),
                    },
                },
            },
            "stats": {
                "glyph": Circle(
                    x="x",
                    y="y",
                    size=20,
                    fill_alpha=0.5,
                    fill_color=mean_color,
                    line_color="white",
                    name="xyStats",
                ),
                "hover_glyph": Circle(
                    x="x",
                    y="y",
                    size=20,
                    fill_alpha=1.0,
                    fill_color=mean_color,
                    line_color="black",
                ),
            },
        },
    }
    return output


def add_glyphs(
    figures: typing.Figures,
    glyphs: typing.Glyphs,
    sources: typing.Sources,
) -> None:
    """
    Bind source data to glyphs and add the glyphs to the given figures.

    Args:
        figures (typing.Figures): A dictionary of Bokeh Figure objects.
        glyphs (typing.Glyphs): A dictionary of Bokeh Glyphs objects.
        sources (typing.Sources): A dictionary of Bokeh ColumnDataSource objects.

    Returns
        None: Adds data bound glyphs to the given figures directly.
    """
    # x figure
    figures["x"].add_glyph(
        source_or_glyph=sources["x"]["distribution"],
        glyph=glyphs["x"]["distribution"]["glyph"],
        hover_glyph=glyphs["x"]["distribution"]["hover_glyph"],
        name="xDistribution",
    )
    figures["x"].add_glyph(
        source_or_glyph=sources["x"]["stats"],
        glyph=glyphs["x"]["stats"]["glyph"],
        hover_glyph=glyphs["x"]["stats"]["hover_glyph"],
        name="xStats",
    )

    # y figure
    figures["y"].add_glyph(
        source_or_glyph=sources["y"]["distribution"],
        glyph=glyphs["y"]["distribution"]["glyph"],
        hover_glyph=glyphs["y"]["distribution"]["hover_glyph"],
        name="yDistribution",
    )
    figures["y"].add_glyph(
        source_or_glyph=sources["y"]["stats"],
        glyph=glyphs["y"]["stats"]["glyph"],
        hover_glyph=glyphs["y"]["stats"]["hover_glyph"],
        name="yStats",
    )

    # xy figure
    figures["xy"].add_glyph(
        source_or_glyph=sources["xy"]["distribution"],
        glyph=glyphs["xy"]["distribution"]["glyph"],
        hover_glyph=glyphs["xy"]["distribution"]["hover_glyph"],
        name="xyDistribution",
    )
    figures["xy"].add_glyph(
        source_or_glyph=sources["xy"]["hdi"]["x"]["lower"],
        glyph=glyphs["xy"]["hdi"]["x"]["lower"]["glyph"],
        hover_glyph=glyphs["xy"]["hdi"]["x"]["lower"]["hover_glyph"],
        name="xyHDIXLower",
    )
    figures["xy"].add_glyph(
        source_or_glyph=sources["xy"]["hdi"]["x"]["upper"],
        glyph=glyphs["xy"]["hdi"]["x"]["upper"]["glyph"],
        hover_glyph=glyphs["xy"]["hdi"]["x"]["upper"]["hover_glyph"],
        name="xyHDIXUpper",
    )
    figures["xy"].add_glyph(
        source_or_glyph=sources["xy"]["hdi"]["y"]["lower"],
        glyph=glyphs["xy"]["hdi"]["y"]["lower"]["glyph"],
        hover_glyph=glyphs["xy"]["hdi"]["y"]["lower"]["hover_glyph"],
        name="xyHDIYLower",
    )
    figures["xy"].add_glyph(
        source_or_glyph=sources["xy"]["hdi"]["y"]["upper"],
        glyph=glyphs["xy"]["hdi"]["y"]["upper"]["glyph"],
        hover_glyph=glyphs["xy"]["hdi"]["y"]["upper"]["hover_glyph"],
        name="xyHDIYUpper",
    )
    figures["xy"].add_glyph(
        source_or_glyph=sources["xy"]["stats"],
        glyph=glyphs["xy"]["stats"]["glyph"],
        hover_glyph=glyphs["xy"]["stats"]["hover_glyph"],
        name="xyStats",
    )


def create_annotations(sources: typing.Sources) -> typing.Annotations:
    """
    Create any annotations for the figures of the tool.

    Args:
        sources (typing.Sources): A dictionary of Bokeh ColumnDataSource objects.

    Returns:
        typing.Annotations: A dictionary of Bokeh Annotation objects.
    """
    palette = choose_palette(1)
    color = palette[0]
    output = {
        "x": Band(
            base="base",
            lower="lower",
            upper="upper",
            source=sources["x"]["hdi"],
            level="underlay",
            fill_color=color,
            fill_alpha=0.2,
            line_color=None,
            name="xHDI",
        ),
        "y": {
            "lower": Band(
                base="base",
                lower="lower",
                upper="upper",
                source=sources["y"]["hdi"]["lower"],
                level="underlay",
                fill_color=color,
                fill_alpha=0.2,
                line_color=None,
                name="yLowerHDI",
            ),
            "upper": Band(
                base="base",
                lower="lower",
                upper="upper",
                source=sources["y"]["hdi"]["upper"],
                level="underlay",
                fill_color=color,
                fill_alpha=0.2,
                line_color=None,
                name="yUpperHDI",
            ),
        },
    }
    return output


def add_annotations(figures: typing.Figures, annotations: typing.Annotations) -> None:
    """
    Add the given annotations to the given figures of the tool.

    Args:
        figures (typing.Figures): A dictionary of Bokeh Figure objects.
    annotations (typing.Annotations): A dictionary of Bokeh Annotation objects.

    Returns:
        None: Adds annotations directly to the given figures.
    """
    figures["x"].add_layout(annotations["x"])
    figures["y"].add_layout(annotations["y"]["lower"])
    figures["y"].add_layout(annotations["y"]["upper"])


def create_tooltips(
    rv_name_x: str,
    rv_name_y: str,
    figures: typing.Figures,
) -> typing.Tooltips:
    """
    Create hover tools for the glyphs used in the figures of the tool.

    Args:
        rv_name_x (str): The name of the random variable data in the x-direction.
        rv_name_y (str): The name of the random variable data in the y-direction.
        figures (typing.Figures): A dictionary of Bokeh Figure objects.

    Returns:
        typing.Tooltips: A dictionary of Bokeh HoverTools objects.
    """
    x_dist = filter_renderers(figure=figures["x"], search="xDistribution")
    x_stats = filter_renderers(figure=figures["x"], search="xStats")
    y_dist = filter_renderers(figure=figures["y"], search="yDistribution")
    y_stats = filter_renderers(figure=figures["y"], search="yStats")
    xy_dist = filter_renderers(figure=figures["xy"], search="xyDistribution")
    xy_lower_x_hdi = filter_renderers(figure=figures["xy"], search="xyLowerXHDI")
    xy_upper_x_hdi = filter_renderers(figure=figures["xy"], search="xyUpperXHDI")
    xy_lower_y_hdi = filter_renderers(figure=figures["xy"], search="xyLowerYHDI")
    xy_upper_y_hdi = filter_renderers(figure=figures["xy"], search="xyUpperYHDI")
    xy_stats = filter_renderers(figure=figures["xy"], search="xyStats")
    output = {
        "x": {
            "distribution": HoverTool(renderers=x_dist, tooltips=[(rv_name_x, "@x")]),
            "stats": HoverTool(renderers=x_stats, tooltips=[("", "@text")]),
        },
        "y": {
            "distribution": HoverTool(renderers=y_dist, tooltips=[(rv_name_y, "@y")]),
            "stats": HoverTool(renderers=y_stats, tooltips=[("", "@text")]),
        },
        "xy": {
            "distribution": HoverTool(
                renderers=xy_dist,
                tooltips=[(rv_name_x, "@x"), (rv_name_y, "@y")],
            ),
            "hdi": {
                "x": {
                    "lower": HoverTool(
                        renderers=xy_lower_x_hdi,
                        tooltips=[(rv_name_x, "@x")],
                    ),
                    "upper": HoverTool(
                        renderers=xy_upper_x_hdi,
                        tooltips=[(rv_name_x, "@x")],
                    ),
                },
                "y": {
                    "lower": HoverTool(
                        renderers=xy_lower_y_hdi,
                        tooltips=[(rv_name_y, "@y")],
                    ),
                    "upper": HoverTool(
                        renderers=xy_upper_y_hdi,
                        tooltips=[(rv_name_y, "@y")],
                    ),
                },
            },
            "stats": HoverTool(
                renderers=xy_stats,
                tooltips=[(rv_name_x, "@x"), (rv_name_y, "@y")],
            ),
        },
    }
    return output


def add_tooltips(figures: typing.Figures, tooltips: typing.Tooltips) -> None:
    """
    Add the given tools to the figures.

    Args:
        figures (typing.Figures): A dictionary of Bokeh Figure objects.
        tooltips (typing.Tooltips): A dictionary of Bokeh HoverTools objects.

    Returns:
        None: Adds the tooltips directly to the given figures.
    """
    figures["x"].add_tools(tooltips["x"]["distribution"])
    figures["x"].add_tools(tooltips["x"]["stats"])
    figures["y"].add_tools(tooltips["y"]["distribution"])
    figures["y"].add_tools(tooltips["y"]["stats"])
    figures["xy"].add_tools(tooltips["xy"]["distribution"])
    figures["xy"].add_tools(tooltips["xy"]["stats"])
    figures["xy"].add_tools(tooltips["xy"]["hdi"]["x"]["lower"])
    figures["xy"].add_tools(tooltips["xy"]["hdi"]["x"]["upper"])
    figures["xy"].add_tools(tooltips["xy"]["hdi"]["y"]["lower"])
    figures["xy"].add_tools(tooltips["xy"]["hdi"]["y"]["upper"])


def create_widgets(
    rv_name_x: str,
    rv_name_y: str,
    rv_names: List[str],
    bw_factor: float,
    bandwidth_x: float,
    bandwidth_y: float,
) -> typing.Widgets:
    """
    Create the widgets used in the tool.

    Args:
        rv_name_x (str): The name of the random variable along the x-axis.
        rv_name_y (str): The name of the random variable along the y-axis.
        rv_names (List[str]): A list of all available random variable names.
        bw_factor (float): Multiplicative factor used when calculating the kernel
            density estimate.
        bandwidth_x (float): The bandwidth used to calculate the KDE along the x-axis.
        bandwidth_y (float): The bandwidth used to calculate the KDE along the y-axis.

    Returns:
        typing.Widgets: A dictionary of Bokeh widget objects.
    """
    output = {
        "rv_select_x": Select(value=rv_name_x, options=rv_names, title="Query (x)"),
        "rv_select_y": Select(value=rv_name_y, options=rv_names, title="Query (y)"),
        "bw_factor_slider": Slider(
            title="Bandwidth factor",
            start=0.01,
            end=2.00,
            value=1.00,
            step=0.01,
        ),
        "hdi_slider_x": Slider(start=1, end=99, step=1, value=89, title="HDI (x)"),
        "hdi_slider_y": Slider(start=1, end=99, step=1, value=89, title="HDI (y)"),
        "bw_div_x": Div(text=f"Bandwidth {rv_name_x}: {bw_factor * bandwidth_x}"),
        "bw_div_y": Div(text=f"Bandwidth {rv_name_y}: {bw_factor * bandwidth_y}"),
    }
    return output


def help_page() -> Div:
    """
    Help tab for the tool.

    Returns:
        Div: Bokeh Div widget containing the help tab information.
    """
    text = """
    <h2>
      Joint plot
    </h2>
    <p style="margin-bottom: 10px">
      A joint plot shows univariate marginals along the x and y axes. The
      central figure shows the bivariate marginal of both random variables.
    </p>
    """
    output = Div(
        text=text,
        disable_math=False,
        min_width=MARGINAL1D_PLOT_WIDTH + MARGINAL2D_PLOT_WIDTH,
    )
    return output


def create_figure_grid(figures: typing.Figures) -> Row:
    """Layout the given figures in a grid, and make one toolbar.

    Parameters
    ----------
    figures : typing.Figures
        A dictionary of Bokeh Figure objects.

    Returns
    -------
    Row
        A Bokeh layout object.
    """
    toolbar = create_toolbar(list(figures.values()))
    grid_box = GridBox(
        children=[
            [figures["x"], 0, 0],
            [figures["xy"], 1, 0],
            [figures["y"], 1, 1],
        ],
    )
    return Row(children=[grid_box, toolbar])


def create_view(figures: typing.Figures, widgets: typing.Widgets) -> Tabs:
    """
    Create the tool view.

    Args:
        figures (typing.Figures): A dictionary of Bokeh Figure objects.
        widgets (typing.Widgets): A dictionary of Bokeh widget objects.

    Returns:
        Tabs: Bokeh Tabs objects.
    """
    help_panel = Panel(child=help_page(), title="Help", name="helpPanel")
    figure_grid = create_figure_grid(figures)
    tool_panel = Panel(
        child=Column(
            children=[
                Row(children=[widgets["rv_select_x"], widgets["rv_select_y"]]),
                Row(
                    children=[
                        figure_grid,
                        Column(
                            children=[
                                widgets["bw_factor_slider"],
                                widgets["hdi_slider_x"],
                                widgets["hdi_slider_y"],
                                widgets["bw_div_x"],
                                widgets["bw_div_y"],
                            ]
                        ),
                    ],
                    css_classes=["bm-tool-loading", "arcs"],
                ),
            ],
        ),
        title="Marginal 2D",
        name="toolPanel",
    )
    return Tabs(tabs=[tool_panel, help_panel])
