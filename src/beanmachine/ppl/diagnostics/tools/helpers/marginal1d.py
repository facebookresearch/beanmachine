# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Methods used to generate the diagnostic tool."""
from typing import List, Optional

import arviz as az

import beanmachine.ppl.diagnostics.tools.typing.marginal1d as typing
import numpy as np
import numpy.typing as npt
from beanmachine.ppl.diagnostics.tools.helpers.plotting import (
    choose_palette,
    create_toolbar,
    filter_renderers,
    style_figure,
)
from bokeh.models.annotations import Band, LabelSet
from bokeh.models.glyphs import Circle, Line
from bokeh.models.layouts import Column, Row
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.markups import Div
from bokeh.models.widgets.panels import Panel, Tabs
from bokeh.models.widgets.sliders import Slider
from bokeh.plotting.figure import figure


PLOT_WIDTH = 500
PLOT_HEIGHT = 500
FIGURE_NAMES = ["marginal", "cumulative"]


def compute_stats(
    data: npt.NDArray,
    kde_x: npt.NDArray,
    kde_y: npt.NDArray,
    hdi_probability: float,
    text_align: Optional[List[str]] = None,
    x_offset: Optional[List[int]] = None,
    y_offset: Optional[List[int]] = None,
    return_labels: bool = False,
) -> typing.StatsAndLabelsData:
    """Compute statistics for the given data, and its KDE.

    Parameters
    ----------
    data : npt.NDArray
        A 2D NumPy array where the length of the first dimension is the number of chains
        of the model, and the length of the second dimension is the number of draws of
        the model.
    kde_x : npt.NDArray
        The x-axis KDE estimate of the random variable data.
    kde_y : npt.NDArray
        The y-axis KDE estimate of the random variable data.
    hdi_probability : float
        The HDI probability to use when calculating the HDI bounds.
    text_align : List[str] | None, optional default is ``None``
        How to display label justifications for the statistics in Bokeh.
    x_offset : List[int] | None, optional default is ``None``
        x-axis offsets for the labels.
    y_offset : List[int] | None, optional default is ``None``
        y-axis offsets for the labels.
    return_labels : bool, optional default is ``False``
        ``True`` returns labels to be used in the Bokeh figure.

    Returns
    -------
    typing.StatsAndLabelsData
        A dictionary for statistics and labels for the Bokeh figures.
    """
    if text_align is None:
        text_align = ["right", "center", "left"]
    if x_offset is None:
        x_offset = [-5, 0, 5]
    if y_offset is None:
        y_offset = [0, 10, 0]

    mean = np.mean(kde_x)
    hdi = az.stats.hdi(data, hdi_prob=hdi_probability)
    x = [hdi[0], mean, hdi[1]]
    y = np.interp(x=np.array(x), xp=kde_x, fp=kde_y)
    text = [
        f"Lower HDI: {hdi[0]:.4}",
        f"Mean: {mean:.4}",
        f"Upper HDI: {hdi[1]:.4}",
    ]
    stats = {"x": x, "y": y.tolist(), "text": text}
    labels = {}
    if return_labels:
        labels = {
            "x": x,
            "y": y.tolist(),
            "text": text,
            "text_align": text_align,
            "x_offset": x_offset,
            "y_offset": y_offset,
        }
    return {"stats": stats, "labels": labels}


def hdi_data(
    data: npt.NDArray,
    kde_x: npt.NDArray,
    kde_y: npt.NDArray,
    hdi_probability: float,
) -> typing.HDIData:
    """Calculate the HDI arrays needed for the Bokeh annotation.

    Parameters
    ----------
    data : npt.NDArray
        A 2D NumPy array where the length of the first dimension is the number of chains
        of the model, and the length of the second dimension is the number of draws of
        the model.
    kde_x : npt.NDArray
        The x-axis KDE estimate of the random variable data.
    kde_y : npt.NDArray
        The y-axis KDE estimate of the random variable data.
    hdi_probability : float
        The HDI probability to use when calculating the HDI bounds.

    Returns
    -------
    typing.HDIData
        Arrays used to create the Bokeh annotation for the HDI.
    """
    hdi = az.stats.hdi(data, hdi_prob=hdi_probability)
    mask = np.ix_(np.logical_and(kde_x >= hdi[0], kde_x <= hdi[1]))[0]
    base = kde_x[mask]
    lower = np.zeros((len(mask)))
    upper = kde_y[mask]
    return {"base": base.tolist(), "lower": lower.tolist(), "upper": upper.tolist()}


def compute_data(
    data: npt.NDArray,
    bw_factor: float,
    hdi_probability: float,
) -> typing.Data:
    """Compute effective sample size estimates using the given data.

    Parameters
    ----------
    data : npt.NDArray
        A 2D NumPy array where the length of the first dimension is the number of chains
        of the model, and the length of the second dimension is the number of draws of
        the model.
    bw_factor : float
        Multiplicative factor used when calculating the kernel density estimate.
    hdi_probability : float
        The HDI probability to use when calculating the HDI bounds.

    Returns
    -------
    typing.Data
        A dictionary of data used for the tool.
    """
    data = data.reshape(-1)
    output = {}
    for figure_name in FIGURE_NAMES:
        output[figure_name] = {}
        x, y, bandwidth = az.stats.density_utils._kde_linear(
            data,
            bw_return=True,
            bw_fct=bw_factor,
        )
        if figure_name == "cumulative":
            y = np.cumsum(y)
        y /= y.max()
        stats_and_labels = compute_stats(
            data,
            x,
            y,
            hdi_probability,
            return_labels=True,
        )
        output[figure_name] = {
            "distribution": {
                "x": x.tolist(),
                "y": y.tolist(),
                "bandwidth": [bandwidth],
            },
            "hdi": hdi_data(data, x, y, hdi_probability),
            "stats": stats_and_labels["stats"],
            "labels": stats_and_labels["labels"],
        }
    return output


def create_sources(data: typing.Data) -> typing.Sources:
    """Create Bokeh sources from the given data that will be bound to glyphs.

    Parameters
    ----------
    data : typing.Data
        A dictionary of data used for the tool.

    Returns
    -------
    typing.Sources
        A dictionary of Bokeh ColumnDataSource objects.
    """
    output = {}
    for figure_name, figure_data in data.items():
        output[figure_name] = {}
        for glyph_name, glyph_data in figure_data.items():
            if "bandwidth" in list(glyph_data.keys()):
                glyph_data.pop("bandwidth")
            output[figure_name][glyph_name] = ColumnDataSource(data=glyph_data)
    return output


def create_figures(rv_name: str) -> typing.Figures:
    """Create the Bokeh figures used for the tool.

    Parameters
    ----------
    rv_name : str
        The string representation of the random variable data.

    Returns
    -------
    typing.Figures
        A dictionary of Bokeh Figure objects.
    """
    output = {}
    for figure_name in FIGURE_NAMES:
        fig = figure(
            width=PLOT_WIDTH,
            height=PLOT_HEIGHT,
            outline_line_color="black",
            title=f"{figure_name} distribution",
            x_axis_label=rv_name,
            y_axis_label=None,
        )
        fig.yaxis.visible = False
        style_figure(fig)
        output[figure_name] = fig
    output[FIGURE_NAMES[0]].x_range = output[FIGURE_NAMES[1]].x_range
    output[FIGURE_NAMES[0]].y_range = output[FIGURE_NAMES[1]].y_range
    return output


def create_glyphs(data: typing.Data) -> typing.Glyphs:
    """Create the glyphs used for the figures of the tool.

    Parameters
    ----------
    data : typing.Data
        A dictionary of data used for the tool.

    Returns
    -------
    typing.Glyphs
        A dictionary of Bokeh Glyphs objects.
    """
    palette = choose_palette(2)
    output = {}
    for figure_name, figure_data in data.items():
        output[figure_name] = {}
        for glyph_name, _ in figure_data.items():
            if glyph_name in ["distribution", "stats"]:
                if glyph_name == "distribution":
                    output[figure_name][glyph_name] = {
                        "glyph": Line(
                            x="x",
                            y="y",
                            line_color=palette[0],
                            line_alpha=0.7,
                            line_width=2.0,
                            name=f"{figure_name}DistributionGlyph",
                        ),
                        "hover_glyph": Line(
                            x="x",
                            y="y",
                            line_color=palette[1],
                            line_alpha=1.0,
                            line_width=2.0,
                            name=f"{figure_name}DistributionHoverGlyph",
                        ),
                    }
                if glyph_name == "stats":
                    output[figure_name][glyph_name] = {
                        "glyph": Circle(
                            x="x",
                            y="y",
                            size=10,
                            fill_color=palette[0],
                            line_color="white",
                            fill_alpha=1.0,
                            name=f"{figure_name}StatsGlyph",
                        ),
                        "hover_glyph": Circle(
                            x="x",
                            y="y",
                            size=10,
                            fill_color=palette[1],
                            line_color="black",
                            fill_alpha=1.0,
                            name=f"{figure_name}StatsHoverGlyph",
                        ),
                    }
    return output


def add_glyphs(
    figures: typing.Figures,
    glyphs: typing.Glyphs,
    sources: typing.Sources,
) -> None:
    """Bind source data to glyphs and add the glyphs to the given figures.

    Parameters
    ----------
    figures : typing.Figures
        A dictionary of Bokeh Figure objects.
    glyphs : typing.Glyphs
        A dictionary of Bokeh Glyphs objects.
    sources : typing.Sources
        A dictionary of Bokeh ColumnDataSource objects.

    Returns
    -------
    None
        Adds data bound glyphs to the given figures directly.
    """
    for figure_name, figure_glyphs in glyphs.items():
        fig = figures[figure_name]
        figure_sources = sources[figure_name]
        for glyph_name, glyphs in figure_glyphs.items():
            glyph_source = figure_sources[glyph_name]
            fig.add_glyph(
                source_or_glyph=glyph_source,
                glyph=glyphs["glyph"],
                hover_glyph=glyphs["hover_glyph"],
                name=glyphs["glyph"].name,
            )


def create_annotations(sources: typing.Sources) -> typing.Annotations:
    """Create any annotations for the figures of the tool.

    Parameters
    ----------
    source : typing.Sources
        A dictionary of Bokeh ColumnDataSource objects.

    Returns
    -------
    typing.Annotations
        A dictionary of Bokeh Annotation objects.
    """
    palette = choose_palette(1)
    output = {}
    for figure_name, figure_sources in sources.items():
        output[figure_name] = {}
        for glyph_name, glyph_source in figure_sources.items():
            if glyph_name == "hdi":
                output[figure_name][glyph_name] = Band(
                    base="base",
                    lower="lower",
                    upper="upper",
                    source=glyph_source,
                    level="underlay",
                    fill_color=palette[0],
                    fill_alpha=0.2,
                    line_width=1.0,
                    line_color="white",
                    name=f"{figure_name}HdiAnnotation",
                )
            elif glyph_name == "labels":
                output[figure_name][glyph_name] = LabelSet(
                    x="x",
                    y="y",
                    text="text",
                    x_offset="x_offset",
                    y_offset="y_offset",
                    text_align="text_align",
                    source=glyph_source,
                    background_fill_color="white",
                    background_fill_alpha=0.8,
                    name=f"{figure_name}LabelAnnotation",
                )
    return output


def add_annotations(figures: typing.Figures, annotations: typing.Annotations) -> None:
    """Add the given annotations to the given figures of the tool.

    Parameters
    ----------
    figures : typing.Figures
        A dictionary of Bokeh Figure objects.
    annotations : typing.Annotations
        A dictionary of Bokeh Annotation objects.

    Returns
    -------
    None
        Adds annotations directly to the given figures.
    """
    for figure_name, annotation_sources in annotations.items():
        fig = figures[figure_name]
        for _, annotation in annotation_sources.items():
            fig.add_layout(annotation)


def create_tooltips(rv_name: str, figures: typing.Figures) -> typing.Tooltips:
    """Create hover tools for the glyphs used in the figures of the tool.

    Parameters
    ----------
    rv_name : str
        The string representation of the random variable data.
    figures : typing.Figures
        A dictionary of Bokeh Figure objects.

    Returns
    -------
    typing.Tooltips
        A dictionary of Bokeh HoverTools objects.
    """
    output = {}
    for figure_name, fig in figures.items():
        output[figure_name] = {
            "distribution": HoverTool(
                renderers=filter_renderers(
                    fig,
                    "DistributionGlyph",
                    "GlyphRenderer",
                    True,
                ),
                tooltips=[(rv_name, "@x")],
            ),
            "stats": HoverTool(
                renderers=filter_renderers(fig, "StatsGlyph", "GlyphRenderer", True),
                tooltips=[("", "@text")],
            ),
        }
    return output


def add_tooltips(figures: typing.Figures, tooltips: typing.Tooltips) -> None:
    """Add the given tools to the figures.

    Parameters
    ----------
    figures : typing.Figures
        A dictionary of Bokeh Figure objects.
    tooltips : typing.Tooltips
        A dictionary of Bokeh HoverTools objects.

    Returns
    -------
    None
        Adds the tooltips directly to the given figures.
    """
    for figure_name, figure_tooltips in tooltips.items():
        fig = figures[figure_name]
        for _, tooltip in figure_tooltips.items():
            fig.add_tools(tooltip)


def create_widgets(
    rv_name: str,
    rv_names: List[str],
    bw_factor: float,
    bandwidth: float,
) -> typing.Widgets:
    """Create the widgets used in the tool.

    Parameters
    ----------
    rv_name : str
        The string representation of the random variable data.
    rv_names : List[str]
        A list of all available random variable names.
    bw_factor : float
        Multiplicative factor used when calculating the kernel density estimate.
    bandwidth : float
        The bandwidth used to calculate the KDE.

    Returns
    -------
    typing.Widgets
        A dictionary of Bokeh widget objects.
    """
    return {
        "rv_select": Select(value=rv_name, options=rv_names, title="Query"),
        "bw_factor_slider": Slider(
            title="Bandwidth factor",
            start=0.01,
            end=2.00,
            value=1.00,
            step=0.01,
        ),
        "bw_div": Div(text=f"Bandwidth: {bw_factor * bandwidth}"),
        "hdi_slider": Slider(start=1, end=99, step=1, value=89, title="HDI"),
    }


def help_page() -> Div:
    """Help tab for the tool.

    Returns
    -------
    Div
        Bokeh Div widget containing the help tab information.
    """
    text = """
    <h2>
      Highest density interval
    </h2>
    <p style="margin-bottom: 10px">
      The highest density interval region is not equal tailed like a typical
      equal tailed interval of 2.5%. Thus it will include the mode(s) of the
      posterior distribution.
    </p>
    <p style="margin-bottom: 10px">
      There is nothing particularly specific about having a default HDI of 89%.
      If fact, the only remarkable thing about defaulting to 89% is that it is
      the highest prime number that does not exceed the unstable 95% threshold.
      See the link to McElreath's book below for further discussion.
    </p>
    <ul>
      <li>
        McElreath R (2020)
        <b>
          Statistical Rethinking: A Bayesian Course with Examples in R and Stan
          2nd edition.
        </b>
        <em>Chapman and Hall/CRC</em>
        <a href=https://dx.doi.org/10.1201/9780429029608 style="color: blue">
          doi: 10.1201/9780429029608
        </a>.
      </li>
    </ul>
    """
    return Div(text=text, disable_math=False, min_width=PLOT_WIDTH)


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
    toolbar = create_toolbar(figures)
    return Row(children=[*list(figures.values()), toolbar])


def create_view(widgets: typing.Widgets, figures: typing.Figures) -> Tabs:
    """Create the tool view.

    Parameters
    ----------
    widgets : typing.Widgets
        A dictionary of Bokeh widget objects.
    figures : typing.Figures
        A dictionary of Bokeh Figure objects.

    Returns
    -------
    Tabs
        Bokeh Tabs objects.
    """
    help_panel = Panel(child=help_page(), title="Help", name="helpPanel")
    tool_panel = Panel(
        child=Column(
            children=[
                widgets["rv_select"],
                create_figure_grid(figures),
                widgets["bw_factor_slider"],
                widgets["bw_div"],
                widgets["hdi_slider"],
            ],
        ),
        title="Marginal 1D",
        name="toolPanel",
    )
    return Tabs(tabs=[tool_panel, help_panel])


def update(
    data: npt.NDArray,
    sources: typing.Sources,
    figures: typing.Figures,
    rv_name: str,
    bw_factor: float,
    hdi_probability: float,
) -> float:
    """Update the tool based on user interaction.

    Parameters
    ----------
    data : npt.NDArray
        A 2D NumPy array where the length of the first dimension is the number of chains
        of the model, and the length of the second dimension is the number of draws of
        the model.
    sources : typing.Sources
        A dictionary of Bokeh ColumnDataSource objects.
    figures : typing.Figures
        A dictionary of Bokeh Figure objects.
    rv_name : str
        The string representation of the random variable data.
    rv_names : List[str]
        A list of all available random variable names.
    hdi_probability : float
        The HDI probability to use when calculating the HDI bounds.

    Returns
    -------
    float
        Returns the bandwidth used to calculate the KDE.
    """
    computed_data = compute_data(data, bw_factor, hdi_probability)
    bw = computed_data["marginal"]["distribution"]["bandwidth"][0]
    for figure_name, figure_sources in sources.items():
        fig = figures[figure_name]
        fig.xaxis.axis_label = rv_name
        figure_data = computed_data[figure_name]
        for glyph_name, glyph_source in figure_sources.items():
            glyph_data = figure_data[glyph_name]
            if "bandwidth" in list(glyph_data.keys()):
                glyph_data.pop("bandwidth")
            glyph_source.data = dict(**glyph_data)
    return float(bw)
