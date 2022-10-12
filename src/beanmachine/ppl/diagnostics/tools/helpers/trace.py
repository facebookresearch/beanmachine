# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Methods used to generate the diagnostic tool."""
from typing import List

import arviz as az
import beanmachine.ppl.diagnostics.tools.typing.trace as typing
import numpy as np
import numpy.typing as npt

from beanmachine.ppl.diagnostics.tools.helpers import marginal1d as m1d
from beanmachine.ppl.diagnostics.tools.helpers.plotting import (
    choose_palette,
    create_toolbar,
    filter_renderers,
    style_figure,
)
from bokeh.core.property.wrappers import PropertyValueList
from bokeh.models.annotations import Legend, LegendItem
from bokeh.models.glyphs import Circle, Line, Quad
from bokeh.models.layouts import Column, Row

from bokeh.models.ranges import Range1d
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets import Div
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.panels import Panel, Tabs
from bokeh.models.widgets.sliders import Slider
from bokeh.plotting.figure import figure


PLOT_WIDTH = 400
PLOT_HEIGHT = 500
TRACE_PLOT_WIDTH = 600
FIGURE_NAMES = ["marginals", "forests", "traces", "ranks"]


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
    num_chains, num_draws = data.shape
    num_samples = num_chains * num_draws
    rank_data = az.plots.plot_utils.compute_ranks(data)
    n_bins = int(np.ceil(2 * np.log2(rank_data.shape[1])) + 1)
    bins = np.histogram_bin_edges(rank_data, bins=n_bins, range=(0, rank_data.size))
    hist_bins = len(bins)
    output = {}
    for figure_name in FIGURE_NAMES:
        output[figure_name] = {}
        for chain in range(num_chains):
            chain_index = chain + 1
            chain_name = f"chain{chain_index}"
            output[figure_name][chain_name] = {}
            chain_data = data[chain]
            marginal = m1d.compute_data(chain_data, bw_factor, hdi_probability)
            marginal = marginal["marginal"]["distribution"]
            mean = float(np.array(marginal["x"]).mean())
            if figure_name == "marginals":
                output[figure_name][chain_name] = {
                    "line": {"x": marginal["x"], "y": marginal["y"]},
                    "chain": chain_index,
                    "mean": mean,
                    "bandwidth": marginal["bandwidth"],
                }
            if figure_name == "forests":
                hdi = az.stats.hdi(chain_data, hdi_probability)
                output[figure_name][chain_name] = {
                    "line": {"x": hdi, "y": [chain_index] * 2},
                    "circle": {"x": [mean], "y": [chain_index]},
                    "chain": chain_index,
                    "mean": mean,
                }
            if figure_name == "traces":
                output[figure_name][chain_name] = {
                    "line": {"x": np.arange(0, num_draws, 1), "y": chain_data},
                    "chain": chain_index,
                    "mean": mean,
                    "bandwidth": marginal["bandwidth"],
                }
            if figure_name == "ranks":
                _, histogram, _ = az.stats.density_utils.histogram(
                    rank_data[chain, :],
                    bins=n_bins,
                )
                normed_hist = histogram / histogram.max()
                chain_rank_mean = normed_hist.mean()
                left = bins[:-1]
                top = normed_hist + chain
                right = bins[1:]
                bottom = np.zeros(hist_bins - 1) + chain
                draws = [f"{int(b[0]):0,}-{int(b[1]):0,}" for b in zip(left, right)]
                quad = {
                    "left": left.tolist(),
                    "right": right.tolist(),
                    "top": top.tolist(),
                    "bottom": bottom.tolist(),
                    "draws": draws,
                    "rank": normed_hist.tolist(),
                }
                x = np.arange(0, num_samples, 1)
                line = {"x": x.tolist(), "y": [chain_rank_mean + chain] * len(x)}
                output[figure_name][chain_name] = {
                    "quad": quad,
                    "line": line,
                    "chain": chain_index,
                    "rank_mean": chain_index - chain_rank_mean,
                    "mean": histogram.mean(),
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
        for i, (chain_name, chain_data) in enumerate(figure_data.items()):
            output[figure_name][chain_name] = {}
            n = len(chain_data["line"]["x"])
            chain_index_list = [chain_data["chain"]] * n
            chain_mean_list = [chain_data["mean"]] * n
            if figure_name == "marginals":
                output[figure_name][chain_name]["line"] = ColumnDataSource(
                    {
                        "x": chain_data["line"]["x"],
                        "y": chain_data["line"]["y"],
                        "chain": chain_index_list,
                        "mean": chain_mean_list,
                    },
                )
            if figure_name == "forests":
                output[figure_name][chain_name]["line"] = ColumnDataSource(
                    {
                        "x": chain_data["line"]["x"],
                        "y": chain_data["line"]["y"],
                        "chain": chain_index_list,
                    },
                )
                output[figure_name][chain_name]["circle"] = ColumnDataSource(
                    {
                        "x": chain_data["circle"]["x"],
                        "y": chain_data["circle"]["y"],
                        "chain": [chain_data["chain"]],
                    },
                )
            if figure_name == "traces":
                output[figure_name][chain_name]["line"] = ColumnDataSource(
                    {
                        "x": chain_data["line"]["x"],
                        "y": chain_data["line"]["y"],
                        "chain": chain_index_list,
                        "mean": ([chain_data["mean"]] * len(chain_data["line"]["x"])),
                    },
                )
            if figure_name == "ranks":
                output[figure_name][chain_name]["line"] = ColumnDataSource(
                    {
                        "x": chain_data["line"]["x"],
                        "y": chain_data["line"]["y"],
                        "chain": chain_index_list,
                        "rank_mean": (
                            [chain_data["rank_mean"]] * len(chain_data["line"]["x"])
                        ),
                    },
                )
                output[figure_name][chain_name]["quad"] = ColumnDataSource(
                    {
                        "left": chain_data["quad"]["left"],
                        "top": chain_data["quad"]["top"],
                        "right": chain_data["quad"]["right"],
                        "bottom": chain_data["quad"]["bottom"],
                        "chain": [i + 1] * len(chain_data["quad"]["left"]),
                        "draws": chain_data["quad"]["draws"],
                        "rank": chain_data["quad"]["rank"],
                    },
                )
    return output


def create_figures(rv_name: str, num_chains: int) -> typing.Figures:
    """Create the Bokeh figures used for the tool.

    Parameters
    ----------
    rv_name : str
        The string representation of the random variable data.
    num_chains : int
        The number of chains of the model.

    Returns
    -------
    typing.Figures
        A dictionary of Bokeh Figure objects.
    """
    output = {}
    for figure_name in FIGURE_NAMES:
        fig = figure()
        if figure_name == "marginals":
            fig = figure(
                plot_width=PLOT_WIDTH,
                plot_height=PLOT_HEIGHT,
                outline_line_color="black",
                title="Marginal",
                x_axis_label=rv_name,
                x_range=Range1d(),
            )
            style_figure(fig)
        if figure_name == "forests":
            fig = figure(
                plot_width=PLOT_WIDTH,
                plot_height=PLOT_HEIGHT,
                outline_line_color="black",
                title="Forest",
                x_axis_label=rv_name,
                y_axis_label="Chain",
                x_range=Range1d(),
            )
            style_figure(fig)
            fig.yaxis.minor_tick_line_color = None
            fig.yaxis.ticker = list(range(1, num_chains + 1))
        if figure_name == "traces":
            fig = figure(
                plot_width=TRACE_PLOT_WIDTH,
                plot_height=PLOT_HEIGHT,
                outline_line_color="black",
                title="Trace",
                x_axis_label="Draw from single chain",
                y_axis_label=rv_name,
                y_range=Range1d(),
            )
            style_figure(fig)
            fig.yaxis.minor_tick_line_color = "grey"
        if figure_name == "ranks":
            fig = figure(
                plot_width=TRACE_PLOT_WIDTH,
                plot_height=PLOT_HEIGHT,
                outline_line_color="black",
                title="Rank",
                x_axis_label="Rank from all chains",
                y_axis_label="Chain",
            )
            style_figure(fig)
            fig.yaxis.minor_tick_line_color = None
            fig.yaxis.ticker = list(range(1, num_chains + 1))
        output[figure_name] = fig
    return output


def create_glyphs(data: typing.Data, num_chains: int) -> typing.Glyphs:
    """Create the glyphs used for the figures of the tool.

    Parameters
    ----------
    data : typing.Data
        A dictionary of data used for the tool.
    num_chains : int
        The number of chains of the model.

    Returns
    -------
    typing.Glyphs
        A dictionary of Bokeh Glyphs objects.
    """
    palette = choose_palette(num_chains)
    output = {}
    for figure_name, figure_data in data.items():
        output[figure_name] = {}
        for i, (chain_name, _) in enumerate(figure_data.items()):
            output[figure_name][chain_name] = {}
            color = palette[i]
            if figure_name == "marginals":
                output[figure_name][chain_name]["line"] = {
                    "glyph": Line(
                        x="x",
                        y="y",
                        line_color=color,
                        line_alpha=0.7,
                        line_width=2.0,
                        name=f"{figure_name}{chain_name.title()}LineGlyph",
                    ),
                    "hover_glyph": Line(
                        x="x",
                        y="y",
                        line_color=color,
                        line_alpha=1.0,
                        line_width=2.0,
                        name=f"{figure_name}{chain_name.title()}LineHoverGlyph",
                    ),
                }
            if figure_name == "forests":
                output[figure_name][chain_name] = {
                    "line": {
                        "glyph": Line(
                            x="x",
                            y="y",
                            line_color=color,
                            line_alpha=0.7,
                            line_width=2.0,
                            name=f"{figure_name}{chain_name.title()}LineGlyph",
                        ),
                        "hover_glyph": Line(
                            x="x",
                            y="y",
                            line_color=color,
                            line_alpha=1.0,
                            line_width=2.0,
                            name=f"{figure_name}{chain_name.title()}LineHoverGlyph",
                        ),
                    },
                    "circle": {
                        "glyph": Circle(
                            x="x",
                            y="y",
                            size=10,
                            fill_color=color,
                            fill_alpha=0.7,
                            line_color="white",
                            name=f"{figure_name}{chain_name.title()}CircleGlyph",
                        ),
                        "hover_glyph": Circle(
                            x="x",
                            y="y",
                            size=10,
                            fill_color=color,
                            fill_alpha=1.0,
                            line_color="black",
                            name=f"{figure_name}{chain_name.title()}CircleHoverGlyph",
                        ),
                    },
                }
            if figure_name == "traces":
                output[figure_name][chain_name]["line"] = {
                    "glyph": Line(
                        x="x",
                        y="y",
                        line_color=color,
                        line_alpha=0.6,
                        line_width=0.6,
                        name=f"{figure_name}{chain_name.title()}LineGlyph",
                    ),
                    "hover_glyph": Line(
                        x="x",
                        y="y",
                        line_color=color,
                        line_alpha=0.6,
                        line_width=1.0,
                        name=f"{figure_name}{chain_name.title()}LineHoverGlyph",
                    ),
                }
            if figure_name == "ranks":
                output[figure_name][chain_name] = {
                    "quad": {
                        "glyph": Quad(
                            left="left",
                            top="top",
                            right="right",
                            bottom="bottom",
                            fill_color=color,
                            fill_alpha=0.7,
                            line_color="white",
                            name=f"{figure_name}{chain_name.title()}QuadGlyph",
                        ),
                        "hover_glyph": Quad(
                            left="left",
                            top="top",
                            right="right",
                            bottom="bottom",
                            fill_color=color,
                            fill_alpha=1.0,
                            line_color="black",
                            name=f"{figure_name}{chain_name.title()}QuadHoverGlyph",
                        ),
                    },
                    "line": {
                        "glyph": Line(
                            x="x",
                            y="y",
                            line_color="grey",
                            line_alpha=0.7,
                            line_width=3.0,
                            line_dash="dashed",
                            name=f"{figure_name}{chain_name.title()}LineGlyph",
                        ),
                        "hover_glyph": Line(
                            x="x",
                            y="y",
                            line_color="grey",
                            line_alpha=1.0,
                            line_width=3.0,
                            line_dash="solid",
                            name=f"{figure_name}{chain_name.title()}LineGlyph",
                        ),
                    },
                }
    return output


def add_glyphs(
    figures: typing.Figures,
    glyphs: typing.Glyphs,
    sources: typing.Sources,
    num_chains: int,
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
    num_chains : int
        The number of chains of the model.

    Returns
    -------
    None
        Adds data bound glyphs to the given figures directly.
    """
    range_min = []
    range_max = []
    for figure_name, figure_sources in sources.items():
        fig = figures[figure_name]
        for chain_name, source in figure_sources.items():
            chain_glyphs = glyphs[figure_name][chain_name]
            fig.add_glyph(
                source_or_glyph=source["line"],
                glyph=chain_glyphs["line"]["glyph"],
                hover_glyph=chain_glyphs["line"]["hover_glyph"],
                name=chain_glyphs["line"]["glyph"].name,
            )
            if figure_name == "marginals":
                range_min.append(min(source["line"].data["x"]))
                range_max.append(max(source["line"].data["x"]))
            if figure_name == "forests":
                fig.add_glyph(
                    source_or_glyph=source["circle"],
                    glyph=chain_glyphs["circle"]["glyph"],
                    hover_glyph=chain_glyphs["circle"]["hover_glyph"],
                    name=chain_glyphs["circle"]["glyph"].name,
                )
            if figure_name == "ranks":
                fig.add_glyph(
                    source_or_glyph=source["quad"],
                    glyph=chain_glyphs["quad"]["glyph"],
                    hover_glyph=chain_glyphs["quad"]["hover_glyph"],
                    name=chain_glyphs["quad"]["glyph"].name,
                )
    range_ = Range1d(start=min(range_min), end=max(range_max))
    figures["marginals"].x_range = range_
    figures["forests"].x_range = range_
    figures["traces"].y_range = range_


def create_annotations(figures: typing.Figures, num_chains: int) -> typing.Annotations:
    """Create any annotations for the figures of the tool.

    Parameters
    ----------
    figures : typing.Figures
        A dictionary of Bokeh Figure objects.
    num_chains : int
        The number of chains of the model.

    Returns
    -------
    typing.Annotations
        A dictionary of Bokeh Annotation objects.
    """
    renderers = []
    for _, fig in figures.items():
        renderers.extend(PropertyValueList(fig.renderers))
    legend_items = []
    for chain in range(num_chains):
        chain_index = chain + 1
        chain_name = f"chain{chain_index}"
        legend_items.append(
            LegendItem(
                renderers=[
                    renderer
                    for renderer in renderers
                    if chain_name in renderer.name.lower()
                ],
                label=chain_name,
            ),
        )
    legend = Legend(
        items=legend_items,
        orientation="horizontal",
        border_line_color="black",
        click_policy="hide",
    )
    return {"traces": {"legend": legend}, "ranks": {"legend": legend}}


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
    for figure_name, figure_annotations in annotations.items():
        fig = figures[figure_name]
        for _, annotation in figure_annotations.items():
            fig.add_layout(annotation, "below")


def create_tooltips(
    rv_name: str,
    figures: typing.Figures,
    num_chains: int,
) -> typing.Tooltips:
    """Create hover tools for the glyphs used in the figures of the tool.

    Parameters
    ----------
    rv_name : str
        The string representation of the random variable data.
    figures : typing.Figures
        A dictionary of Bokeh Figure objects.
    num_chains : int
        The number of chains of the model.

    Returns
    -------
    typing.Tooltips
        A dictionary of Bokeh HoverTools objects.
    """
    output = {}
    for figure_name, fig in figures.items():
        output[figure_name] = []
        for chain in range(num_chains):
            chain_index = chain + 1
            chain_name = f"chain{chain_index}"
            if figure_name == "marginals":
                glyph_name = f"{figure_name}{chain_name.title()}LineGlyph"
                output[figure_name].append(
                    HoverTool(
                        renderers=filter_renderers(fig, glyph_name),
                        tooltips=[
                            ("Chain", "@chain"),
                            ("Mean", "@mean"),
                            (rv_name, "@x"),
                        ],
                    ),
                )
            if figure_name == "forests":
                glyph_name = f"{figure_name}{chain_name.title()}CircleGlyph"
                output[figure_name].append(
                    HoverTool(
                        renderers=filter_renderers(fig, glyph_name),
                        tooltips=[("Chain", "@chain"), ("Mean", "@x")],
                    ),
                )
            if figure_name == "traces":
                glyph_name = f"{figure_name}{chain_name.title()}LineGlyph"
                output[figure_name].append(
                    HoverTool(
                        renderers=filter_renderers(fig, glyph_name),
                        tooltips=[
                            ("Chain", "@chain"),
                            ("Mean", "@mean"),
                            (rv_name, "@y"),
                        ],
                    ),
                )
            if figure_name == "ranks":
                glyph_name = f"{figure_name}{chain_name.title()}LineGlyph"
                output[figure_name].append(
                    HoverTool(
                        renderers=filter_renderers(fig, glyph_name),
                        tooltips=[("Chain", "@chain"), ("Rank mean", "@rank_mean")],
                    ),
                )
                glyph_name = f"{figure_name}{chain_name.title()}QuadGlyph"
                output[figure_name].append(
                    HoverTool(
                        renderers=filter_renderers(fig, glyph_name),
                        tooltips=[
                            ("Chain", "@chain"),
                            ("Draws", "@draws"),
                            ("Rank", "@rank"),
                        ],
                    ),
                )
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
    for figure_name, fig in figures.items():
        for tips in tooltips[figure_name]:
            fig.add_tools(tips)


def create_widgets(rv_name: str, rv_names: List[str]) -> typing.Widgets:
    """Create the widgets used in the tool.

    Parameters
    ----------
    rv_name : str
        The string representation of the random variable data.
    rv_names : List[str]
        A list of all available random variable names.

    Returns
    -------
    typing.Widgets
        A dictionary of Bokeh widget objects.
    """
    return {
        "rv_select": Select(value=rv_name, options=rv_names, title="Query"),
        "bw_factor_slider": Slider(
            start=0.01,
            end=2.00,
            step=0.01,
            value=1.0,
            title="Bandwidth factor",
        ),
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
    <h2>Rank plots</h2>
    <p style="margin-bottom: 10px">
      Rank plots are a histogram of the samples over time. All samples across
      all chains are ranked and then we plot the average rank for each chain on
      regular intervals. If the chains are mixing well this histogram should
      look roughly uniform. If it looks highly irregular that suggests chains
      might be getting stuck and not adequately exploring the sample space.
      See the paper by Vehtari <em>et al</em> for more information.
    </p>
    <h2>Trace plots</h2>
    <p style="margin-bottom: 10px">
      The more familiar trace plots are also included in this widget. You can
      click on the legend to show/hide different chains and compare them to the
      rank plots.
    </p>
    <ul>
      <li>
        Vehtari A, Gelman A, Simpson D, Carpenter B, Bürkner PC (2021)
        <b>
          Rank-normalization, folding, and localization: An improved \\(\\hat{R}\\)
          for assessing convergence of MCMC (with discussion)
        </b>.
        <em>Bayesian Analysis</em> 16(2)
        667–718.
        <a href=https://dx.doi.org/10.1214/20-BA1221 style="color: blue">
          doi: 10.1214/20-BA1221
        </a>.
      </li>
    </ul>
    """
    return Div(text=text, disable_math=False, min_width=PLOT_WIDTH)


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
    toolbar = create_toolbar(figures)
    help_panel = Panel(child=help_page(), title="Help", name="helpPanel")
    marginal_panel = Panel(
        child=Column(children=[figures["marginals"], widgets["bw_factor_slider"]]),
        title="Marginal 1D",
    )
    forest_panel = Panel(
        child=Column(children=[figures["forests"], widgets["hdi_slider"]]),
        title="Forest",
    )
    left_panels = Tabs(tabs=[marginal_panel, forest_panel])
    trace_panel = Panel(child=Column(children=[figures["traces"]]), title="Trace")
    rank_panel = Panel(child=Column(children=[figures["ranks"]]), title="Rank")
    right_panels = Tabs(tabs=[trace_panel, rank_panel])
    tool_panel = Panel(
        child=Column(
            children=[
                widgets["rv_select"],
                Row(children=[left_panels, right_panels, toolbar]),
            ],
        ),
        title="Trace tool",
    )
    return Tabs(tabs=[tool_panel, help_panel])


def update(
    data: npt.NDArray,
    rv_name: str,
    sources: typing.Sources,
    figures: typing.Figures,
    hdi_probability: float,
    bw_factor: float,
) -> None:
    """Update the tool based on user interaction.

    Parameters
    ----------
    data : npt.NDArray
        A 2D NumPy array where the length of the first dimension is the number of chains
        of the model, and the length of the second dimension is the number of draws of
        the model.
    rv_name : str
        The string representation of the random variable data.
    sources : typing.Sources
        A dictionary of Bokeh ColumnDataSource objects.
    figures : typing.Figures
        A dictionary of Bokeh Figure objects.
    hdi_probability : float
        The HDI probability to use when calculating the HDI bounds.
    bw_factor : float
        Multiplicative factor used when calculating the kernel density estimate.

    Returns
    -------
    None
        Updates Bokeh ColumnDataSource objects.
    """
    range_min = []
    range_max = []
    computed_data = compute_data(data, bw_factor, hdi_probability)
    for figure_name, figure_sources in sources.items():
        figure_data = computed_data[figure_name]
        fig = figures[figure_name]
        for chain_name, chain_sources in figure_sources.items():
            chain_data = figure_data[chain_name]
            for glyph_name, source in chain_sources.items():
                glyph_data = chain_data[glyph_name]
                source.data = glyph_data
                if figure_name == "marginals" and glyph_name == "line":
                    range_min.append(min(glyph_data["x"]))
                    range_max.append(max(glyph_data["x"]))
        if figure_name in ["marginals", "forests"]:
            fig.xaxis.axis_label = rv_name
        if figure_name == "traces":
            fig.yaxis.axis_label = rv_name
    figures["marginals"].x_range.start = min(range_min)
    figures["marginals"].x_range.end = max(range_max)
    figures["forests"].x_range.start = min(range_min)
    figures["forests"].x_range.end = max(range_max)
    figures["traces"].y_range.start = min(range_min)
    figures["traces"].y_range.end = max(range_max)
