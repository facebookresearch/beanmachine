# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Methods used to generate the diagnostic tool."""

from typing import List

from beanmachine.ppl.diagnostics.tools.trace import typing
from beanmachine.ppl.diagnostics.tools.utils import plotting_utils
from bokeh.core.property.wrappers import PropertyValueList
from bokeh.models.annotations import Legend, LegendItem
from bokeh.models.glyphs import Circle, Line, Quad
from bokeh.models.layouts import Column, Row
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.markups import Div
from bokeh.models.widgets.panels import Panel, Tabs
from bokeh.models.widgets.sliders import Slider
from bokeh.plotting.figure import figure

PLOT_WIDTH = 400
PLOT_HEIGHT = 500
TRACE_PLOT_WIDTH = 600
FIGURE_NAMES = ["marginals", "forests", "traces", "ranks"]
# Define what the empty data object looks like in order to make the browser handle all
# computations.
EMPTY_DATA = {}


def create_empty_data(num_chains: int) -> typing.Data:
    """Create an empty data object for the tool.

    We do not know a priori how many chains a model will have, so we use this method to
    build an empty data object with the given number of chains.

    Parameters
    ----------
    num_chains : int
        The number of chains from the model.

    Returns
    -------
    typing.Data
        An empty data object to be filled by JavaScript.
    """
    output = {
        "marginals": {},
        "forests": {},
        "traces": {},
        "ranks": {},
    }
    for chain in range(num_chains):
        chain_index = chain + 1
        chain_name = f"chain{chain_index}"
        marginal = {
            "line": {"x": [], "y": []},
            "chain": [],
            "mean": [],
            "bandwidth": [],
        }
        forest = {
            "line": {"x": [], "y": []},
            "circle": {"x": [], "y": []},
            "chain": [],
            "mean": [],
        }
        trace = {
            "line": {"x": [], "y": []},
            "chain": [],
            "mean": [],
        }
        rank = {
            "quad": {
                "left": [],
                "top": [],
                "right": [],
                "bottom": [],
                "chain": [],
                "draws": [],
                "rank": [],
            },
            "line": {"x": [], "y": []},
            "chain": [],
            "rankMean": [],
            "mean": [],
        }
        single_chain_data = [marginal, forest, trace, rank]
        chain_data = dict(zip(FIGURE_NAMES, single_chain_data))
        for figure_name in FIGURE_NAMES:
            output[figure_name][chain_name] = chain_data[figure_name]
    return output


def create_sources(num_chains: int) -> typing.Sources:
    """Create Bokeh sources from the given data that will be bound to glyphs.

    Parameters
    ----------
    num_chains : int
        The number of chains from the model.

    Returns
    -------
    typing.Sources
        A dictionary of Bokeh ColumnDataSource objects.
    """
    global EMPTY_DATA
    if not EMPTY_DATA:
        EMPTY_DATA = create_empty_data(num_chains=num_chains)

    output = {}
    for figure_name, figure_data in EMPTY_DATA.items():
        output[figure_name] = {}
        for chain_name, chain_data in figure_data.items():
            output[figure_name][chain_name] = {}
            if figure_name == "marginals":
                output[figure_name][chain_name]["line"] = ColumnDataSource(
                    {
                        "x": chain_data["line"]["x"],
                        "y": chain_data["line"]["y"],
                        "chain": chain_data["chain"],
                        "mean": chain_data["mean"],
                    },
                )
            if figure_name == "forests":
                output[figure_name][chain_name]["line"] = ColumnDataSource(
                    {
                        "x": chain_data["line"]["x"],
                        "y": chain_data["line"]["y"],
                    },
                )
                output[figure_name][chain_name]["circle"] = ColumnDataSource(
                    {
                        "x": chain_data["circle"]["x"],
                        "y": chain_data["circle"]["y"],
                        "chain": chain_data["chain"],
                    },
                )
            if figure_name == "traces":
                output[figure_name][chain_name]["line"] = ColumnDataSource(
                    {
                        "x": chain_data["line"]["x"],
                        "y": chain_data["line"]["y"],
                        "chain": chain_data["chain"],
                        "mean": chain_data["mean"],
                    },
                )
            if figure_name == "ranks":
                output[figure_name][chain_name]["line"] = ColumnDataSource(
                    {
                        "x": chain_data["line"]["x"],
                        "y": chain_data["line"]["y"],
                        "chain": chain_data["chain"],
                        "rankMean": chain_data["rankMean"],
                    },
                )
                output[figure_name][chain_name]["quad"] = ColumnDataSource(
                    {
                        "left": chain_data["quad"]["left"],
                        "top": chain_data["quad"]["top"],
                        "right": chain_data["quad"]["right"],
                        "bottom": chain_data["quad"]["bottom"],
                        "chain": chain_data["chain"],
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
        The number of chains from the model.

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
            sizing_mode="scale_both",
        )
        plotting_utils.style_figure(fig)
        # NOTE: There are several figures where we do not want the x-axis to change its
        #       limits. This is why we set the x_range to an object from Bokeh called
        #       Range1d.
        if figure_name == "marginals":
            fig.title = "Marginal"
            fig.xaxis.axis_label = rv_name
            # fig.x_range = Range1d()
            fig.yaxis.visible = False
        elif figure_name == "forests":
            fig.title = "Forest"
            fig.xaxis.axis_label = rv_name
            fig.yaxis.axis_label = "Chain"
            fig.yaxis.minor_tick_line_color = None
            fig.yaxis.ticker.desired_num_ticks = num_chains
            # fig.x_range = Range1d()
        elif figure_name == "traces":
            fig.title = "Trace"
            fig.xaxis.axis_label = "Draw from single chain"
            fig.yaxis.axis_label = rv_name
            fig.width = TRACE_PLOT_WIDTH
            # fig.x_range = Range1d()
        elif figure_name == "ranks":
            fig.title = "Rank"
            fig.xaxis.axis_label = "Rank from all chains"
            fig.yaxis.axis_label = "Chain"
            fig.width = TRACE_PLOT_WIDTH
            fig.yaxis.minor_tick_line_color = None
            fig.yaxis.ticker.desired_num_ticks = num_chains
        output[figure_name] = fig
    return output


def create_glyphs(num_chains: int) -> typing.Glyphs:
    """Create the glyphs used for the figures of the tool.

    Parameters
    ----------
    num_chains : int
        The number of chains from the model.

    Returns
    -------
    typing.Glyphs
        A dictionary of Bokeh Glyphs objects.
    """
    global EMPTY_DATA
    if not EMPTY_DATA:
        EMPTY_DATA = create_empty_data(num_chains=num_chains)

    palette = plotting_utils.choose_palette(num_colors=num_chains)
    output = {}
    for figure_name, figure_data in EMPTY_DATA.items():
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
            elif figure_name == "forests":
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
    # range_min = []
    # range_max = []
    for figure_name, figure_sources in sources.items():
        fig = figures[figure_name]
        for chain_name, source in figure_sources.items():
            chain_glyphs = glyphs[figure_name][chain_name]
            # NOTE: Every figure has a line glyph, so we always add it here.
            fig.add_glyph(
                source_or_glyph=source["line"],
                glyph=chain_glyphs["line"]["glyph"],
                hover_glyph=chain_glyphs["line"]["hover_glyph"],
                name=chain_glyphs["line"]["glyph"].name,
            )
            # We want to keep the x-axis from moving when changing queries, so we add
            # the bounds below from the marginal figure. All figures that need to keep
            # its range stable are linked to the marginal figure's range below.
            if figure_name == "marginals":
                pass
                # data = source["line"].data["x"]
                # minimum = min(data) if len(data) != 0 else 0
                # maximum = max(data) if len(data) != 0 else 1
                # range_min.append(minimum)
                # range_max.append(maximum)
            elif figure_name == "forests":
                fig.add_glyph(
                    source_or_glyph=source["circle"],
                    glyph=chain_glyphs["circle"]["glyph"],
                    hover_glyph=chain_glyphs["circle"]["hover_glyph"],
                    name=chain_glyphs["circle"]["glyph"].name,
                )
            elif figure_name == "ranks":
                fig.add_glyph(
                    source_or_glyph=source["quad"],
                    glyph=chain_glyphs["quad"]["glyph"],
                    hover_glyph=chain_glyphs["quad"]["hover_glyph"],
                    name=chain_glyphs["quad"]["glyph"].name,
                )
    # Link figure ranges together.
    # figures["marginals"].x_range = Range1d(
    #     start=min(range_min) if len(range_min) != 0 else 0,
    #     end=max(range_max) if len(range_max) != 0 else 1,
    # )
    figures["forests"].x_range = figures["marginals"].x_range
    # figures["traces"].y_range = figures["marginals"].x_range


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
    output = {"traces": {"legend": legend}, "ranks": {"legend": legend}}
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
                        renderers=plotting_utils.filter_renderers(fig, glyph_name),
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
                        renderers=plotting_utils.filter_renderers(fig, glyph_name),
                        tooltips=[
                            ("Chain", "@chain"),
                            (rv_name, "@x"),
                        ],
                    ),
                )
            if figure_name == "traces":
                glyph_name = f"{figure_name}{chain_name.title()}LineGlyph"
                output[figure_name].append(
                    HoverTool(
                        renderers=plotting_utils.filter_renderers(fig, glyph_name),
                        tooltips=[
                            ("Chain", "@chain"),
                            ("Mean", "@mean"),
                            (rv_name, "@y"),
                        ],
                    ),
                )
            if figure_name == "ranks":
                output[figure_name].append(
                    {
                        "line": HoverTool(
                            renderers=plotting_utils.filter_renderers(
                                fig, f"{figure_name}{chain_name.title()}LineGlyph"
                            ),
                            tooltips=[("Chain", "@chain"), ("Rank mean", "@rankMean")],
                        ),
                        "quad": HoverTool(
                            renderers=plotting_utils.filter_renderers(
                                fig, f"{figure_name}{chain_name.title()}QuadGlyph"
                            ),
                            tooltips=[
                                ("Chain", "@chain"),
                                ("Draws", "@draws"),
                                ("Rank", "@rank"),
                            ],
                        ),
                    }
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
            if figure_name == "ranks":
                for _, tips_ in tips.items():
                    fig.add_tools(tips_)
            else:
                fig.add_tools(tips)


def create_widgets(rv_names: List[str], rv_name: str) -> typing.Widgets:
    """Create the widgets used in the tool.

    Parameters
    ----------
    rv_names : List[str]
        A list of all available random variable names.
    rv_name : str
        The string representation of the random variable data.

    Returns
    -------
    typing.Widgets
        A dictionary of Bokeh widget objects.
    """
    output = {
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
    return output


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
        <a
          href=https://dx.doi.org/10.1214/20-BA1221
          style="color: blue"
          target="_blank"
        >
          doi: 10.1214/20-BA1221
        </a>.
      </li>
    </ul>
    """
    return Div(text=text, disable_math=False, min_width=PLOT_WIDTH)


def create_view(figures: typing.Figures, widgets: typing.Widgets) -> Tabs:
    """Create the tool view.

    Parameters
    ----------
    figures : typing.Figures
        A dictionary of Bokeh Figure objects.
    widgets : typing.Widgets
        A dictionary of Bokeh widget objects.

    Returns
    -------
    Tabs
        Bokeh Tabs objects.
    """
    toolbar = plotting_utils.create_toolbar(list(figures.values()))
    help_panel = Panel(child=help_page(), title="Help", name="helpPanel")
    marginal_panel = Panel(
        child=Column(
            children=[figures["marginals"], widgets["bw_factor_slider"]],
            sizing_mode="scale_both",
        ),
        title="Marginals",
    )
    forest_panel = Panel(
        child=Column(
            children=[figures["forests"], widgets["hdi_slider"]],
            sizing_mode="scale_both",
        ),
        title="HDIs",
    )
    left_panels = Tabs(tabs=[marginal_panel, forest_panel], sizing_mode="scale_both")
    trace_panel = Panel(
        child=Column(children=[figures["traces"]], sizing_mode="scale_both"),
        title="Traces",
    )
    rank_panel = Panel(
        child=Column(children=[figures["ranks"]], sizing_mode="scale_both"),
        title="Ranks",
    )
    right_panels = Tabs(tabs=[trace_panel, rank_panel], sizing_mode="scale_both")
    tool_panel = Panel(
        child=Column(
            children=[
                widgets["rv_select"],
                Row(
                    children=[left_panels, right_panels, toolbar],
                    sizing_mode="scale_both",
                ),
            ],
            sizing_mode="scale_both",
        ),
        title="Trace tool",
    )
    return Tabs(tabs=[tool_panel, help_panel], sizing_mode="scale_both")
