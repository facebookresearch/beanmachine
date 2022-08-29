"""Methods used to generate the diagnostic tool."""
from __future__ import annotations

import arviz as az

import beanmachine.ppl.diagnostics.tools.typing.autocorrelation as typing
import numpy as np
import numpy.typing as npt
from beanmachine.ppl.diagnostics.tools.helpers.plotting import (
    choose_palette,
    create_toolbar,
    style_figure,
)
from bokeh.models.annotations import BoxAnnotation
from bokeh.models.glyphs import Quad
from bokeh.models.layouts import Column, Row
from bokeh.models.ranges import DataRange1d
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.markups import Div
from bokeh.models.widgets.panels import Panel, Tabs
from bokeh.models.widgets.sliders import RangeSlider
from bokeh.plotting.figure import figure


FIGURE_NAMES: typing.Figure_names = None
PLOT_WIDTH = 500
PLOT_HEIGHT = 300
N_DISPLAY_COLUMNS = 2


def compute_data(data: npt.NDArray) -> typing.Data:
    """Compute autocorrelation data using the given data.

    Parameters
    ----------
    data : npt.NDArray
        A 2D NumPy array where the length of the first dimension is the number of chains
        of the model, and the length of the second dimension is the number of draws of
        the model.

    Returns
    -------
    typing.Data
        A dictionary of data used for the tool.
    """
    num_chains, num_draws = data.shape
    # Standard error of a Normal distribution within 95% of the density.
    confidence_interval = (1.96 / num_draws) ** 0.5
    autocorr = az.stats.stats_utils.autocorr(data)
    figure_names = []
    output = {}
    for chain in range(num_chains):
        chain_index = chain + 1
        figure_name = f"chain{chain_index}"
        figure_names.append(figure_name)
        chain_data = autocorr[chain, :]
        bins = np.arange(len(chain_data) + 1)
        left = bins[:-1]
        top = chain_data
        right = bins[1:]
        bottom = np.zeros(len(chain_data))
        output[figure_name] = {
            "quad": {
                "left": left.tolist(),
                "top": top.tolist(),
                "right": right.tolist(),
                "bottom": bottom.tolist(),
            },
            "box": {"bottom": -1 * confidence_interval, "top": confidence_interval},
        }
    # Here we set the figure names, which will be used in the other methods of the tool.
    global FIGURE_NAMES
    if FIGURE_NAMES is None:
        FIGURE_NAMES = figure_names
    return output


def create_sources(data: typing.Data) -> typing.Sources:
    """Create Bokeh sources from the given data that will be bound to glyphs.

    Parameters
    ----------
    data : typing.Data
        Computed data for the tool.

    Returns
    -------
    typing.Sources
        A dictionary of Bokeh ColumnDataSource objects.
    """
    num_chains = len(data)
    output = {}
    for chain in range(num_chains):
        chain_index = chain + 1
        figure_name = f"chain{chain_index}"
        output[figure_name] = {}
        chain_data = data[figure_name]["quad"]
        output[figure_name]["quad"] = ColumnDataSource(data=chain_data)
    return output


def create_figures(num_chains: int) -> typing.Figures:
    """Create the Bokeh figures used for the tool.

    Parameters
    ----------
    num_chains : int
        The number of chains of the model.

    Returns
    -------
    typing.Figures
        A dictionary of Bokeh figures.
    """
    output = {}
    for chain in range(num_chains):
        chain_index = chain + 1
        figure_name = f"chain{chain_index}"
        fig = figure(
            plot_width=PLOT_WIDTH,
            plot_height=PLOT_HEIGHT,
            outline_line_color="black",
            title=f"Chain {chain_index}",
            x_range=DataRange1d(start=-1, end=100),
            y_range=DataRange1d(start=-1.2, end=1.2),
            x_axis_label="Draw",
            y_axis_label="Autocorrelation",
        )
        style_figure(fig)
        output[figure_name] = fig
    figure_names = list(output.keys())
    first_fig = output[figure_names[0]]
    x_range = first_fig.x_range
    y_range = first_fig.y_range
    for _, fig in output.items():
        fig.x_range = x_range
        fig.y_range = y_range
    return output


def create_glyphs(num_chains: int) -> typing.Glyphs:
    """Create the glyphs used for the figures of the tool.

    Parameters
    ----------
    num_chains : int
        The number of chains of the model.

    Returns
    -------
    typing.Glyphs
        A dictionary of Bokeh Glyphs objects.
    """
    palette = choose_palette(num_chains)
    output = {}
    for chain in range(num_chains):
        chain_index = chain + 1
        figure_name = f"chain{chain_index}"
        output[figure_name] = {}
        color = palette[chain]
        glyph = Quad(
            left="left",
            top="top",
            right="right",
            bottom="bottom",
            fill_color=color,
            line_color="white",
            fill_alpha=1.0,
            name=f"autocorrelationGlyphChain{chain_index}",
        )
        hover_glyph = Quad(
            left="left",
            top="top",
            right="right",
            bottom="bottom",
            fill_color=color,
            line_color="black",
            fill_alpha=1.0,
            name=f"autocorrelationGlyphChain{chain_index}",
        )
        output[figure_name]["quad"] = {"glyph": glyph, "hover_glyph": hover_glyph}
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
    for figure_name, fig in figures.items():
        glyph = glyphs[figure_name]["quad"]["glyph"]
        hover_glyph = glyphs[figure_name]["quad"]["hover_glyph"]
        glyph_name = glyph.name
        source = sources[figure_name]["quad"]
        fig.add_glyph(
            source_or_glyph=source,
            glyph=glyph,
            hover_glyph=hover_glyph,
            name=glyph_name,
        )


def create_annotations(data: typing.Data) -> typing.Annotations:
    """Create any annotations for the figures of the tool.

    Parameters
    ----------
    data : typing.Data
        Computed data for the tool.

    Returns
    -------
    typing.Annotations
        A dictionary of Bokeh Annotation objects.
    """
    figure_names = data.keys()
    palette = choose_palette(len(figure_names))
    output = {}
    for i, (figure_name, figure_data) in enumerate(data.items()):
        color = palette[i]
        output[figure_name] = BoxAnnotation(
            bottom=figure_data["box"]["bottom"],
            top=figure_data["box"]["top"],
            fill_color=color,
            fill_alpha=0.2,
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
    for figure_name, fig in figures.items():
        fig.add_layout(annotations[figure_name])


def create_tooltips(figures: typing.Figures) -> typing.Tooltips:
    """Create hover tools for the glyphs used in the figures of the tool.

    Parameters
    ----------
    figures : typing.Figures
        A dictionary of Bokeh Figure objects.

    Returns
    -------
    typing.Tooltips
        A dictionary of Bokeh HoverTools objects.
    """
    output = {}
    for figure_name, fig in figures.items():
        output[figure_name] = HoverTool(
            renderers=fig.renderers,
            tooltips=[("Autocorrelation", "@top")],
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
        fig.add_tools(tooltips[figure_name])


def create_widgets(rv_name: str, rv_names: list[str], num_draws: int) -> typing.Widgets:
    """Create the widgets used in the tool.

    Parameters
    ----------
    rv_name : str
        The string representation of the random variable data.
    rv_names : list[str]
        A list of all available random variable names.
    num_draws : int
        The number of draws used in the model for a single chain.

    Returns
    -------
    typing.Widgets
        A dictionary of Bokeh widget objects.
    """
    end = 10 if num_draws <= 2 * 100 else 100
    rv_select = Select(
        value=rv_name,
        options=rv_names,
        title="Query",
        name="autocorrelationRVSelect",
    )
    range_slider = RangeSlider(
        start=0,
        end=num_draws,
        value=(0, end),
        step=end,
        title="Autocorrelation range",
        name="autocorrelationRangeSlider",
    )
    return {"rv_select": rv_select, "range_slider": range_slider}


def help_page() -> Div:
    """Help tab for the tool.

    Returns
    -------
    Div
        Bokeh Div widget containing the help tab information.
    """
    text = """
        <h2>
          Autocorrelation plots
        </h2>
        <p style="margin-bottom: 10px">
          Autocorrelation plots measure how predictive the last several samples are
          of the current sample. Autocorrelation may vary between -1.0
          (deterministically anticorrelated) and 1.0 (deterministically correlated).
          We compute autocorrelation approximately, so it may sometimes exceed these
          bounds. In an ideal world, the current sample is chosen independently of
          the previous samples: an autocorrelation of zero. This is not possible in
          practice, due to stochastic noise and the mechanics of how inference
          works.
        </p>
    """
    return Div(text=text, disable_math=False, min_width=800)


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
    figure_values = list(figures.values())
    figure_rows = []
    while len(figure_values):
        figs = figure_values[:N_DISPLAY_COLUMNS]
        for i, fig in enumerate(figs):
            if i != 0:
                fig.yaxis.axis_label = None
        figure_rows.append(figs)
        for fig in figs:
            figure_values.pop(figure_values.index(fig))
    for i, figure_row in enumerate(figure_rows):
        if i != len(figure_rows) - 1:
            for fig in figure_row:
                fig.xaxis.axis_label = None
    figure_layout = []
    for i in range(len(figure_rows)):
        figure_layout.append(Row(children=figure_rows[i]))
    return Row(children=[Column(children=figure_layout), toolbar])


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
        Bokeh Tabs object.
    """
    help_panel = Panel(child=help_page(), title="Help", name="helpPanel")
    figure_layout = create_figure_grid(figures)
    tool_panel = Panel(
        child=Column(
            children=[widgets["rv_select"], figure_layout, widgets["range_slider"]],
        ),
        title="Autocorrelation",
        name="toolPanel",
    )
    return Tabs(tabs=[tool_panel, help_panel])


def update(data: npt.NDArray, sources: typing.Sources) -> None:
    """Update the tool based on user interaction.

    Parameters
    ----------
    data : npt.NDArray
        A 2D NumPy array where the length of the first dimension is the number of chains
        of the model, and the length of the second dimension is the number of draws of
        the model.
    sources : typing.Sources
        A dictionary of Bokeh ColumnDataSource objects.

    Returns
    -------
    None
        When Bokeh sources are updated with new data, the figures of the tool will be
        redrawn.
    """
    computed_data = compute_data(data)
    for figure_name, figure_data in computed_data.items():
        sources[figure_name]["quad"].data = dict(**figure_data["quad"])
