# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Methods used to generate the diagnostic tool."""
from typing import List

import arviz as az

import beanmachine.ppl.diagnostics.tools.typing.effective_sample_size as typing
import numpy as np
import numpy.typing as npt
from beanmachine.ppl.diagnostics.tools.helpers.plotting import (
    choose_palette,
    create_toolbar,
    filter_renderers,
    style_figure,
)
from bokeh.models.annotations import Legend, LegendItem
from bokeh.models.glyphs import Circle, Line
from bokeh.models.layouts import Column, Row
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets import Div
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.panels import Panel, Tabs
from bokeh.plotting.figure import figure


PLOT_WIDTH = 1000
PLOT_HEIGHT = 500
FIGURE_NAMES = ["ess"]


def compute_data(
    data: npt.NDArray,
    first_draw: float = 0.0,
    num_points: int = 20,
) -> typing.Data:
    """Compute effective sample size estimates using the given data.

    Parameters
    ----------
    data : npt.NDArray
        A 2D NumPy array where the length of the first dimension is the number of chains
        of the model, and the length of the second dimension is the number of draws of
        the model.
    first_draw : float, optional default is 0
        The first draw index.
    num_points : int, optional default is 20
        The number of divisions in the model samples to compute the effective sample
        size for.

    Returns
    -------
    typing.Data
        A dictionary of data used for the tool.
    """
    num_chains, num_draws = data.shape
    num_samples = num_chains * num_draws
    rule_of_thumb = 100 * num_chains
    ess_x = np.linspace(
        start=num_samples / num_points,
        stop=num_samples,
        num=num_points,
    )
    draw_divisions = np.linspace(
        start=num_draws // num_points,
        stop=num_draws,
        num=num_points,
        dtype=np.integer,
    )
    ess_bulk_y = [
        az.stats.diagnostics._ess_bulk(data[:, first_draw:draw_division])
        for draw_division in draw_divisions
    ]
    ess_tail_y = [
        az.stats.diagnostics._ess_tail(data[:, first_draw:draw_division])
        for draw_division in draw_divisions
    ]
    rule_of_thumb_x = np.linspace(start=0, stop=num_samples, num=num_points)
    rule_of_thumb_y = rule_of_thumb * np.ones(num_points)
    rule_of_thumb_label = [rule_of_thumb] * len(ess_x)
    return {
        "ess": {
            "bulk": {"x": ess_x.tolist(), "y": ess_bulk_y},
            "tail": {"x": ess_x.tolist(), "y": ess_tail_y},
            "rule_of_thumb": {
                "x": rule_of_thumb_x.tolist(),
                "y": rule_of_thumb_y.tolist(),
                "label": rule_of_thumb_label,
            },
        },
    }


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
    return {
        "ess": {
            "bulk": {
                "line": ColumnDataSource(data=data["ess"]["bulk"]),
                "circle": ColumnDataSource(data=data["ess"]["bulk"]),
            },
            "tail": {
                "line": ColumnDataSource(data=data["ess"]["tail"]),
                "circle": ColumnDataSource(data=data["ess"]["tail"]),
            },
            "rule_of_thumb": {
                "line": ColumnDataSource(data=data["ess"]["rule_of_thumb"]),
            },
        },
    }


def create_figures() -> typing.Figures:
    """Create the Bokeh figures used for the tool.

    Returns
    -------
    typing.Figures
        A dictionary of Bokeh Figure objects.
    """
    fig = figure(
        plot_width=PLOT_WIDTH,
        plot_height=PLOT_HEIGHT,
        outline_line_color="black",
        title="Effective Sample Size",
    )
    style_figure(fig)
    return {"ess": fig}


def create_glyphs() -> typing.Glyphs:
    """Create the glyphs used for the figures of the tool.

    Returns
    -------
    typing.Glyphs
        A dictionary of Bokeh Glyphs objects.
    """
    palette = choose_palette(4)
    bulk_glyph_color = palette[0]
    tail_glyph_color = palette[1]
    rule_of_thumb_color = palette[3]
    output = {"ess": {}}
    for glyph_name in ["bulk", "tail", "rule_of_thumb"]:
        glyph_token = "".join([token.title() for token in glyph_name.split("_")])
        if glyph_name in ["bulk", "tail"]:
            color = bulk_glyph_color if glyph_name == "bulk" else tail_glyph_color
            output["ess"][glyph_name] = {
                "line": {
                    "glyph": Line(
                        x="x",
                        y="y",
                        line_color=color,
                        line_alpha=1.0,
                        line_width=2.0,
                        name=f"ess{glyph_token}LineGlyph",
                    ),
                    "hover_glyph": Line(
                        x="x",
                        y="y",
                        line_color=color,
                        line_alpha=1.0,
                        line_width=2.0,
                        name=f"ess{glyph_token}LineHoverGlyph",
                    ),
                },
                "circle": {
                    "glyph": Circle(
                        x="x",
                        y="y",
                        size=10,
                        fill_color=color,
                        line_color="white",
                        fill_alpha=1.0,
                        name=f"ess{glyph_token}CircleGlyph",
                    ),
                    "hover_glyph": Circle(
                        x="x",
                        y="y",
                        size=10,
                        fill_color=color,
                        line_color="black",
                        fill_alpha=1.0,
                        name=f"ess{glyph_token}CircleHoverGlyph",
                    ),
                },
            }
        elif glyph_name == "rule_of_thumb":
            output["ess"][glyph_name] = {
                "line": {
                    "glyph": Line(
                        x="x",
                        y="y",
                        line_color=rule_of_thumb_color,
                        line_alpha=0.7,
                        line_width=4.0,
                        line_dash="dashed",
                        name=f"ess{glyph_token}LineGlyph",
                    ),
                    "hover_glyph": Line(
                        x="x",
                        y="y",
                        line_color=rule_of_thumb_color,
                        line_alpha=0.7,
                        line_width=4.0,
                        line_dash="solid",
                        name=f"ess{glyph_token}LineHoverGlyph",
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
    fig = figures["ess"]
    figure_glyphs = glyphs["ess"]
    figure_sources = sources["ess"]
    for glyph_name, glyphs in figure_glyphs.items():
        glyph_sources = figure_sources[glyph_name]
        for glyph_type, glyph in glyphs.items():
            glyph_source = glyph_sources[glyph_type]
            fig.add_glyph(
                source_or_glyph=glyph_source,
                glyph=glyph["glyph"],
                hover_glyph=glyph["hover_glyph"],
                name=glyph["glyph"].name,
            )


def create_annotations(figures: typing.Figures) -> typing.Annotations:
    """Create any annotations for the figures of the tool.

    Parameters
    ----------
    figures : typing.Figures
        A dictionary of Bokeh Figure objects.

    Returns
    -------
    typing.Annotations
        A dictionary of Bokeh Annotation objects.
    """
    legend_items = []
    fig = figures["ess"]
    output = {}
    for glyph_name in ["bulk", "tail", "rule_of_thumb"]:
        glyph_token = "".join([token.title() for token in glyph_name.split("_")])
        search = f"ess{glyph_token}"
        if glyph_name in ["bulk", "tail"]:
            label = glyph_name.title()
        else:
            label = "Rule-of-thumb"
        filtered_renderers = filter_renderers(fig, search, "GlyphRenderer", True)
        legend_items.append(
            LegendItem(
                label=label,
                renderers=filtered_renderers,
                name=f"ess{glyph_token}LegendItem",
            ),
        )
    output["ess"] = {
        "legend": Legend(
            items=legend_items,
            orientation="horizontal",
            border_line_color="black",
            background_fill_alpha=1.0,
            name="essLegend",
        ),
    }
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
    figures["ess"].add_layout(annotations["ess"]["legend"], "below")


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
    output = {"ess": {}}
    fig = figures["ess"]
    tooltips = []
    for glyph_name in ["bulk", "tail", "rule_of_thumb"]:
        glyph_token = "".join([token.title() for token in glyph_name.split("_")])
        if glyph_name in ["bulk", "tail"]:
            tooltips = [("Total draws", "@x{0,}"), ("ESS", "@y{0,}")]
        elif glyph_name == "rule_of_thumb":
            tooltips = [("Rule-of-thumb", "@label")]
        filtered_renderers = filter_renderers(
            fig,
            f"ess{glyph_token}",
            "GlyphRenderer",
            True,
        )
        tips_name = "".join(
            [c.lower() if i == 0 else c for i, c in enumerate(list(glyph_token))],
        )
        tips = HoverTool(
            renderers=filtered_renderers,
            tooltips=tooltips,
            name=f"{tips_name}Tips",
        )
        output["ess"][glyph_name] = tips
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
    fig = figures["ess"]
    figure_tooltips = tooltips["ess"]
    for _, tips in figure_tooltips.items():
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
    return {"rv_select": Select(value=rv_name, options=rv_names, title="Query")}


def help_page() -> Div:
    """Help tab for the tool.

    Returns
    -------
    Div
        Bokeh Div widget containing the help tab information.
    """
    text = """
    <h2>
      Effective sample size diagnostic
    </h2>
    <p style="margin-bottom: 10px">
      MCMC samplers do not draw truly independent samples from the target
      distribution, which means that our samples are correlated. In an ideal
      situation all samples would be independent, but we do not have that
      luxury. We can, however, measure the number of <em>effectively
      independent</em> samples we draw, which is called the effective sample
      size. You can read more about how this value is calculated in the Vehtari
      <em>et al</em> paper. In brief, it is a measure that combines information
      from the \\(\\hat{R}\\) value with the autocorrelation estimates within the
      chains.
    </p>
    <p style="margin-bottom: 10px">
      ESS estimates come in two variants, <em>ess_bulk</em> and
      <em>ess_tail</em>. The former is the default, but the latter can be useful
      if you need good estimates of the tails of your posterior distribution.
      The rule of thumb for <em>ess_bulk</em> is for this value to be greater
      than 100 per chain on average. The <em>ess_tail</em> is an estimate for
      effectively independent samples considering the more extreme values of the
      posterior. This is not the number of samples that landed in the tails of
      the posterior, but rather a measure of the number of effectively
      independent samples if we sampled the tails of the posterior. The rule of
      thumb for this value is also to be greater than 100 per chain on average.
    </p>
    <p style="margin-bottom: 10px">
      When the model is converging properly, both the bulk and tail lines should
      be <em>roughly</em> linear.
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
    return Row(children=[figures["ess"], toolbar])


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
    figure_layout = create_figure_grid(figures)
    tool_panel = Panel(
        child=Column(
            children=[widgets["rv_select"], figure_layout],
        ),
        title="Effective Sample Size",
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
    figure_data = computed_data["ess"]
    figure_sources = sources["ess"]
    for glyph_name, glyph_sources in figure_sources.items():
        glyph_data = figure_data[glyph_name]
        for _, glyph_source in glyph_sources.items():
            glyph_source.data = dict(**glyph_data)
