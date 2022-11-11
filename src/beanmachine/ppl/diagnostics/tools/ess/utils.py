# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Methods used to generate the diagnostic tool."""
from typing import List

from beanmachine.ppl.diagnostics.tools.ess import typing
from beanmachine.ppl.diagnostics.tools.utils import plotting_utils
from bokeh.models.annotations import Legend, LegendItem
from bokeh.models.glyphs import Circle, Line
from bokeh.models.layouts import Column, Row
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.markups import Div
from bokeh.models.widgets.panels import Panel, Tabs
from bokeh.plotting.figure import figure


PLOT_WIDTH = 600
PLOT_HEIGHT = 200
FIGURE_NAMES = ["ess"]
# Define what the empty data object looks like in order to make the browser handle all
# computations.
EMPTY_DATA = {
    "ess": {
        "bulk": {"x": [], "y": []},
        "tail": {"x": [], "y": []},
        "ruleOfThumb": {"x": [], "y": []},
    }
}


def create_sources() -> typing.Sources:
    """
    Create Bokeh sources from the given data that will be bound to glyphs.

    Returns:
        typing.Sources: A dictionary of Bokeh ``ColumnDataSource`` objects.
    """
    output = {
        "ess": {
            "bulk": {
                "line": ColumnDataSource({"x": [], "y": []}),
                "circle": ColumnDataSource({"x": [], "y": []}),
            },
            "tail": {
                "line": ColumnDataSource({"x": [], "y": []}),
                "circle": ColumnDataSource({"x": [], "y": []}),
            },
            "ruleOfThumb": {"line": ColumnDataSource({"x": [], "y": []})},
        }
    }
    return output


def create_figures() -> typing.Figures:
    """
    Create the Bokeh figures used for the tool.

    Returns:
        typing.Figures: A dictionary of Bokeh ``Figure`` objects.
    """
    fig = figure(
        outline_line_color="black",
        title="Effective Sample Size",
        max_width=PLOT_WIDTH,
        max_height=PLOT_HEIGHT,
        x_axis_label="Total number of draws",
        y_axis_label="Effective Sample Size",
        sizing_mode="scale_both",
    )
    fig.grid.grid_line_alpha = 0.3
    fig.grid.grid_line_color = "grey"
    fig.grid.grid_line_width = 0.3
    fig.xaxis.minor_tick_line_color = "grey"
    fig.yaxis.minor_tick_line_color = "grey"
    fig.y_range.start = 0
    output = {"ess": fig}
    return output


def create_glyphs() -> typing.Glyphs:
    """Create the glyphs used for the figures of the tool.

    Returns
    -------
    typing.Glyphs
        A dictionary of Bokeh Glyphs objects.
    """
    palette = plotting_utils.choose_palette(4)
    bulk_color = palette[0]
    tail_color = palette[1]
    rule_of_thumb_color = palette[3]
    output = {
        "ess": {
            "bulk": {
                "circle": {
                    "glyph": Circle(
                        x="x",
                        y="y",
                        size=10,
                        fill_color=bulk_color,
                        line_color="white",
                        fill_alpha=1.0,
                        name="essBulkCircle",
                    ),
                    "hover_glyph": Circle(
                        x="x",
                        y="y",
                        size=10,
                        fill_color=bulk_color,
                        line_color="white",
                        fill_alpha=1.0,
                    ),
                },
                "line": {
                    "glyph": Line(
                        x="x",
                        y="y",
                        line_color=bulk_color,
                        line_alpha=0.7,
                        line_width=2.0,
                        name="essBulkLine",
                    )
                },
            },
            "tail": {
                "circle": {
                    "glyph": Circle(
                        x="x",
                        y="y",
                        size=10,
                        fill_color=tail_color,
                        line_color="white",
                        fill_alpha=1.0,
                        name="essTailCircle",
                    ),
                    "hover_glyph": Circle(
                        x="x",
                        y="y",
                        size=10,
                        fill_color=tail_color,
                        line_color="white",
                        fill_alpha=1.0,
                    ),
                },
                "line": {
                    "glyph": Line(
                        x="x",
                        y="y",
                        line_color=tail_color,
                        line_alpha=0.7,
                        line_width=2.0,
                        name="essTailLine",
                    )
                },
            },
            "ruleOfThumb": {
                "line": {
                    "glyph": Line(
                        x="x",
                        y="y",
                        line_color=rule_of_thumb_color,
                        line_alpha=0.7,
                        line_width=4.0,
                        line_dash="dashed",
                        name="ruleOfThumb",
                    ),
                    "hover_glyph": Line(
                        x="x",
                        y="y",
                        line_color=rule_of_thumb_color,
                        line_alpha=0.7,
                        line_width=4.0,
                        line_dash="solid",
                    ),
                },
            },
        },
    }
    return output


def add_glyphs(
    sources: typing.Sources,
    figures: typing.Figures,
    glyphs: typing.Glyphs,
) -> None:
    """
    Bind source data to glyphs and add the glyphs to the given figures.

    Args:
        sources (typing.Sources): A dictionary of Bokeh ``ColumnDataSource`` objects.
        figures (typing.Figures): A dictionary of Bokeh ``Figure`` objects.
        glyphs (typing.Glyphs): A dictionary of Bokeh ``Glyph`` objects.

    Returns:
        None: Adds data bound glyphs to the given figures directly.
    """
    # Add the bulk ESS glyphs.
    figures["ess"].add_glyph(
        source_or_glyph=sources["ess"]["bulk"]["line"],
        glyph=glyphs["ess"]["bulk"]["line"]["glyph"],
        name="essBulk",
    )
    figures["ess"].add_glyph(
        source_or_glyph=sources["ess"]["bulk"]["circle"],
        glyph=glyphs["ess"]["bulk"]["circle"]["glyph"],
        hover_glyph=glyphs["ess"]["bulk"]["circle"]["hover_glyph"],
        name="essBulkCircle",
    )
    # Add the tail ESS glyphs.
    figures["ess"].add_glyph(
        source_or_glyph=sources["ess"]["tail"]["line"],
        glyph=glyphs["ess"]["tail"]["line"]["glyph"],
        name="essTail",
    )
    figures["ess"].add_glyph(
        source_or_glyph=sources["ess"]["tail"]["circle"],
        glyph=glyphs["ess"]["tail"]["circle"]["glyph"],
        hover_glyph=glyphs["ess"]["tail"]["circle"]["hover_glyph"],
        name="essTailCircle",
    )
    # Add the rule-of-thumb glyphs.
    figures["ess"].add_glyph(
        source_or_glyph=sources["ess"]["ruleOfThumb"]["line"],
        glyph=glyphs["ess"]["ruleOfThumb"]["line"]["glyph"],
        hover_glyph=glyphs["ess"]["ruleOfThumb"]["line"]["hover_glyph"],
        name="ruleOfThumb",
    )


def create_annotations(figures: typing.Figures) -> typing.Annotations:
    """Create any annotations for the figures of the tool.

    Args:
        figures (typing.Figures): A dictionary of Bokeh ``Figure`` objects.

    Returns:
        typing.Annotations: A dictionary of Bokeh ``Annotation`` objects.
    """
    bulk = LegendItem(
        label="Bulk",
        renderers=[
            renderer
            for renderer in figures["ess"].renderers
            if renderer.name and renderer.name.startswith("essBulk")
        ],
    )
    tail = LegendItem(
        label="Tail",
        renderers=[
            renderer
            for renderer in figures["ess"].renderers
            if renderer.name and renderer.name.startswith("essTail")
        ],
    )
    rule_of_thumb = LegendItem(
        label="Rule of thumb",
        renderers=[
            renderer
            for renderer in figures["ess"].renderers
            if renderer.name and renderer.name == "ruleOfThumb"
        ],
    )
    legend = Legend(
        items=[bulk, tail, rule_of_thumb],
        orientation="horizontal",
        border_line_color="black",
        background_fill_alpha=1.0,
        name="essLegend",
    )
    output = {"ess": legend}
    return output


def add_annotations(figures: typing.Figures, annotations: typing.Annotations) -> None:
    """
    Add the given annotations to the given figures of the tool.

    Args:
        figures (typing.Figures): A dictionary of Bokeh ``Figure`` objects.
        annotations (typing.Annotations): A dictionary of Bokeh ``Annotation`` objects.

    Returns:
        None: Adds annotations directly to the given figures.
    """
    figures["ess"].add_layout(annotations["ess"], "below")


def create_tooltips(figures: typing.Figures) -> typing.Tooltips:
    """
    Create hover tools for the glyphs used in the figures of the tool.

    Args:
        figures (typing.Figures): A dictionary of Bokeh ``Figure`` objects.

    Returns:
        typing.Tooltips: A dictionary of Bokeh ``HoverTools`` objects.
    """
    bulk_renderers = [
        renderer
        for renderer in figures["ess"].renderers
        if renderer.name and renderer.name == "essBulkCircle"
    ]
    tail_renderers = [
        renderer
        for renderer in figures["ess"].renderers
        if renderer.name and renderer.name == "essTailCircle"
    ]
    rule_of_thumb_renderers = [
        renderer
        for renderer in figures["ess"].renderers
        if renderer.name and renderer.name == "ruleOfThumb"
    ]
    output = {
        "ess": {
            "bulk": HoverTool(
                renderers=bulk_renderers,
                tooltips=[("Total draws", "@x{0,}"), ("ESS", "@y{0,}")],
            ),
            "tail": HoverTool(
                renderers=tail_renderers,
                tooltips=[("Total draws", "@x{0,}"), ("ESS", "@y{0,}")],
            ),
            "ruleOfThumb": HoverTool(
                renderers=rule_of_thumb_renderers,
                tooltips=[("Rule-of-thumb", "@y{0,}")],
            ),
        }
    }
    return output


def add_tooltips(figures: typing.Figures, tooltips: typing.Tooltips) -> None:
    """
    Add the given tools to the given figures.

    Args:
        figures (typing.Figures): A dictionary of Bokeh ``Figure`` objects.
        tooltips (typing.Tooltips): A dictionary of Bokeh ``HoverTools`` objects.

    Returns:
        None: Adds the tooltips directly to the given figures.
    """
    figures["ess"].add_tools(tooltips["ess"]["bulk"])
    figures["ess"].add_tools(tooltips["ess"]["tail"])
    figures["ess"].add_tools(tooltips["ess"]["ruleOfThumb"])


def create_widgets(rv_name: str, rv_names: List[str]) -> typing.Widgets:
    """
    Create the widgets used in the tool.

    Args:
        rv_name (str): The string representation of the random variable data.
        rv_names (List[str]): A list of all available random variable names in the
            model.

    Returns:
        typing.Widgets: A dictionary of Bokeh widget objects.
    """
    output = {
        "rv_select": Select(
            value=rv_name,
            options=rv_names,
            title="Query",
            max_width=PLOT_WIDTH,
            sizing_mode="scale_width",
        ),
    }
    return output


def help_page() -> Div:
    """
    Help tab for the tool.

    Returns:
        Div: Bokeh ``Div`` widget containing the help tab information.
    """
    text = """
    <h2>Effective sample size diagnostic</h2>
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
    return Div(
        text=text,
        disable_math=False,
        max_width=PLOT_WIDTH,
        max_height=PLOT_HEIGHT,
        sizing_mode="scale_both",
    )


def create_view(figures: typing.Figures, widgets: typing.Widgets) -> Tabs:
    """
    Create the tool view.

    Args:
        figures (typing.Figures): A dictionary of Bokeh ``Figure`` objects.
        widgets (typing.Widgets): A dictionary of Bokeh ``Widget`` objects.

    Returns:
        Tabs: Bokeh ``Tabs`` object that contains the tool.
    """
    toolbar = plotting_utils.create_toolbar(figures=list(figures.values()))
    figure_grid = Row(
        children=[*list(figures.values()), toolbar],
        css_classes=["bm-tool-loading", "arcs"],
        sizing_mode="scale_both",
    )
    help_panel = Panel(child=help_page(), title="Help", name="helpPanel")
    tool_panel = Panel(
        child=Column(
            children=[widgets["rv_select"], figure_grid],
            sizing_mode="scale_both",
        ),
        title="ESS",
        name="toolPanel",
    )
    tabs = Tabs(
        tabs=[tool_panel, help_panel],
        sizing_mode="scale_both",
    )
    return tabs
