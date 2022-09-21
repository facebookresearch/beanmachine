# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Plotting utilities for the diagnostic tools."""

from typing import List

from bokeh.core.property.nullable import Nullable
from bokeh.core.property.primitive import Null
from bokeh.core.property.wrappers import PropertyValueList
from bokeh.models.layouts import Column, Row
from bokeh.models.renderers import GlyphRenderer
from bokeh.models.tools import ProxyToolbar, ResetTool, SaveTool, ToolbarBox
from bokeh.palettes import Colorblind
from bokeh.plotting.figure import Figure


def style_figure(figure: Figure) -> None:
    figure.grid.grid_line_alpha = 0.3
    figure.grid.grid_line_color = "grey"
    figure.grid.grid_line_width = 0.3
    figure.xaxis.minor_tick_line_color = "grey"
    figure.yaxis.minor_tick_line_color = "grey"


def choose_palette(num_colors: int) -> List[str]:
    palette_indices = [key for key in Colorblind.keys() if num_colors <= key]
    if not palette_indices:
        palette_index = max(Colorblind.keys())
    else:
        palette_index = min(palette_indices)
    return Colorblind[palette_index]


def create_toolbar(figures: List[Figure]) -> ToolbarBox:
    toolbars = []
    for figure in figures:
        toolbars.append(figure.toolbar)
        figure.toolbar_location = Nullable(Null)._default
    tools = []
    for toolbar in toolbars:
        tools.extend(toolbar.tools)
    tools = [tool for tool in tools if type(tool).__name__ != "HoverTool"]
    if len(tools) == 0:
        tools = [SaveTool(), ResetTool()]
    return ToolbarBox(
        toolbar=ProxyToolbar(toolbars=toolbars, tools=tools),
        toolbar_location="right",
    )


def create_figure_grid(figures: List[Figure], num_figure_columns: int) -> Row:
    """Layout the given figures in a grid, and make one toolbar.

    Parameters
    ----------
    figures : typing.Figures
        A dictionary of Bokeh Figure objects.
    num_figure_columns : int
        The number of columns to display the figures with.

    Returns
    -------
    Row
        A Bokeh layout object.
    """
    toolbar = create_toolbar(figures)
    figure_rows = []
    while len(figures):
        figs = figures[:num_figure_columns]
        for i, fig in enumerate(figs):
            if i != 0:
                fig.yaxis.axis_label = None
        figure_rows.append(figs)
        for fig in figs:
            figures.pop(figures.index(fig))
    for i, figure_row in enumerate(figure_rows):
        if i != len(figure_rows) - 1:
            for fig in figure_row:
                fig.xaxis.axis_label = None
    figure_layout = []
    for i in range(len(figure_rows)):
        figure_layout.append(Row(children=figure_rows[i]))
    return Row(children=[Column(children=figure_layout), toolbar])


def filter_renderers(
    figure: Figure,
    search: str,
    glyph_type: str = "GlyphRenderer",
    substring: bool = False,
) -> List[GlyphRenderer]:
    """Filter Bokeh figure renderers given the search string.

    Parameters
    ----------
    figure : Figure
        A Bokeh figure to filter renderers from.
    search : str
        The search string to filter for.
    glyph_type : str, optional default is "GlyphRenderer"
        The glyph type.
    substring : bool, optional default is ``False``
        Flag to indicate if we should use the search term as a substring search.
        - ``False`` indicates the name of the glyph == search.
        - ``True`` indicates the search is a substring of the glyph name.

    Returns
    -------
    List[GlyphRenderer]
        A list of Bokeh glyph renderer objects.
    """
    output = []
    renderers = PropertyValueList(figure.renderers)
    for renderer in renderers:
        if renderer.name is not None and type(renderer).__name__ == glyph_type:
            if substring and search in renderer.name:
                output.append(renderer)
            if not substring and renderer.name == search:
                output.append(renderer)
    return output
