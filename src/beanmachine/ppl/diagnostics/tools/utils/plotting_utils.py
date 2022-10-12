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
    """
    Style the given Bokeh `figure`.

    Args:
        figure (Figure): A Bokeh `Figure` object.

    Returns:
        None: Styles the given figure without copying.
    """
    figure.grid.grid_line_alpha = 0.3
    figure.grid.grid_line_color = "grey"
    figure.grid.grid_line_width = 0.3
    figure.xaxis.minor_tick_line_color = "grey"
    figure.yaxis.minor_tick_line_color = "grey"


def choose_palette(num_colors: int) -> List[str]:
    """
    Determine which palette from Bokeh's Colorblind to use.

    Args:
        num_colors (int): The number of colors to use for the palette.

    Returns:
        List[str]: A list of colors to be used as the palette for a figure.
    """
    palette_indices = [key for key in Colorblind.keys() if num_colors <= key]
    if not palette_indices:
        palette_index = max(Colorblind.keys())
    else:
        palette_index = min(palette_indices)
    return Colorblind[palette_index]


def create_toolbar(figures: List[Figure]) -> ToolbarBox:
    """
    Create a single toolbar for the given list of figures.

    This method ignores all `HoverTool` entries in the final toolbar object. The
    rational for ignoring them is to prevent the final toolbar from having too much
    visual clutter.

    Args:
        figures (List[Figure]): A list of Bokeh `Figure` objects that all have their own
            toolbars that will be merged into one.

    Returns:
        ToolbarBox: The merged toolbar.
    """
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
    """
    Similar to Bokeh's `grid_plot` method, except we merge toolbars in this method.

    Args:
        figures (List[Figure]): A list of Bokeh `Figure` objects.
        num_figure_columns (int): The number of columns for the grid.

    Returns:
        Row: Returns a single Bokeh `Row` object that contains all the given figures.
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
    """
    Find renderers in the given figure using the `search` string.

    Filters renderers from the given figure based on renderer names that match the given
    search parameters.

    Args:
        figure (Figure): A Bokeh `Figure` object.
        search (str): A string to filter renderer names with.
        glyph_type (:obj:`str`, optional): The type of renderer to search for in the
            figure. Default is `GlyphRenderer`.
        substring (:obj:`bool`, optional): Flag to indicate if the given `search` string
            should be used as a substring search. Default is `False`.

    Returns:
        List[GlyphRenderer]: A list of renderers that match the search parameters.
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
