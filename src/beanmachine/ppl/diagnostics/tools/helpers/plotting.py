# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Plotting utilities for the diagnostics tools."""
from typing import Dict, List

from bokeh.core.property.nullable import Nullable
from bokeh.core.property.primitive import Null
from bokeh.core.property.wrappers import PropertyValueList
from bokeh.models.renderers import GlyphRenderer
from bokeh.models.tools import ProxyToolbar, ResetTool, SaveTool, ToolbarBox
from bokeh.palettes import Colorblind
from bokeh.plotting.figure import Figure


def style_figure(figure: Figure) -> None:
    """Give the given figure a uniform style.

    Parameters
    ----------
    fig : Figure
        Bokeh Figure object.

    Returns
    -------
    None
        Directly applies a uniform styling to the given figure.
    """
    figure.grid.grid_line_alpha = 0.3
    figure.grid.grid_line_color = "grey"
    figure.grid.grid_line_width = 0.3
    figure.xaxis.minor_tick_line_color = "grey"
    figure.yaxis.minor_tick_line_color = "grey"


def choose_palette(n: int) -> tuple:
    """Choose an appropriate palette from Bokeh's Colorblind palette.

    Parameters
    ----------
    n : int
        The number of colors to choose from the Colorblind palette.

    Returns
    -------
    tuple
        A tuple object that contains color information.
    """
    palette_indices = [key for key in Colorblind.keys() if n <= key]
    if not palette_indices:
        palette_index = max(Colorblind.keys())
    else:
        palette_index = min(palette_indices)
    return Colorblind[palette_index]


def create_toolbar(figures: Dict[str, Figure]) -> ToolbarBox:
    """Create a single toolbar for multiple figures.

    This will also remove any ``HoverTool`` tools on the figures. These are removed from
    figure groupings as there are sometimes many different hover tools for a single
    figure. Aggregating all the hover tools into a single toolbar will make the toolbar
    look like it only has hover tools in it, and nothing else.

    Parameters
    ----------
    figures : Dict[str, Figure]
        A dictionary of Bokeh figures.

    Returns
    -------
    ToolbarBox
        A single toolbar for the given Bokeh figures.
    """
    toolbars = []
    for _, figure in figures.items():
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
