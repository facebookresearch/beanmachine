"""Plotting utilities for the diagnostic widgets."""
from typing import Dict, Tuple

from bokeh.models import Annotation, Glyph
from bokeh.models.sources import ColumnDataSource
from bokeh.palettes import Colorblind
from bokeh.plotting.figure import Figure


def style_figure(fig: Figure) -> None:
    """Apply uniform style to the given figure.

    Args:
        fig (Figure): Bokeh figure to apply the style to.

    Returns:
        None: Directly manipulates the given Bokeh `Figure` object.
    """
    fig.grid.grid_line_alpha = 0.2
    fig.grid.grid_line_color = "grey"
    fig.grid.grid_line_width = 0.2
    fig.xaxis.minor_tick_line_color = "grey"
    fig.yaxis.minor_tick_line_color = "grey"


def choose_palette(n: int) -> Tuple:
    """Choose an appropriate colorblind palette, given `n`.

    Args:
        n (int): The number of colors to use for a plot. Typically this is the same
            number of glyphs you have in your plot.

    Returns:
        palette (Tuple[str]): A tuple of color strings from Bokeh's colorblind palette.
    """
    palette_indices = [key for key in Colorblind.keys() if n <= key]
    if not palette_indices:
        palette_index = max(Colorblind.keys())
    else:
        palette_index = min(palette_indices)
    palette = Colorblind[palette_index]
    return palette


def add_glyph_to_figure(
    fig: Figure,
    source: ColumnDataSource,
    glyph_dict: Dict[str, Glyph],
) -> None:
    """Add the given glyph to the given figure.

    Args:
        fig (Figure): Bokeh `Figure` object.
        source (ColumnDataSource): Bokeh `ColumnDataSource` object. This is where the
            given glyph will get data bound to it.
        glyph_dict (Dict[str, Glyph]): A dictionary of Bokeh `Glyph` objects. The
            dictionary consists of three objects:
            - `glyph`: The actual glyph to add to the figure.
            - `hover_glyph`: If hover interactions are applied, this is the glyph a user
              will see if the plot has hover tools.
            - `muted_glyph`: If a figure has a legend, and a click policy, then this
              describes how the glyph will be muted.

    Returns:
        None: Directly manipulates the given figure by binding the glyph to it.
    """
    glyph = glyph_dict["glyph"]
    hover_glyph = glyph_dict.get("hover_glyph", None)
    muted_glyph = glyph_dict.get("muted_glyph", None)
    name = glyph.name
    fig.add_glyph(
        source_or_glyph=source,
        glyph=glyph,
        hover_glyph=hover_glyph,
        muted_glyph=muted_glyph,
        name=name,
    )


def add_annotation_to_figure(fig: Figure, annotation: Annotation) -> None:
    """Add the given annotation to the given figure.

    Args:
        fig (Figure): Bokeh `Figure` object.
        annotation (Annotation): Bokeh `Annotation` object.

    Returns:
        None: Directly binds the given annotation to the given figure.
    """
    fig.add_layout(annotation)
