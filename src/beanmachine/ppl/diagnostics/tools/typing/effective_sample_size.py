"""Effective Sample Size diagnostic tool types for a Bean Machine model."""
from __future__ import annotations

from typing import Any, TypedDict

from bokeh.models.annotations import Legend
from bokeh.models.glyphs import Circle, Line
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets.inputs import Select
from bokeh.plotting.figure import Figure


# NOTE: These are the types pyre gives us when using `reveal_type(...)` on the outputs
# of the methods.
Data = dict[str, dict[str, dict[str, Any]]]
Sources = dict[str, dict[str, dict[str, ColumnDataSource]]]
Figures = dict[str, Figure]
Glyphs = dict[str, dict[Any, Any]]
Annotations = dict[Any, Any]
Tooltips = dict[str, dict[Any, Any]]
Widgets = dict[str, Select]

# NOTE: TypedDict objects are for reference only. Due to the way pyre accesses keys in
#       dictionaries, and how NumPy casts arrays when using tolist(), we are unable to
#       use them, but they provide semantic information for the different types.


class _BulkData(TypedDict):
    x: list[float]
    y: list[float]


class _TailData(TypedDict):
    x: list[float]
    y: list[float]


class _RuleOfThumbData(TypedDict):
    x: list[float]
    y: list[float]
    label: list[str]


class _GlyphData(TypedDict):
    bulk: _BulkData
    tail: _TailData
    rule_of_thumb: _RuleOfThumbData


class _Data(TypedDict):
    ess: _GlyphData


class _BulkSource(TypedDict):
    line: ColumnDataSource
    circle: ColumnDataSource


class _TailSource(TypedDict):
    line: ColumnDataSource
    circle: ColumnDataSource


class _RuleOfThumbSource(TypedDict):
    line: ColumnDataSource


class _GlyphSources(TypedDict):
    bulk: _BulkSource
    tail: _TailSource
    rule_of_thumb: _RuleOfThumbSource


class _Sources(TypedDict):
    ess: _GlyphSources


class _Figures(TypedDict):
    ess: Figure


class _BulkLineGlyphs(TypedDict):
    glyph: Line
    hover_glyph: Line


class _BulkCircleGlyphs(TypedDict):
    glyph: Circle
    hover_glyph: Circle


class _BulkGlyphs(TypedDict):
    line: _BulkLineGlyphs
    circle: _BulkCircleGlyphs


class _TailLineGlyphs(TypedDict):
    glyph: Line
    hover_glyph: Line


class _TailCircleGlyphs(TypedDict):
    glyph: Circle
    hover_glyph: Circle


class _TailGlyphs(TypedDict):
    line: _TailLineGlyphs
    circle: _TailCircleGlyphs


class _RuleOfThumbGlyphs(TypedDict):
    glyph: Line
    hover_glyph: Line


class _FigureGlyphs(TypedDict):
    bulk: _BulkGlyphs
    tail: _TailGlyphs
    rule_of_thumb: _RuleOfThumbGlyphs


class _Glyphs(TypedDict):
    ess: _FigureGlyphs


class _FigureAnnotations(TypedDict):
    legend: Legend


class _Annotations(TypedDict):
    ess: _FigureGlyphs


class _FigureTooltips(TypedDict):
    bulk: HoverTool
    tail: HoverTool
    rule_of_thumb: HoverTool


class _Tooltips(TypedDict):
    ess: _FigureTooltips


class _Widgets(TypedDict):
    rv_select: Select
