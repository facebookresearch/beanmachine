# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Effective Sample Size diagnostic tool types for a Bean Machine model."""
from typing import Any, Dict, List, TypedDict

from bokeh.models.annotations import Legend
from bokeh.models.glyphs import Circle, Line
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets.inputs import Select
from bokeh.plotting.figure import Figure


# NOTE: These are the types pyre gives us when using `reveal_type(...)` on the outputs
# of the methods.
Data = Dict[str, Dict[str, Dict[str, Any]]]
Sources = Dict[str, Dict[str, Dict[str, ColumnDataSource]]]
Figures = Dict[str, Figure]
Glyphs = Dict[str, Dict[Any, Any]]
Annotations = Dict[Any, Any]
Tooltips = Dict[str, Dict[Any, Any]]
Widgets = Dict[str, Select]

# NOTE: TypedDict objects are for reference only. Due to the way pyre accesses keys in
#       dictionaries, and how NumPy casts arrays when using tolist(), we are unable to
#       use them, but they provide semantic information for the different types.


class _BulkData(TypedDict):
    x: List[float]
    y: List[float]


class _TailData(TypedDict):
    x: List[float]
    y: List[float]


class _RuleOfThumbData(TypedDict):
    x: List[float]
    y: List[float]
    label: List[str]


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
