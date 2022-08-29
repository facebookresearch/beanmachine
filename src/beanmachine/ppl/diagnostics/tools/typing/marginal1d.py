"""Marginal 1D diagnostic tool types for a Bean Machine model."""
from __future__ import annotations

from typing import Any, TypedDict

from bokeh.models.annotations import Band, LabelSet
from bokeh.models.glyphs import Circle, Line
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.markups import Div
from bokeh.models.widgets.sliders import Slider
from bokeh.plotting.figure import Figure


# NOTE: These are the types pyre gives us when using `reveal_type(...)` on the outputs
# of the methods.
StatsAndLabelsData = dict[str, dict[str, Any]]
HDIData = dict[str, Any]
Data = dict[Any, Any]
Sources = dict[Any, Any]
Figures = dict[Any, Any]
Glyphs = dict[Any, Any]
Annotations = dict[Any, Any]
Tooltips = dict[Any, Any]
Widgets = dict[str, Div | Select | Slider]

# NOTE: TypedDict objects are for reference only. Due to the way pyre accesses keys in
#       dictionaries, and how NumPy casts arrays when using tolist(), we are unable to
#       use them, but they provide semantic information for the different types.


class _DistributionData(TypedDict):
    x: list[float]
    y: list[float]
    bandwidth: list[float]


class _HDIData(TypedDict):
    base: list[float]
    lower: list[float]
    upper: list[float]


class _StatsData(TypedDict):
    x: list[float]
    y: list[float]
    text: list[str]


class _LabelsData(TypedDict):
    x: list[float]
    y: list[float]
    text: list[str]
    text_align: list[str]
    x_offset: list[int]
    y_offset: list[int]


class _StatsAndLabelsData(TypedDict):
    x: list[float]
    y: list[float]
    text: list[str]
    text_align: list[str]
    x_offset: list[int]
    y_offset: list[int]


class _GlyphData(TypedDict):
    distribtution: _DistributionData
    hdi: _HDIData
    stats: _StatsData
    labels: _LabelsData


class _Data(TypedDict):
    marginal: _GlyphData
    cumulative: _GlyphData


class _GlyphSources(TypedDict):
    distribution: ColumnDataSource
    hdi: ColumnDataSource
    stats: ColumnDataSource
    labels: ColumnDataSource


class _Sources(TypedDict):
    marginal: _GlyphSources
    cumulative: _GlyphSources


class _Figures(TypedDict):
    marginal: Figure
    cumulative: Figure


class _DistributionGlyph(TypedDict):
    glyph: Line
    hover_glyph: Line


class _StatsGlyph(TypedDict):
    glyph: Circle
    hover_glyph: Circle


class _MarginalFigureGlyphs(TypedDict):
    distribution: _DistributionGlyph
    stats: _StatsGlyph


class _CumulativeFigureGlyphs(TypedDict):
    distribution: _DistributionGlyph
    stats: _StatsGlyph


class _Glyphs(TypedDict):
    marginal: _MarginalFigureGlyphs
    cumulative: _CumulativeFigureGlyphs


class _MarginalFigureAnnotations(TypedDict):
    hdi: Band
    labels: LabelSet


class _CumulativeFigureAnnotations(TypedDict):
    hdi: Band
    labels: LabelSet


class _Annotations(TypedDict):
    marginal: _MarginalFigureAnnotations
    cumulative: _CumulativeFigureAnnotations


class _MarginalFigureTooltips(TypedDict):
    distribution: HoverTool
    stats: HoverTool


class _CumulativeFigureTooltips(TypedDict):
    distribution: HoverTool
    stats: HoverTool


class _Tooltips(TypedDict):
    marginal: _MarginalFigureTooltips
    cumulative: _CumulativeFigureTooltips


class _Widgets(TypedDict):
    rv_select: Select
    bw_factor_slider: Slider
    bw_div: Div
    hdi_slider: Slider
