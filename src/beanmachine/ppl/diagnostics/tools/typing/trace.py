"""Marginal 2D diagnostic tool types for a Bean Machine model."""
from __future__ import annotations

from typing import Any, TypedDict

from beanmachine.ppl.diagnostics.tools.typing import marginal1d as m1d

from bokeh.models.annotations import Legend
from bokeh.models.glyphs import Circle, Line, Quad
from bokeh.models.sources import ColumnDataSource
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.sliders import Slider
from bokeh.plotting.figure import Figure


# NOTE: These are the types pyre gives us when using `reveal_type(...)` on the outputs
# of the methods.
Data = dict[Any, Any]
Sources = dict[Any, Any]
Figures = dict[Any, Any]
Glyphs = dict[Any, Any]
Annotations = dict[str, dict[str, Legend]]
Tooltips = dict[Any, Any]
Widgets = dict[str, Select | Slider]

# NOTE: TypedDict objects are for reference only. Due to the way pyre accesses keys in
#       dictionaries, and how NumPy casts arrays when using tolist(), we are unable to
#       use them, but they provide semantic information for the different types.


class _DistributionData(TypedDict):
    x: list[float]
    y: list[float]
    bandwidth: list[float]


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


class _MarginalDataSingleChain(TypedDict):
    distribution: _DistributionData
    hdi: m1d.HDIData
    stats: _StatsData
    labels: _LabelsData


class _ForestLineData(TypedDict):
    x: list[float]
    y: list[float]


class _ForestCircleData(TypedDict):
    x: list[float]
    y: list[float]


class _ForestDataSingleChain(TypedDict):
    line: _ForestLineData
    circle: _ForestCircleData
    chain: int
    mean: float


class _TraceLineData(TypedDict):
    x: list[float]
    y: list[float]


class _TraceDataSingleChain(TypedDict):
    line: _TraceLineData
    chain: int
    mean: float


class _RankQuadData(TypedDict):
    left: list[float]
    top: list[float]
    right: list[float]
    bottom: list[float]
    draws: list[str]
    rank: list[float]


class _RankLineData(TypedDict):
    x: list[float]
    y: list[float]


class _RankDataSingleChain(TypedDict):
    quad: _RankQuadData
    line: _RankLineData
    chain: int
    rank_mean: float
    mean: float


MarginalData = dict[str, _MarginalDataSingleChain]
ForestData = dict[str, _ForestDataSingleChain]
TraceData = dict[str, _TraceDataSingleChain]
RankData = dict[str, _RankDataSingleChain]


class _Data(TypedDict):
    marginals: MarginalData
    forests: ForestData
    traces: TraceData
    ranks: RankData


class _MarginalSourceSingleChain(TypedDict):
    line: ColumnDataSource


class _ForestSourceSingleChain(TypedDict):
    line: ColumnDataSource
    circle: ColumnDataSource


class _TraceSourceSingleChain(TypedDict):
    line: ColumnDataSource


class _RankSourceSingleChain(TypedDict):
    quad: ColumnDataSource
    line: ColumnDataSource


MarginalSources = dict[str, _MarginalSourceSingleChain]
ForestSources = dict[str, _ForestSourceSingleChain]
TraceSources = dict[str, _TraceSourceSingleChain]
RankSources = dict[str, _RankSourceSingleChain]


class _FigureSources(TypedDict):
    marginals: MarginalSources
    forests: ForestSources
    traces: TraceSources
    ranks: RankSources


_Sources = dict[str, _FigureSources]


class _Figures(TypedDict):
    marginals: Figure
    forests: Figure
    traces: Figure
    ranks: Figure


class _MarginalLineGlyph(TypedDict):
    glyph: Line
    hover_glyph: Line


class _MarginalGlyphSingleChain(TypedDict):
    line: _MarginalLineGlyph


class _ForestLineGlyph(TypedDict):
    glyph: Line
    hover_glyph: Line


class _ForestCircleGlyph(TypedDict):
    glyph: Circle
    hover_glyph: Circle


class _ForestGlyphSingleChain(TypedDict):
    line: _ForestLineGlyph
    circle: _ForestCircleGlyph


class _TraceLineGlyph(TypedDict):
    glyph: Line
    hover_glyph: Line


class _TraceGlyphSingleChain(TypedDict):
    line: _TraceLineGlyph


class _RankQuadGlyph(TypedDict):
    glyph: Quad
    hover_glyph: Quad


class _RankLineGlyph(TypedDict):
    glyph: Line
    hover_glyph: Line


class _RankGlyphSingleChain(TypedDict):
    quad: _RankQuadGlyph
    line: _RankLineGlyph


MarginalGlyphs = dict[str, _MarginalGlyphSingleChain]
ForestGlyphs = dict[str, _ForestGlyphSingleChain]
TraceGlyphs = dict[str, _TraceGlyphSingleChain]
RankGlyphs = dict[str, _RankGlyphSingleChain]


class _FigureGlyphs(TypedDict):
    marginals: MarginalGlyphs
    forests: ForestGlyphs
    traces: TraceGlyphs
    ranks: RankGlyphs


_Glyphs = dict[str, _FigureGlyphs]

_FigureAnnotations = dict[str, Legend]
_Annotations = dict[str, _FigureAnnotations]


class _Widgets(TypedDict):
    rv_select: Select
    bw_factor_slider: Slider
    hdi_slider: Slider
