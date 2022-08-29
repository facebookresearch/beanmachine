# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Marginal 2D diagnostic tool types for a Bean Machine model."""
from typing import Any, Dict, List, TypedDict

from beanmachine.ppl.diagnostics.tools.typing import marginal1d as m1d

from bokeh.models.annotations import Legend
from bokeh.models.glyphs import Circle, Line, Quad
from bokeh.models.sources import ColumnDataSource
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.sliders import Slider
from bokeh.plotting.figure import Figure


# NOTE: These are the types pyre gives us when using `reveal_type(...)` on the outputs
# of the methods.
Data = Dict[Any, Any]
Sources = Dict[Any, Any]
Figures = Dict[Any, Any]
Glyphs = Dict[Any, Any]
Annotations = Dict[str, Dict[str, Legend]]
Tooltips = Dict[Any, Any]
Widgets = Dict[str, Select | Slider]

# NOTE: TypedDict objects are for reference only. Due to the way pyre accesses keys in
#       dictionaries, and how NumPy casts arrays when using tolist(), we are unable to
#       use them, but they provide semantic information for the different types.


class _DistributionData(TypedDict):
    x: List[float]
    y: List[float]
    bandwidth: List[float]


class _StatsData(TypedDict):
    x: List[float]
    y: List[float]
    text: List[str]


class _LabelsData(TypedDict):
    x: List[float]
    y: List[float]
    text: List[str]
    text_align: List[str]
    x_offset: List[int]
    y_offset: List[int]


class _MarginalDataSingleChain(TypedDict):
    distribution: _DistributionData
    hdi: m1d.HDIData
    stats: _StatsData
    labels: _LabelsData


class _ForestLineData(TypedDict):
    x: List[float]
    y: List[float]


class _ForestCircleData(TypedDict):
    x: List[float]
    y: List[float]


class _ForestDataSingleChain(TypedDict):
    line: _ForestLineData
    circle: _ForestCircleData
    chain: int
    mean: float


class _TraceLineData(TypedDict):
    x: List[float]
    y: List[float]


class _TraceDataSingleChain(TypedDict):
    line: _TraceLineData
    chain: int
    mean: float


class _RankQuadData(TypedDict):
    left: List[float]
    top: List[float]
    right: List[float]
    bottom: List[float]
    draws: List[str]
    rank: List[float]


class _RankLineData(TypedDict):
    x: List[float]
    y: List[float]


class _RankDataSingleChain(TypedDict):
    quad: _RankQuadData
    line: _RankLineData
    chain: int
    rank_mean: float
    mean: float


MarginalData = Dict[str, _MarginalDataSingleChain]
ForestData = Dict[str, _ForestDataSingleChain]
TraceData = Dict[str, _TraceDataSingleChain]
RankData = Dict[str, _RankDataSingleChain]


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


MarginalSources = Dict[str, _MarginalSourceSingleChain]
ForestSources = Dict[str, _ForestSourceSingleChain]
TraceSources = Dict[str, _TraceSourceSingleChain]
RankSources = Dict[str, _RankSourceSingleChain]


class _FigureSources(TypedDict):
    marginals: MarginalSources
    forests: ForestSources
    traces: TraceSources
    ranks: RankSources


_Sources = Dict[str, _FigureSources]


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


MarginalGlyphs = Dict[str, _MarginalGlyphSingleChain]
ForestGlyphs = Dict[str, _ForestGlyphSingleChain]
TraceGlyphs = Dict[str, _TraceGlyphSingleChain]
RankGlyphs = Dict[str, _RankGlyphSingleChain]


class _FigureGlyphs(TypedDict):
    marginals: MarginalGlyphs
    forests: ForestGlyphs
    traces: TraceGlyphs
    ranks: RankGlyphs


_Glyphs = Dict[str, _FigureGlyphs]

_FigureAnnotations = Dict[str, Legend]
_Annotations = Dict[str, _FigureAnnotations]


class _Widgets(TypedDict):
    rv_select: Select
    bw_factor_slider: Slider
    hdi_slider: Slider
