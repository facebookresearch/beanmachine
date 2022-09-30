# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Trace diagnostic tool types for a Bean Machine model."""

from typing import Any, Dict, List, Union

from beanmachine.ppl.diagnostics.tools import NotRequired, TypedDict
from bokeh.models.annotations import Legend
from bokeh.models.glyphs import Circle, Line, Quad
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.sliders import Slider
from bokeh.plotting.figure import Figure


# NOTE: These are the types pyre gives us when using `reveal_type(...)` on the outputs
#       of the methods.
Data = Dict[str, Dict[Any, Any]]
Sources = Dict[Any, Any]
Figures = Dict[Any, Any]
Glyphs = Dict[Any, Any]
Annotations = Dict[str, Dict[str, Legend]]
Tooltips = Dict[Any, Any]
Widgets = Dict[str, Union[Select, Slider]]


# NOTE: TypedDict objects are for reference only. Due to the way pyre accesses keys in
#       dictionaries, and how NumPy casts arrays when using tolist(), we are unable to
#       use them, but they provide semantic information for the different types. We must
#       ignore a lot of lines due to the issue discussed here
#       https://pyre-check.org/docs/errors/#13-uninitialized-attribute.


class _LineOrCircleGlyphData(TypedDict):  # pyre-ignore
    x: List[float]
    y: List[float]


class _QuadGlyphData(TypedDict):  # pyre-ignore
    """Follow the RankHistogram interface in stats/histogram.js."""

    left: List[float]
    top: List[float]
    right: List[float]
    bottom: List[float]
    chain: List[int]
    draws: List[str]
    rank: List[float]


class _MarginalDataSingleChain(TypedDict):  # pyre-ignore
    line: _LineOrCircleGlyphData
    chain: int
    mean: float
    bandwidth: float


class _ForestDataSingleChain(TypedDict):  # pyre-ignore
    line: _LineOrCircleGlyphData
    circle: _LineOrCircleGlyphData
    chain: int
    mean: float


class _TraceDataSingleChain(TypedDict):  # pyre-ignore
    line: _LineOrCircleGlyphData
    chain: int
    mean: float


class _RankDataSingleChain(TypedDict):  # pyre-ignore
    quad: _QuadGlyphData
    line: _LineOrCircleGlyphData
    chain: List[int]
    rankMean: List[float]
    mean: List[float]


_MarginalDataAllChains = Dict[str, _MarginalDataSingleChain]
_ForestDataAllChains = Dict[str, _ForestDataSingleChain]
_TraceDataAllChains = Dict[str, _TraceDataSingleChain]
_RankDataAllChains = Dict[str, _RankDataSingleChain]


class _Data(TypedDict):  # pyre-ignore
    marginals: _MarginalDataAllChains
    forests: _ForestDataAllChains
    traces: _TraceDataAllChains
    ranks: _RankDataAllChains


class _SourceSingleChain(TypedDict):  # pyre-ignore
    line: ColumnDataSource
    circle: NotRequired[ColumnDataSource]
    quad: NotRequired[ColumnDataSource]


_SourceAllChains = Dict[str, _SourceSingleChain]


class _Sources(TypedDict):  # pyre-ignore
    marginals: _SourceAllChains
    forests: _SourceAllChains
    traces: _SourceAllChains
    ranks: _SourceAllChains


class _Figures(TypedDict):  # pyre-ignore
    marginals: Figure
    forests: Figure
    traces: Figure
    ranks: Figure


class _RankTooltip(TypedDict):  # pyre-ignore
    line: HoverTool
    quad: HoverTool


class _Tooltips(TypedDict):  # pyre-ignore
    marginals: List[HoverTool]
    forests: List[HoverTool]
    traces: List[HoverTool]
    ranks: List[_RankTooltip]


class _Glyph(TypedDict):  # pyre-ignore
    glyph: Union[Circle, Line, Quad]
    hover_glyph: Union[Circle, Line, Quad]


class _GlyphSingleChain(TypedDict):  # pyre-ignore
    line: _Glyph
    circle: NotRequired[_Glyph]
    quad: NotRequired[_Glyph]


_GlyphAllChains = Dict[str, _GlyphSingleChain]


class _Glyphs(TypedDict):  # pyre-ignore
    marginals: _GlyphAllChains
    forests: _GlyphAllChains
    traces: _GlyphAllChains
    ranks: _GlyphAllChains


_Annotations = Dict[str, Dict[str, Legend]]


class _Widgets(TypedDict):  # pyre-ignore
    rv_select: Select
    bw_factor_slider: Slider
    hdi_slider: Slider
