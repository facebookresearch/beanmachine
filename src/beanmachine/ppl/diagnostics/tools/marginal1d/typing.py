# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Marginal 1D diagnostic tool types for a Bean Machine model."""

from typing import Any, Dict, List, Union

from beanmachine.ppl.diagnostics.tools import TypedDict

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
StatsAndLabelsData = Dict[str, Dict[str, Any]]
HDIData = Dict[str, Any]
Data = Dict[Any, Any]
Sources = Dict[Any, Any]
Figures = Dict[Any, Any]
Glyphs = Dict[Any, Any]
Annotations = Dict[Any, Any]
Tooltips = Dict[Any, Any]
Widgets = Dict[str, Union[Div, Select, Slider]]


# NOTE: TypedDict objects are for reference only. Due to the way pyre accesses keys in
#       dictionaries, and how NumPy casts arrays when using tolist(), we are unable to
#       use them, but they provide semantic information for the different types. We must
#       ignore a lot of lines due to the issue discussed here
#       https://pyre-check.org/docs/errors/#13-uninitialized-attribute.


class _DistributionData(TypedDict):  # pyre-ignore
    x: List[float]
    y: List[float]
    bandwidth: float


class _HDIData(TypedDict):  # pyre-ignore
    base: List[float]
    lower: List[float]
    upper: List[float]


class _StatsData(TypedDict):  # pyre-ignore
    x: List[float]
    y: List[float]
    text: List[str]


class _LabelsData(TypedDict):  # pyre-ignore
    x: List[float]
    y: List[float]
    text: List[str]
    text_align: List[str]
    x_offset: List[int]
    y_offset: List[int]


class _StatsAndLabelsData(TypedDict):  # pyre-ignore
    x: List[float]
    y: List[float]
    text: List[str]
    text_align: List[str]
    x_offset: List[int]
    y_offset: List[int]


class _GlyphData(TypedDict):  # pyre-ignore
    distribtution: _DistributionData
    hdi: _HDIData
    stats: _StatsData
    labels: _LabelsData


class _Data(TypedDict):  # pyre-ignore
    marginal: _GlyphData
    cumulative: _GlyphData


class _GlyphSources(TypedDict):  # pyre-ignore
    distribution: ColumnDataSource
    hdi: ColumnDataSource
    stats: ColumnDataSource
    labels: ColumnDataSource


class _Sources(TypedDict):  # pyre-ignore
    marginal: _GlyphSources
    cumulative: _GlyphSources


class _Figures(TypedDict):  # pyre-ignore
    marginal: Figure
    cumulative: Figure


class _DistributionGlyph(TypedDict):  # pyre-ignore
    glyph: Line
    hover_glyph: Line


class _StatsGlyph(TypedDict):  # pyre-ignore
    glyph: Circle
    hover_glyph: Circle


class _MarginalFigureGlyphs(TypedDict):  # pyre-ignore
    distribution: _DistributionGlyph
    stats: _StatsGlyph


class _CumulativeFigureGlyphs(TypedDict):  # pyre-ignore
    distribution: _DistributionGlyph
    stats: _StatsGlyph


class _Glyphs(TypedDict):  # pyre-ignore
    marginal: _MarginalFigureGlyphs
    cumulative: _CumulativeFigureGlyphs


class _MarginalFigureAnnotations(TypedDict):  # pyre-ignore
    hdi: Band
    labels: LabelSet


class _CumulativeFigureAnnotations(TypedDict):  # pyre-ignore
    hdi: Band
    labels: LabelSet


class _Annotations(TypedDict):  # pyre-ignore
    marginal: _MarginalFigureAnnotations
    cumulative: _CumulativeFigureAnnotations


class _MarginalFigureTooltips(TypedDict):  # pyre-ignore
    distribution: HoverTool
    stats: HoverTool


class _CumulativeFigureTooltips(TypedDict):  # pyre-ignore
    distribution: HoverTool
    stats: HoverTool


class _Tooltips(TypedDict):  # pyre-ignore
    marginal: _MarginalFigureTooltips
    cumulative: _CumulativeFigureTooltips


class _Widgets(TypedDict):  # pyre-ignore
    rv_select: Select
    bw_factor_slider: Slider
    bw_div: Div
    hdi_slider: Slider
