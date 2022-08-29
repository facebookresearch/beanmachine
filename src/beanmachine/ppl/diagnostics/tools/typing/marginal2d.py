"""Marginal 2D diagnostic tool types for a Bean Machine model."""
from __future__ import annotations

from typing import Any, TypedDict

from beanmachine.ppl.diagnostics.tools.typing import marginal1d as m1d

from bokeh.models.annotations import Band
from bokeh.models.glyphs import Circle, Image, Line
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.markups import Div
from bokeh.models.widgets.sliders import Slider
from bokeh.plotting.figure import Figure


# NOTE: These are the types pyre gives us when using `reveal_type(...)` on the outputs
# of the methods.
XYData = dict[
    str,
    dict[str, dict[str, dict[str, Any]]] | dict[str, list[Any]],
]
YData = dict[str, dict[str, Any]]
Data = dict[str, Any]
Sources = dict[Any, Any]
Figures = dict[Any, Any]
Glyphs = dict[Any, Any]
Annotations = dict[str, dict[str, Band] | Band]
Tooltips = dict[Any, Any]
Widgets = dict[str, Div | Select | Slider]

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


class _XData(TypedDict):
    distribution: _DistributionData
    hdi: m1d.HDIData
    stats: _StatsData
    labels: _LabelsData


class _YHDIData(TypedDict):
    top: m1d.HDIData
    bottom: m1d.HDIData


class _YData(TypedDict):
    distribution: _DistributionData
    hdi: _YHDIData
    stats: _StatsData
    labels: _LabelsData


class _XYDistributionData(TypedDict):
    image: list[list[float]]
    xmin: list[float]
    xmax: list[float]
    ymin: list[float]
    ymax: list[float]
    dw: list[float]
    dh: list[float]


class _XYHDIDatum(TypedDict):
    x: list[float]
    y: list[float]


class _XYHDIDataLowerUpper(TypedDict):
    lower: _XYHDIDatum
    upper: _XYHDIDatum


class _XYHDIData(TypedDict):
    x: _XYHDIDataLowerUpper
    y: _XYHDIDataLowerUpper


class _XYStatsData(TypedDict):
    x: list[float]
    y: list[float]


class _XYLabelsData(TypedDict):
    mean: list[str]


class _XYData(TypedDict):
    distribution: _XYDistributionData
    hdi: _XYHDIData
    stats: _XYStatsData
    labels: _XYLabelsData


class _Data(TypedDict):
    x: _XData
    y: _YData
    xy: _XYData


class _XSource(TypedDict):
    distribution: ColumnDataSource
    hdi: ColumnDataSource
    stats: ColumnDataSource


class _YSource(TypedDict):
    distribution: ColumnDataSource
    hdi: ColumnDataSource
    stats: ColumnDataSource


class _XYSourceX(TypedDict):
    lower: ColumnDataSource
    upper: ColumnDataSource


class _XYSourceY(TypedDict):
    lower: ColumnDataSource
    upper: ColumnDataSource


class _XYSourceXAndY(TypedDict):
    x: _XYSourceX
    y: _XYSourceY


class _XYSource(TypedDict):
    distribution: ColumnDataSource
    hdi: _XYSourceXAndY
    stats: ColumnDataSource


class _Sources(TypedDict):
    x: _XSource
    y: _YSource
    xy: _XYSource


class _Figures(TypedDict):
    x: Figure
    y: Figure
    xy: Figure


class _XDistributionGlyph(TypedDict):
    glyph: Line
    hover_glyph: Line


class _XStatsGlyph(TypedDict):
    glyph: Circle
    hover_glyph: Circle


class _XGlyphs(TypedDict):
    distribution: _XDistributionGlyph
    stats: _XStatsGlyph


class _YDistributionGlyph(TypedDict):
    glyph: Line
    hover_glyph: Line


class _YStatsGlyph(TypedDict):
    glyph: Circle
    hover_glyph: Circle


class _YGlyphs(TypedDict):
    distribution: _YDistributionGlyph
    stats: _YStatsGlyph


class _XYHDIGlyphsXLower(TypedDict):
    glyph: Line
    hover_glyph: Line


class _XYHDIGlyphsXUpper(TypedDict):
    glyph: Line
    hover_glyph: Line


class _XYHDIGlyphsX(TypedDict):
    lower: _XYHDIGlyphsXLower
    upper: _XYHDIGlyphsXUpper


class _XYHDIGlyphsYLower(TypedDict):
    glyph: Line
    hover_glyph: Line


class _XYHDIGlyphsYUpper(TypedDict):
    glyph: Line
    hover_glyph: Line


class _XYHDIGlyphsY(TypedDict):
    lower: _XYHDIGlyphsYLower
    upper: _XYHDIGlyphsYUpper


class _XYHDIGlyphs(TypedDict):
    x: _XYHDIGlyphsX
    y: _XYHDIGlyphsY


class _XYStatsGlyph(TypedDict):
    glyph: Circle
    hover_glyph: Circle


class _XYGlyphs(TypedDict):
    distribution: Image
    hdi: _XYHDIGlyphs
    stats: _XYStatsGlyph


class _Glyphs(TypedDict):
    x: _XGlyphs
    y: _YGlyphs
    xy: _XYGlyphs


class _YAnnotations(TypedDict):
    top: Band
    bottom: Band


class _Annotations(TypedDict):
    x: Band
    y: _YAnnotations


class _XTooltips(TypedDict):
    distribution: HoverTool
    stats: HoverTool


class _XYHDITooltipsX(TypedDict):
    lower: HoverTool
    upper: HoverTool


class _XYHDITooltipsY(TypedDict):
    lower: HoverTool
    upper: HoverTool


class _XYHDITooltips(TypedDict):
    x: _XYHDITooltipsX
    y: _XYHDITooltipsY


class _XYTooltips(TypedDict):
    hdi: _XYHDITooltips
    mean: HoverTool


class _YTooltips(TypedDict):
    distribution: HoverTool
    stats: HoverTool


class _Tooltips(TypedDict):
    x: _XTooltips
    y: _YTooltips
    xy: _XYTooltips


class _Widgets(TypedDict):
    x_rv_select: Select
    y_rv_select: Select
    x_hdi_slider: Slider
    y_hdi_slider: Slider
    x_bw_div: Div
    y_bw_div: Div
