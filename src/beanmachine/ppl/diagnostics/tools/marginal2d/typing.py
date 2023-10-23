# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Marginal 2D diagnostic tool types for a Bean Machine model."""
from __future__ import annotations

from typing import Any, Dict, List, Union

from beanmachine.ppl.diagnostics.tools import TypedDict
from bokeh.models.annotations import Band
from bokeh.models.glyphs import Circle, Image, Line
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.markups import Div
from bokeh.models.widgets.sliders import Slider
from bokeh.plotting import Figure


# NOTE: These are the types pyre gives us when using `reveal_type(...)` on the outputs
# of the methods.
Data = Dict[
    str,
    Union[
        Dict[
            str,
            Union[
                Dict[str, List[Any]],
                Dict[str, Dict[str, List[Any]]],
                Dict[str, Union[List[Any], float]],
            ],
        ],
        Dict[
            str,
            Union[Dict[str, List[Any]], Dict[str, Dict[str, Dict[str, List[Any]]]]],
        ],
        Dict[str, Union[Dict[str, List[Any]], Dict[str, Union[List[Any], float]]]],
    ],
]
Sources = Dict[
    str,
    Union[
        Dict[str, Union[Dict[str, Dict[str, ColumnDataSource]], ColumnDataSource]],
        Dict[str, Union[Dict[str, ColumnDataSource], ColumnDataSource]],
        Dict[str, ColumnDataSource],
    ],
]
Figures = Dict[str, Any]
Glyphs = Dict[
    str,
    Union[
        Dict[
            str,
            Union[
                Dict[str, Dict[str, Dict[str, Line]]],
                Dict[str, Circle],
            ],
        ],
        Dict[str, Union[Dict[str, Circle], Dict[str, Line]]],
    ],
]
Annotations = Dict[str, Union[Dict[str, Band], Band]]
Tooltips = Dict[
    str,
    Union[
        Dict[str, Union[Dict[str, Dict[str, HoverTool]], HoverTool]],
        Dict[str, HoverTool],
    ],
]
Widgets = Dict[str, Union[Div, Select, Slider]]


# NOTE: TypedDict objects are for reference only. Due to the way pyre accesses keys in
#       dictionaries, and how NumPy casts arrays when using tolist(), we are unable to
#       use them, but they provide semantic information for the different types.


class XSource(TypedDict):  # pyre-ignore
    distribution: ColumnDataSource
    hdi: ColumnDataSource
    stats: ColumnDataSource


class YHDISource(TypedDict):  # pyre-ignore
    lower: ColumnDataSource
    upper: ColumnDataSource


class YSource(TypedDict):  # pyre-ignore
    distribution: ColumnDataSource
    hdi: YHDISource
    stats: ColumnDataSource


class XYHDIBoundsSource(TypedDict):  # pyre-ignore
    lower: ColumnDataSource
    upper: ColumnDataSource


class XYHDISource(TypedDict):  # pyre-ignore
    x: XYHDIBoundsSource
    y: XYHDIBoundsSource


class XYSource(TypedDict):  # pyre-ignore
    distribution: ColumnDataSource
    hdi: XYHDISource
    stats: ColumnDataSource


class _Sources(TypedDict):  # pyre-ignore
    x: XSource
    y: YSource
    xy: XYSource


class _Figures(TypedDict):  # pyre-ignore
    x: Figure
    y: Figure
    xy: Figure


class LineGlyph(TypedDict):  # pyre-ignore
    glyph: Line
    hover_glyph: Line


class CircleGlyph(TypedDict):  # pyre-ignore
    glyph: Circle
    hover_glyph: Circle


class XorYGlyphs(TypedDict):  # pyre-ignore
    distribution: LineGlyph
    stats: CircleGlyph


class LowerOrUpperHDIGlyphs(TypedDict):  # pyre-ignore
    lower: LineGlyph
    upper: LineGlyph


class XYHDIGlyphs(TypedDict):  # pyre-ignore
    x: LowerOrUpperHDIGlyphs
    y: LowerOrUpperHDIGlyphs


class XYGlyphs(TypedDict):  # pyre-ignore
    distribution: Image
    hdi: XYHDIGlyphs
    stats: CircleGlyph


class _Glyphs(TypedDict):  # pyre-ignore
    x: XorYGlyphs
    y: XorYGlyphs
    xy: XYGlyphs


class YAnnotations(TypedDict):  # pyre-ignore
    lower: Band
    upper: Band


class _Annotations(TypedDict):  # pyre-ignore
    x: Band
    y: YAnnotations


class XorYTooltips(TypedDict):  # pyre-ignore
    distribution: HoverTool
    stats: HoverTool


class LowerOrUpperTooltips(TypedDict):  # pyre-ignore
    lower: HoverTool
    upper: HoverTool


class XYHDITooltips(TypedDict):  # pyre-ignore
    x: LowerOrUpperTooltips
    y: LowerOrUpperTooltips


class XYTooltips(TypedDict):  # pyre-ignore
    hdi: XYHDITooltips
    stats: HoverTool


class _Tooltips(TypedDict):  # pyre-ignore
    x: XorYTooltips
    y: XorYTooltips
    xy: XYTooltips


class _Widgets(TypedDict):  # pyre-ignore
    rv_select_x: Select
    rv_select_y: Select
    hdi_slider_x: Slider
    hdi_slider_y: Slider
    bw_div_x: Div
    bw_div_y: Div
