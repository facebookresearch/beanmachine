"""Autocorrelation diagnostic tool types for a Bean Machine model."""
from __future__ import annotations

from typing import Any, TypedDict

from bokeh.models.annotations import BoxAnnotation
from bokeh.models.glyphs import Quad
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.sliders import RangeSlider
from bokeh.plotting.figure import Figure


Figure_names = None | list[str]

# NOTE: These are the types pyre gives us when using `reveal_type(...)` on the outputs
# of the methods.
Data = dict[Any, Any]
Sources = dict[Any, Any]
Figures = dict[Any, Any]
Glyphs = dict[Any, Any]
Annotations = dict[Any, Any]
Tooltips = dict[Any, Any]
Widgets = dict[str, RangeSlider | Select]

# NOTE: TypedDict objects are for reference only. Due to the way pyre accesses keys in
#       dictionaries, and how NumPy casts arrays when using tolist(), we are unable to
#       use them, but they provide semantic information for the different types.


class _QuadData(TypedDict):
    left: list[float]
    top: list[float]
    right: list[float]
    bottom: list[float]


class _BoxData(TypedDict):
    bottom: float
    top: float


class _FigureData(TypedDict):
    quad: _QuadData
    box: _BoxData


class _FigureSources(TypedDict):
    quad: ColumnDataSource


class _FigureGlyphs(TypedDict):
    quad: Quad


class _FigureAnnotations(TypedDict):
    box: BoxAnnotation


class _FigureTooltips(TypedDict):
    quad: HoverTool


class _Widgets(TypedDict):
    rv_select: Select
    range_slider: RangeSlider


# NOTE: We do not have a priori information about the number of chains in the output
#       data. This is why we are not creating a TypedDict object for the Data type with
#       named keys like chain1, chain2, etc..
_Data = dict[str, _FigureData]
_Sources = dict[str, _FigureSources]
_Figures = dict[str, Figure]
_Glyphs = dict[str, _FigureGlyphs]
_Annotations = dict[str, _FigureAnnotations]
_Tooltips = dict[str, _FigureTooltips]
