"""Autocorrelation diagnostic tool types for a Bean Machine model."""
from typing import Any, Dict, List, TypedDict

from bokeh.models.annotations import BoxAnnotation
from bokeh.models.glyphs import Quad
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.sliders import RangeSlider
from bokeh.plotting.figure import Figure


Figure_names = None | List[str]

# NOTE: These are the types pyre gives us when using `reveal_type(...)` on the outputs
# of the methods.
Data = Dict[Any, Any]
Sources = Dict[Any, Any]
Figures = Dict[Any, Any]
Glyphs = Dict[Any, Any]
Annotations = Dict[Any, Any]
Tooltips = Dict[Any, Any]
Widgets = Dict[str, RangeSlider | Select]

# NOTE: TypedDict objects are for reference only. Due to the way pyre accesses keys in
#       dictionaries, and how NumPy casts arrays when using tolist(), we are unable to
#       use them, but they provide semantic information for the different types.


class _QuadData(TypedDict):
    left: List[float]
    top: List[float]
    right: List[float]
    bottom: List[float]


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
_Data = Dict[str, _FigureData]
_Sources = Dict[str, _FigureSources]
_Figures = Dict[str, Figure]
_Glyphs = Dict[str, _FigureGlyphs]
_Annotations = Dict[str, _FigureAnnotations]
_Tooltips = Dict[str, _FigureTooltips]
