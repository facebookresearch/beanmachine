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
from bokeh.models.widgets.groups import RadioButtonGroup
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.markups import Div
from bokeh.models.widgets.sliders import Slider
from bokeh.plotting.figure import Figure


# NOTE: These are the types pyre gives us when using `reveal_type(...)` on the outputs
#       of the methods.
Data = Dict[
    str,
    Dict[str, Union[Dict[str, List[Any]], Dict[str, Union[List[Any], float]]]],
]
Sources = Dict[Any, Any]
Figures = Dict[Any, Any]
Glyphs = Dict[Any, Any]
Annotations = Dict[Any, Any]
Tooltips = Dict[Any, Any]
Widgets = Dict[str, Union[Div, RadioButtonGroup, Select, Slider]]


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


class _GlyphData(TypedDict):  # pyre-ignore
    distribtution: _DistributionData
    hdi: _HDIData
    stats: _StatsData
    labels: _LabelsData


class Data_(TypedDict):  # pyre-ignore
    """
    Definition of the Marginal 1D tool's data.

    Attributes:
        marginal (Dict[str, Dict[str, Union[List[int], List[float], List[str], float]]]):
            Data for the marginal figure.
        cumulative (Dict[str, Dict[str, Union[List[int], List[float], List[str], float]]]):
            Data for the cumulative figure.

    .. code-block:: python

      {
          "marginal": {
              "distribution": {
                  "x": list[float],
                  "y": list[float],
                  "bandwidth": float,
              },
              "hdi": {
                  "base": list[float],
                  "lower": list[float],
                  "upper": list[float],
              },
              "stats": {"x": list[float], "y": list[float], "text": list[str]},
              "labels": {
                  "x": list[float],
                  "y": list[float],
                  "text": list[str],
                  "text_align": list[str],
                  "x_offset": list[int],
                  "y_offset": list[int],
              },
          },
          "cumulative": {
              "distribution": {
                  "x": list[float],
                  "y": list[float],
                  "bandwidth": float,
              },
              "hdi": {
                  "base": list[float],
                  "lower": list[float],
                  "upper": list[float],
              },
              "stats": {"x": list[float], "y": list[float], "text": list[str]},
              "labels": {
                  "x": list[float],
                  "y": list[float],
                  "text": list[str],
                  "text_align": list[str],
                  "x_offset": list[int],
                  "y_offset": list[int],
              },
          },
      }
    """

    marginal: _GlyphData
    cumulative: _GlyphData


class _Source(TypedDict):  # pyre-ignore
    distribution: ColumnDataSource
    hdi: ColumnDataSource
    stats: ColumnDataSource
    labels: ColumnDataSource


class Sources_(TypedDict):  # pyre-ignore
    """
    Definition of the Marginal 1D tool's Bokeh ``ColumnDataSource`` objects.

    Attributes:
        marginal (Dict[str, ColumnDataSource]): Bokeh ``ColumnDataSource`` objects for
            the marginal figure.
        cumulative (Dict[str, ColumnDataSource]): Bokeh ``ColumnDataSource`` objects for
            the cumulative figure.

    .. code-block:: python

       {
           "marginal": {
               "distribution": ColumnDataSource({"x": list[float], "y": list[float]}),
               "hdi": ColumnDataSource(
                   {"base": list[float], "lower": list[float], "upper": list[float]}
               ),
               "stats": ColumnDataSource(
                   {"x": list[float], "y": list[float], "text": list[str]}
               ),
               "labels": ColumnDataSource(
                   {
                       "x": list[float],
                       "y": list[float],
                       "text": list[str],
                       "text_align": list[str],
                       "x_offset": list[int],
                       "y_offset": list[int],
                   }
               ),
           },
           "cumulative": {
               "distribution": ColumnDataSource({"x": list[float], "y": list[float]}),
               "hdi": ColumnDataSource(
                   {"base": list[float], "lower": list[float], "upper": list[float]}
               ),
               "stats": ColumnDataSource(
                   {"x": list[float], "y": list[float], "text": list[str]}
               ),
               "labels": ColumnDataSource(
                   {
                       "x": list[float],
                       "y": list[float],
                       "text": list[str],
                       "text_align": list[str],
                       "x_offset": list[int],
                       "y_offset": list[int],
                   }
               ),
           },
       }
    """

    marginal: _Source
    cumulative: _Source


class Figures_(TypedDict):  # pyre-ignore
    """
    Definition of the Marginal 1D tool's figures.

    Attributes:
        marginal (Figure): Bokeh ``Figure`` object for the marginal.
        cumulative (Figure): Bokeh ``Figure`` object for the cumulative.

    .. code-block:: python

       {
           "marginal": Figure,
           "cumulative": Figure,
       }
    """

    marginal: Figure
    cumulative: Figure


class _DistributionGlyph(TypedDict):  # pyre-ignore
    glyph: Line
    hover_glyph: Line


class _StatsGlyph(TypedDict):  # pyre-ignore
    glyph: Circle
    hover_glyph: Circle


class _FigureGlyphs(TypedDict):  # pyre-ignore
    distribution: _DistributionGlyph
    stats: _StatsGlyph


class Glyphs_(TypedDict):  # pyre-ignore
    """
    Definition of the Marginal 1D tool's glyphs.

    These are different than the tool's annotations, which are defined in an annotation
    object.

    Attributes:
        marginal (Dict[str, Dict[str, Union[Circle, Line]]]): Bokeh ``Glyph`` objects
            for the marginal figure.
        cumulative (Dict[str, Dict[str, Union[Circle, Line]]]): Bokeh ``Glyph`` objects
            for the cumulative figure.

    .. code-block:: python

       {
           "marginal": {
               "distribution": {"glyph": Line, "hover_glyph": Line},
               "stats": {"glyph": Circle, "hover_glyph": Circle},
           },
           "cumulative": {
               "distribution": {"glyph": Line, "hover_glyph": Line},
               "stats": {"glyph": Circle, "hover_glyph": Circle},
           },
       }
    """

    marginal: _FigureGlyphs
    cumulative: _FigureGlyphs


class _FigureAnnotations(TypedDict):  # pyre-ignore
    hdi: Band
    labels: LabelSet


class Annotations_(TypedDict):  # pyre-ignore
    """
    Definition of the Marginal 1D tool's annotations.

    These are different than the tool's glyphs, which are defined in an glyph object.

    Attributes:
        marginal (Dict[str, Union[Band, LabelSet]]): Bokeh ``Annotation`` objects for
            the marginal figure.
        cumulative (Dict[str, Union[Band, LabelSet]]): Bokeh ``Annotation`` objects for
            the cumulative figure.

    .. code-block:: python

       {
           "marginal": {"hdi": Band, "labels": LabelSet},
           "cumulative": {"hdi": Band, "labels": LabelSet},
       }
    """

    marginal: _FigureAnnotations
    cumulative: _FigureAnnotations


class _Tooltip(TypedDict):  # pyre-ignore
    distribution: HoverTool
    stats: HoverTool


class Tooltips_(TypedDict):  # pyre-ignore
    """
    Definition of the Marginal 1D tool's tooltips.

    Attributes:
        marginal (Dict[str, HoverTool]): Bokeh ``HoverTool`` objects defining the
            marginal figure's hover tools.
        cumulative (Dict[str, HoverTool]): Bokeh ``HoverTool`` objects defining the
            cumulative figure's hover tools.

    .. code-block:: python

       {
           "marginal": {"distribution": HoverTool, "stats": HoverTool},
           "cumulative": {"distribution": HoverTool, "stats": HoverTool},
       }
    """

    marginal: _Tooltip
    cumulative: _Tooltip


class Widgets_(TypedDict):  # pyre-ignore
    """
    Definition of the Marginal 1D tool's widgets.

    Attributes:
        rv_select (Select): Bokeh ``Select`` widget that contains the random variable
            names.
        bw_factor_slider (Slider): Bokeh ``Slider`` widget that allows the user to
            change the bandwidth used to calculate the Kernel Density Estimate (KDE).
        bw_div (Div): Bokeh ``Div`` widget that displays the bandwidth used when
            calculating the KDE.
        hdi_slider (Slider): Bokeh ``Slider`` widget that adjusts the Highest Density
            Interval highlighted on the tool.
        stats_button (RadioButtonGroup): Bokeh ``RadioButtonGroup`` that allows the user
            to visualize the point statistic shown by the button.

    .. code-block:: python

       {
           "rv_select": Select,
           "bw_factor_slider": Slider,
           "bw_div": Div,
           "hdi_slider": Slider,
           "stats_button": RadioButtonGroup,
       }
    """

    rv_select: Select
    bw_factor_slider: Slider
    bw_div: Div
    hdi_slider: Slider
    stats_button: RadioButtonGroup
