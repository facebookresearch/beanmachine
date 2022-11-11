# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Effective Sample Size (ESS) diagnostic tool types for a Bean Machine model."""
from typing import Dict, List, Union

from beanmachine.ppl.diagnostics.tools import NotRequired, TypedDict
from bokeh.models.annotations import Legend
from bokeh.models.glyphs import Circle, Line
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets.inputs import Select
from bokeh.plotting.figure import Figure


# NOTE: These are the types pyre gives us when using `reveal_type(...)` on the outputs
#       of the methods.
Data = Dict[str, Dict[str, Dict[str, List[float]]]]
Sources = Dict[str, Dict[str, Dict[str, ColumnDataSource]]]
Figures = Dict[str, Figure]
Glyphs = Dict[
    str,
    Dict[
        str,
        Union[
            Dict[str, Dict[str, Line]],
            Dict[str, Union[Dict[str, Circle], Dict[str, Line]]],
        ],
    ],
]
Annotations = Dict[str, Legend]
Tooltips = Dict[str, Dict[str, HoverTool]]
Widgets = Dict[str, Select]


# NOTE: TypedDict objects are for reference only. Due to the way pyre accesses keys in
#       dictionaries, and how NumPy casts arrays when using tolist(), we are unable to
#       use them, but they provide semantic information for the different types. We must
#       ignore a lot of lines due to the issue discussed here
#       https://pyre-check.org/docs/errors/#13-uninitialized-attribute.
class _GlyphData(TypedDict):  # pyre-ignore
    x: List[float]
    y: List[float]


class _NamedGlyphData(TypedDict):  # pyre-ignore
    bulk: Dict[str, _GlyphData]
    tail: Dict[str, _GlyphData]
    ruleOfThumb: Dict[str, _GlyphData]


class Data_(TypedDict):  # pyre-ignore
    """
    Definition of the Effective Sample Size tool's data. An example of how the data
    looks is below.

    Attributes:
        ess (Dict[str, Dict[str, Dict[str, List[Union[float, str]]]]]): Data for the
            effective sample size figure.

    .. code-block:: python

       {
           "ess": {
               "bulk": {"x": list[float], "y": list[float]},
               "tail": {"x": list[float], "y": list[float]},
               "ruleOfThumb": {"x": list[float], "y": list[float], "text": list[str]},
           }
       }

    """

    ess: Dict[str, _NamedGlyphData]


class Figures_(TypedDict):  # pyre-ignore
    """
    Definition of the Effective Sample Size tool's figures. An example of how the figure
    object looks is below.

    Attributes:
        ess (Figure): Bokeh ``Figure`` object for the effective sample size.

    .. code-block:: python

       {
           "ess": Figure,
       }
    """

    ess: Figure


class _Glyph(TypedDict):  # pyre-ignore
    glyph: Union[Circle, Line]
    hover_glyph: NotRequired[Union[Circle, Line]]


class _GlyphType(TypedDict):  # pyre-ignore
    circle: NotRequired[Dict[str, _Glyph]]
    line: Dict[str, _Glyph]


class _NamedGlyphs(TypedDict):  # pyre-ignore
    bulk: Dict[str, _GlyphType]
    tail: Dict[str, _GlyphType]
    ruleOfThumb: Dict[str, _GlyphType]


class Glyphs_(TypedDict):  # pyre-ignore
    """
    Definition of the effective sample size tool's glyphs. An example of how the glyphs
    object looks is below.

    Attributes:
        ess (Dict[str, Dict[str, Union[Dict[str, Dict[str, Line]], Dict[str, Union[Dict[str, Circle], Dict[str, Line]]]]]]):
        Bokeh ``Glyph`` objects for the ess figure.

    .. code-block:: python

       {
           "ess": {
               "bulk": {
                   "circle": {"glyph": Circle(), "hover_glyph": Circle()},
                   "line": {"glyph": Line()},
               },
               "tail": {
                   "circle": {"glyph": Circle(), "hover_glyph": Circle()},
                   "line": {"glyph": Line()},
               },
               "ruleOfThumb": {
                   "line": {"glyph": Line(), "hover_glyph": Line()},
               },
           },
       }
    """

    ess: Dict[str, _NamedGlyphs]


class Annotations_(TypedDict):  # pyre-ignore
    """
    Definition of the Effective Sample Size tool's annotations. An example of how the
    annotation object looks is below.

    Attributes:
        ess (Legend): Bokeh ``Legend`` object.

    .. code-block:: python

       {
           "ess": Legend,
       }
    """

    ess: Legend


class Tooltips_(TypedDict):  # pyre-ignore
    """
    Definition of the Effective Sample Size tool's tooltips. An example of how the
    tooltips object looks is below.

    Attributes:
        ess (Dict[str, HoverTool]): A dictionary of Bokeh ``HoverTool`` objects.

    .. code-block:: python

       {
           "ess": {
               "bulk": HoverTool,
               "tail": HoverTool,
               "ruleOfThumb": HoverTool,
           },
       }
    """

    ess: Dict[str, HoverTool]


class Widgets_(TypedDict):  # pyre-ignore
    """
    Definition of the Effective Sample Size tool's widgets. An example of how the
    widgets object looks is below.

    Attributes:
        rv_select (Select): Bokeh ``Select`` object.

    .. code-block:: python

       {
           "rv_select": Select,
       }
    """

    rv_select: Select
