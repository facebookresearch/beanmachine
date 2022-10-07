# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Base class for diagnostic tools of a Bean Machine model."""

import re
from abc import ABC, abstractmethod
from typing import TypeVar

from beanmachine.ppl.diagnostics.tools import JS_DIST_DIR
from beanmachine.ppl.diagnostics.tools.utils import plotting_utils
from beanmachine.ppl.diagnostics.tools.utils.model_serializers import serialize_bm
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from bokeh.embed import file_html
from bokeh.models import Model
from bokeh.resources import INLINE
from IPython.display import display, HTML


T = TypeVar("T", bound="Base")


class Base(ABC):
    @abstractmethod
    def __init__(self: T, mcs: MonteCarloSamples) -> None:
        self.data = serialize_bm(mcs)
        self.rv_names = ["Select a random variable..."] + list(self.data.keys())
        self.num_chains = mcs.num_chains
        self.num_draws = mcs.get_num_samples()
        self.palette = plotting_utils.choose_palette(self.num_chains)
        self.tool_js = self.load_tool_js()

    def load_tool_js(self: T) -> str:
        name = self.__class__.__name__
        name_tokens = re.findall(r"[A-Z][^A-Z]*", name)
        name = "_".join(name_tokens)
        path = JS_DIST_DIR.joinpath(f"{name.lower()}.js")
        with path.open() as f:
            tool_js = f.read()
        return tool_js

    def show(self: T) -> None:
        doc = self.create_document()
        html = file_html(doc, resources=INLINE, template=self.html_template())
        display(HTML(html))

    def html_template(self: T) -> str:
        return """
            {% block postamble %}
            <style>
            .bk.bm-tool-loading {
              overflow: hidden;
            }
            .bk.bm-tool-loading:before {
              position: absolute;
              height: 100%;
              width: 100%;
              content: '';
              z-index: 1000;
              background-color: rgb(255, 255, 255, 0.75);
              border-color: lightgray;
              background-repeat: no-repeat;
              background-position: center;
              background-size: auto 50%;
              border-width: 1px;
              cursor: progress|;
            }
            .bk.bm-tool-loading.arcs:hover:before {
              content: "Please select a Query from the Select menu above.";
              font: x-large Arial, sans-serif;
              color: black;
              cursor: progress;
              display: flex;
              justify-content: center;
              align-items: center;
            }
            </style>
            {% endblock %}
        """

    @abstractmethod
    def create_document(self: T) -> Model:
        ...
