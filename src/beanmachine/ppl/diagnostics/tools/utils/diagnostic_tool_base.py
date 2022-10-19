# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Base class for diagnostic tools of a Bean Machine model."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Mapping

from beanmachine.ppl.diagnostics.tools import JS_DIST_DIR
from beanmachine.ppl.diagnostics.tools.utils import plotting_utils
from beanmachine.ppl.diagnostics.tools.utils.model_serializers import serialize_bm
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from bokeh.embed import file_html, json_item
from bokeh.models import Model
from bokeh.resources import INLINE


class DiagnosticToolBaseClass(ABC):
    """
    Base class for visual diagnostic tools.

    Args:
        mcs (MonteCarloSamples): The return object from running a Bean Machine model.

    Attributes:
        data (Dict[str, List[List[float]]]): JSON serializable representation of the
            given `mcs` object.
        rv_names (List[str]): The list of random variables string names for the given
            model.
        num_chains (int): The number of chains of the model.
        num_draws (int): The number of draws of the model for each chain.
        palette (List[str]): A list of color values used for the glyphs in the figures.
            The colors are specifically chosen from the Colorblind palette defined in
            Bokeh.
        tool_js (str):The JavaScript callbacks needed to render the Bokeh tool
            independently from a Python server.
    """

    @abstractmethod
    def __init__(self: DiagnosticToolBaseClass, mcs: MonteCarloSamples) -> None:
        self.data = serialize_bm(mcs)
        self.rv_names = ["Select a random variable..."] + list(self.data.keys())
        self.num_chains = mcs.num_chains
        self.num_draws = mcs.get_num_samples()
        self.palette = plotting_utils.choose_palette(self.num_chains)
        self.tool_js = self.load_tool_js()

    def load_tool_js(self: DiagnosticToolBaseClass) -> str:
        """
        Load the JavaScript for the diagnostic tool.

        Tools must be built by `yarn` in order for this method to find the appropriate
        file. If no file is found, then the tools will not function, and an error will
        be shown to the user.

        Returns:
            str: A string containing all the JavaScript necessary to run the tool in a
                notebook.

        Raises:
            FileNotFoundError: Raised if the diagnostic tool has not been built by
                `yarn` prior to its use.
        """
        name = self.__class__.__name__
        name_tokens = re.findall(r"[A-Z][^A-Z]*", name)
        name = "_".join(name_tokens)
        path = JS_DIST_DIR.joinpath(f"{name.lower()}.js")
        with path.open() as f:
            tool_js = f.read()
        return tool_js

    def show(self: DiagnosticToolBaseClass) -> None:
        """
        Show the diagnostic tool in the notebook.

        This method uses IPython's `display` and `HTML` methods in order to display the
        diagnostic tool in a notebook. The Bokeh `Model` object returned by the
        `create_document` method is converted to HTML using Bokeh's `file_html` method.
        The advantage of encapsulating the tool in this manner is that it allows all the
        JavaScript needed to render the tool to exist in the output cell of the
        notebook. Doing so will allow the Bokeh Application to render in Google's Colab
        or Meta's Bento notebooks, which do not allow calls to third party JavaScript to
        be loaded and executed. The disadvantage is that it embeds duplicate JavaScript
        if more than one tool is used in a notebook.
        """
        # import Ipython only when we need to render the plot, so that we don't
        # need to have jupyter notebook as one of our dependencies
        from IPython.display import display, HTML

        doc = self.create_document()
        html = file_html(doc, resources=INLINE, template=self.html_template())
        display(HTML(html))

    def html_template(self: DiagnosticToolBaseClass) -> str:
        """
        HTML template object used to inject CSS styles for Bokeh Applications.

        We inject CSS into the output for the diagnostic tools because we need users to
        interact with the tool before it renders any statistics. The reason for this is
        due to the lack of a callback between Bokeh and BokehJS for an "on load" event.
        The necessary callback for an "on load" event is being worked on for the Bokeh
        3.0 release. Until Bokeh 3.0 is released, this is a work-around that makes the
        user interact with the tool before any rendering occurs.

        Returns:
            str: Template for injecting CSS in the HTML returned by `create_document`.
        """
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
    def create_document(self: DiagnosticToolBaseClass) -> Model:
        """To be implemented by the inheriting class."""
        ...

    def _tool_json(self: DiagnosticToolBaseClass) -> Mapping[Any, Any]:
        """
        Debugging method used primarily when creating a new diagnostic tool.

        Returns:
            Dict[Any, Any]: Creates a JSON serializable object using Bokeh's `json_item`
                method and the output from `create_document`.
        """
        doc = self.create_document()
        json_data = json_item(doc)
        return json_data
