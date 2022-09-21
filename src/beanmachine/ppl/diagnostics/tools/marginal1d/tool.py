# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Marginal 1D diagnostic tool for a Bean Machine model."""

from typing import TypeVar

from beanmachine.ppl.diagnostics.tools.marginal1d import utils
from beanmachine.ppl.diagnostics.tools.utils.base import Base
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from bokeh.embed import file_html
from bokeh.models.callbacks import CustomJS
from bokeh.resources import INLINE


T = TypeVar("T", bound="Marginal1d")


class Marginal1d(Base):
    """Marginal (1D) tool.

    Parameters
    ----------
    mcs : MonteCarloSamples
        Bean Machine model object.

    Attributes
    ----------
    data : Dict[str, List[List[float]]]
        JSON serialized version of the Bean Machine model.
    rv_names : List[str]
        The list of random variables string names for the given model.
    num_chains : int
        The number of chains of the model.
    palette : List[str]
        A list of color values used for the glyphs in the figures. The colors are
        specifically chosen from the Colorblind palette defined in Bokeh.
    js : str
        The JavaScript callbacks needed to render the Bokeh tool independently from
        a Python server.
    """

    def __init__(self: T, mcs: MonteCarloSamples) -> None:
        super(Marginal1d, self).__init__(mcs)

    def create_document(self: T) -> str:
        """Create the Bokeh document for display in Jupyter."""
        # Initialize widget values using Python.
        rv_name = self.rv_names[0]
        bw_factor = 1.0
        hdi_probability = 0.89

        # Compute the initial data displayed in the tool using Python. It is not quite
        # clear how to initialize the tool without requiring Python to first render it
        # using data it calculates.
        rv_data = self.data[rv_name]
        computed_data = utils.compute_data(
            data=rv_data,
            bw_factor=bw_factor,
            hdi_probability=hdi_probability,
        )
        bandwidth = computed_data["marginal"]["distribution"]["bandwidth"]

        # Create the Bokeh sources using Python with data calculated in Python.
        sources = utils.create_sources(data=computed_data)

        # Create the figures for the tool using Python.
        figures = utils.create_figures(rv_name=rv_name)

        # Create the glyphs and attach them to the figures using Python.
        glyphs = utils.create_glyphs(data=computed_data)
        utils.add_glyphs(sources=sources, figures=figures, glyphs=glyphs)

        # Create the annotations and attach them to the figures using Python.
        annotations = utils.create_annotations(sources=sources)
        utils.add_annotations(figures=figures, annotations=annotations)

        # Create the tool tips and attach them to the figures using Python.
        tooltips = utils.create_tooltips(figures=figures, rv_name=rv_name)
        utils.add_tooltips(figures=figures, tooltips=tooltips)

        # Create the widgets for the tool using Python.
        widgets = utils.create_widgets(
            rv_names=self.rv_names,
            rv_name=rv_name,
            bandwidth=bandwidth,
            bw_factor=bw_factor,
        )

        # Below we create callbacks for the widgets using JavaScript. The below JS uses
        # a try/catch block to ensure the proper methods are in the notebook output
        # cell no matter which widget a user interacts with first.
        # NOTE: When this callback is invoked, data is no longer being calculated with
        #       Python, and is instead being calculated by the browser.
        callback_js = f"""
            const rvName = widgets.rv_select.value;
            const rvData = data[rvName].flat();
            let bw = 0.0;
            try {{
              bw = marginal1d.update(
                rvData,
                rvName,
                bwFactor,
                hdiProbability,
                sources,
                figures,
              );
            }} catch (error) {{
              {self.js}
              bw = marginal1d.update(
                rvData,
                rvName,
                bwFactor,
                hdiProbability,
                sources,
                figures,
              );
            }}
        """

        # Each widget requires the following dictionary for the CustomJS method. Notice
        # that the callback_js object above uses the names defined as keys in the below
        # object with values defined by the Python objects.
        callback_arguments = {
            "data": self.data,
            "widgets": widgets,
            "sources": sources,
            "figures": figures,
        }

        # Each widget requires slightly different JS, except for the sliders.
        rv_select_js = f"""
            const bwFactor = 1.0;
            const hdiProbability = 0.89;
            widgets.bw_factor_slider.value = bwFactor;
            widgets.hdi_slider.value = 100 * hdiProbability;
            {callback_js};
            widgets.bw_div.text = `Bandwidth: ${{bwFactor * bw}}`;
            figures.marginal.reset.emit();
        """
        slider_js = f"""
            const bwFactor = widgets.bw_factor_slider.value;
            const hdiProbability = widgets.hdi_slider.value / 100;
            {callback_js};
            widgets.bw_div.text = `Bandwidth: ${{bwFactor * bw}}`;
        """
        rv_select_callback = CustomJS(args=callback_arguments, code=rv_select_js)
        slider_callback = CustomJS(args=callback_arguments, code=slider_js)

        # Below is where we tell Python to use the JS callbacks based on user
        # interaction.
        widgets["rv_select"].js_on_change("value", rv_select_callback)
        widgets["bw_factor_slider"].js_on_change("value", slider_callback)
        widgets["hdi_slider"].js_on_change("value", slider_callback)

        # Create the tool view using Python. Note that all subsequent callbacks will be
        # performed by JavaScript.
        tool_view = utils.create_view(figures=figures, widgets=widgets)
        return file_html(tool_view, resources=INLINE)
