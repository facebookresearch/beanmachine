# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Marginal 1D diagnostic tool for a Bean Machine model."""
from __future__ import annotations

from beanmachine.ppl.diagnostics.tools.marginal1d import utils
from beanmachine.ppl.diagnostics.tools.utils.diagnostic_tool_base import (
    DiagnosticToolBaseClass,
)
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from bokeh.models import Model
from bokeh.models.callbacks import CustomJS


class Marginal1d(DiagnosticToolBaseClass):
    """
    Marginal 1D diagnostic tool.

    Args:
        mcs (MonteCarloSamples): The return object from running a Bean Machine model.

    Attributes:
        data (Dict[str, List[List[float]]]): JSON serializable representation of the
            given ``mcs`` object.
        rv_names (List[str]): The list of random variables string names for the given
            model.
        num_chains (int): The number of chains of the model.
        num_draws (int): The number of draws of the model for each chain.
        palette (List[str]): A list of color values used for the glyphs in the figures.
            The colors are specifically chosen from the ``Colorblind`` palette defined
            in Bokeh.
        tool_js (str):The JavaScript callbacks needed to render the Bokeh tool
            independently from a Python server.
    """

    def __init__(self: Marginal1d, mcs: MonteCarloSamples) -> None:
        super(Marginal1d, self).__init__(mcs)

    def create_document(self: Marginal1d) -> Model:
        """
        Create the Bokeh document for the diagnostic tool.

        Returns:
            Model: A Bokeh Model object.
        """
        # Initialize widget values using Python.
        rv_name = self.rv_names[0]
        bw_factor = 1.0
        bandwidth = 1.0

        # NOTE: We are going to use Python and Bokeh to render the tool in the notebook
        #       output cell, however, we WILL NOT use Python to calculate any of the
        #       statistics displayed in the tool. We do this so we can make the BROWSER
        #       run all the calculations based on user interactions. If we did not
        #       employ this strategy, then the initial display a user would receive
        #       would be calculated by Python, and any subsequent updates would be
        #       calculated by JavaScript. The side-effect of having two backends
        #       calculate data could cause the figures to flicker, which would not be a
        #       good end user experience.
        #
        #       Bokeh 3.0 is implementing an "on load" feature, which would nullify this
        #       requirement, and until that version is released, we have to employ this
        #       work-around.

        # Create empty Bokeh sources using Python.
        sources = utils.create_sources()

        # Create empty figures for the tool using Python.
        figures = utils.create_figures(rv_name=rv_name)

        # Create empty glyphs and attach them to the figures using Python.
        glyphs = utils.create_glyphs()
        utils.add_glyphs(sources=sources, figures=figures, glyphs=glyphs)

        # Create empty annotations and attach them to the figures using Python.
        annotations = utils.create_annotations(sources=sources)
        utils.add_annotations(figures=figures, annotations=annotations)

        # Create empty tool tips and attach them to the figures using Python.
        tooltips = utils.create_tooltips(figures=figures, rv_name=rv_name)
        utils.add_tooltips(figures=figures, tooltips=tooltips)

        # Create the widgets for the tool using Python.
        widgets = utils.create_widgets(
            rv_names=self.rv_names,
            rv_name=rv_name,
            bandwidth=bandwidth,
            bw_factor=bw_factor,
        )

        # Create the view of the tool and serialize it into HTML using static resources
        # from Bokeh. Embedding the tool in this manner prevents external CDN calls for
        # JavaScript resources, and prevents the user from having to know where the
        # Bokeh server is.
        tool_view = utils.create_view(figures=figures, widgets=widgets)

        # Create callbacks for the tool using JavaScript.
        callback_js = f"""
            const rvName = widgets.rv_select.value;
            const rvData = data[rvName].flat();
            let bw = 0.0;
            // Remove the CSS classes that dim the tool output on initial load.
            const toolTab = toolView.tabs[0];
            const toolChildren = toolTab.child.children;
            const dimmedComponent = toolChildren[1];
            dimmedComponent.css_classes = [];
            try {{
              bw = marginal1d.update(
                rvData,
                rvName,
                bwFactor,
                hdiProbability,
                sources,
                figures,
                tooltips,
                widgets,
              );
            }} catch (error) {{
              {self.tool_js}
              bw = marginal1d.update(
                rvData,
                rvName,
                bwFactor,
                hdiProbability,
                sources,
                figures,
                tooltips,
                widgets,
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
            "tooltips": tooltips,
            "toolView": tool_view,
            "widgets": widgets,
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
        button_callback = CustomJS(args=callback_arguments, code=slider_js)

        # Tell Python to use the JavaScript.
        widgets["rv_select"].js_on_change("value", rv_select_callback)
        widgets["bw_factor_slider"].js_on_change("value", slider_callback)
        widgets["hdi_slider"].js_on_change("value", slider_callback)
        widgets["stats_button"].js_on_change("active", button_callback)

        return tool_view
