# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Trace diagnostic tool for a Bean Machine model."""

from typing import Optional, TypeVar

from beanmachine.ppl.diagnostics.tools.trace import utils
from beanmachine.ppl.diagnostics.tools.utils.base import Base
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from bokeh.embed import file_html
from bokeh.models.callbacks import CustomJS
from bokeh.resources import INLINE


T = TypeVar("T", bound="Trace")


class Trace(Base):
    """Trace tool.

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
    num_draws : int
        The number of draws of the model for each chain.
    palette : List[str]
        A list of color values used for the glyphs in the figures. The colors are
        specifically chosen from the Colorblind palette defined in Bokeh.
    js : str
        The JavaScript callbacks needed to render the Bokeh tool independently from
        a Python server.
    name : str
        The name to use when saving Bokeh JSON to disk.
    """

    def __init__(self: T, mcs: MonteCarloSamples) -> None:
        super(Trace, self).__init__(mcs)

    def create_document(self: T, name: Optional[str] = None) -> str:
        if name is not None:
            self.name = name

        # Initialize widget values using Python.
        rv_name = self.rv_names[0]

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
        sources = utils.create_sources(num_chains=self.num_chains)

        # Create empty figures for the tool using Python.
        figures = utils.create_figures(rv_name=rv_name, num_chains=self.num_chains)

        # Create empty glyphs and attach them to the figures using Python.
        glyphs = utils.create_glyphs(num_chains=self.num_chains)
        utils.add_glyphs(sources=sources, figures=figures, glyphs=glyphs)

        # Create empty annotations and attach them to the figures using Python.
        annotations = utils.create_annotations(
            figures=figures,
            num_chains=self.num_chains,
        )
        utils.add_annotations(figures=figures, annotations=annotations)

        # Create empty tool tips and attach them to the figures using Python.
        tooltips = utils.create_tooltips(
            figures=figures,
            rv_name=rv_name,
            num_chains=self.num_chains,
        )
        utils.add_tooltips(figures=figures, tooltips=tooltips)

        # Create the widgets for the tool using Python.
        widgets = utils.create_widgets(rv_names=self.rv_names, rv_name=rv_name)

        # Create callbacks for the tool using JavaScript.
        callback_js = f"""
            const rvName = widgets.rv_select.value;
            const rvData = data[rvName];
            try {{
              trace.update(
                rvData,
                rvName,
                bwFactor,
                hdiProbability,
                sources,
                figures,
                tooltips,
              );
            }} catch (error) {{
              {self.js}
              trace.update(
                rvData,
                rvName,
                bwFactor,
                hdiProbability,
                sources,
                figures,
                tooltips,
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
        }

        # Each widget requires slightly different JS.
        rv_select_js = f"""
            const bwFactor = 1.0;
            const hdiProbability = 0.89;
            widgets.bw_factor_slider.value = bwFactor;
            widgets.hdi_slider.value = 100 * hdiProbability;
            {callback_js};
            figures.marginals.reset.emit();
        """
        slider_js = f"""
            const bwFactor = widgets.bw_factor_slider.value;
            const hdiProbability = widgets.hdi_slider.value / 100;
            {callback_js};
        """
        slider_callback = CustomJS(args=callback_arguments, code=slider_js)
        rv_select_callback = CustomJS(args=callback_arguments, code=rv_select_js)

        # Tell Python to use the JavaScript.
        widgets["rv_select"].js_on_change("value", rv_select_callback)
        widgets["bw_factor_slider"].js_on_change("value", slider_callback)
        widgets["hdi_slider"].js_on_change("value", slider_callback)

        # Create the view of the tool and serialize it into HTML using static resources
        # from Bokeh. Embedding the tool in this manner prevents external CDN calls for
        # JavaScript resources, and prevents the user from having to know where the
        # Bokeh server is.
        tool_view = utils.create_view(figures=figures, widgets=widgets)
        output = file_html(tool_view, resources=INLINE)
        return output
