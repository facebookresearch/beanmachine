"""Marginal 1D diagnostic tool for a Bean Machine model."""
from typing import Any, TypeVar

import arviz as az

import beanmachine.ppl.diagnostics.tools.helpers.marginal1d as tool
from bokeh.models.callbacks import CustomJS
from bokeh.plotting import show


T = TypeVar("T", bound="Marginal1d")


class Marginal1d:
    """Marginal1d diagnostic tool."""

    bw_cache = {}

    def __init__(self: T, idata: az.InferenceData) -> None:
        """Initialize."""
        self.idata = idata
        self.rv_identifiers = list(self.idata["posterior"].data_vars)
        self.rv_names = sorted(
            [str(rv_identifier) for rv_identifier in self.rv_identifiers],
        )

    def modify_doc(self: T, doc: Any) -> None:
        """Modify the Jupyter document in order to display the tool."""
        # Initialize the widgets.
        rv_name = self.rv_names[0]
        hdi_probability = 0.89  # az.rcParams["stats.hdi_prob"]
        bw_factor = 1.0
        rv_identifier = self.rv_identifiers[self.rv_names.index(rv_name)]

        # Compute the initial data displayed in the tool.
        rv_data = self.idata["posterior"][rv_identifier].values
        computed_data = tool.compute_data(rv_data, bw_factor, hdi_probability)
        bw = float(computed_data["marginal"]["distribution"]["bandwidth"][0])

        # Create the Bokeh source(s).
        sources = tool.create_sources(computed_data)

        # Create the figure(s).
        figures = tool.create_figures(rv_name)

        # Create the glyph(s) and attach them to the figure(s).
        glyphs = tool.create_glyphs(computed_data)
        tool.add_glyphs(figures, glyphs, sources)

        # Create the annotation(s) and attache them to the figure(s).
        annotations = tool.create_annotations(sources)
        tool.add_annotations(figures, annotations)

        # Create the tool tip(s) and attach them to the figure(s).
        tooltips = tool.create_tooltips(rv_name, figures)
        tool.add_tooltips(figures, tooltips)

        # Create the widget(s) for the tool.
        widgets = tool.create_widgets(rv_name, self.rv_names, bw_factor, bw)

        # Create the callback(s) for the widget(s).
        def update_rv_select(attr: Any, old: str, new: str) -> None:
            rv_name = new
            bw_factor = 1.0
            hdi_probability = 0.89
            rv_identifier = self.rv_identifiers[self.rv_names.index(rv_name)]
            rv_data = self.idata["posterior"][rv_identifier].values
            bw = tool.update(
                rv_data,
                sources,
                figures,
                rv_name,
                bw_factor,
                hdi_probability,
            )
            widgets["bw_factor_slider"].value = bw_factor
            widgets["hdi_slider"].value = 100 * hdi_probability
            widgets["bw_div"].text = f"Bandwidth: {bw_factor * bw}"

        def update_bw_factor_slider(attr: Any, old: float, new: float) -> None:
            rv_name = widgets["rv_select"].value
            bw_factor = new
            hdi_probability = widgets["hdi_slider"].value / 100
            rv_identifier = self.rv_identifiers[self.rv_names.index(rv_name)]
            rv_data = self.idata["posterior"][rv_identifier].values
            bw = tool.update(
                rv_data,
                sources,
                figures,
                rv_name,
                bw_factor,
                hdi_probability,
            )
            widgets["bw_div"].text = f"Bandwidth: {bw_factor * bw}"

        def update_hdi_slider(attr: Any, old: int, new: int) -> None:
            rv_name = widgets["rv_select"].value
            bw_factor = widgets["bw_factor_slider"].value
            hdi_probability = new / 100
            rv_identifier = self.rv_identifiers[self.rv_names.index(rv_name)]
            rv_data = self.idata["posterior"][rv_identifier].values
            bw = tool.update(
                rv_data,
                sources,
                figures,
                rv_name,
                bw_factor,
                hdi_probability,
            )
            widgets["bw_div"].text = f"Bandwidth: {bw_factor * bw}"

        widgets["rv_select"].on_change("value", update_rv_select)
        # NOTE: We are using Bokeh's CustomJS model in order to reset the ranges of the
        #       figures.
        cjs = CustomJS(args={"p": list(figures.values())[0]}, code="p.reset.emit()")
        widgets["rv_select"].js_on_change("value", cjs)
        widgets["bw_factor_slider"].on_change("value", update_bw_factor_slider)
        widgets["hdi_slider"].on_change("value", update_hdi_slider)

        tool_view = tool.create_view(widgets, figures)
        doc.add_root(tool_view)

    def show_tool(self: T) -> None:
        """Show the diagnostic tool.

        Returns
        -------
        None
            Directly displays the tool in Jupyter.
        """
        show(self.modify_doc)
