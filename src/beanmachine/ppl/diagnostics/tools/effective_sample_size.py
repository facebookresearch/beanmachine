"""Effective Sample Size (ESS) diagnostic tool for a Bean Machine model."""
from typing import Any, TypeVar

import arviz as az

import beanmachine.ppl.diagnostics.tools.helpers.effective_sample_size as tool
from bokeh.core.enums import LegendClickPolicy
from bokeh.models.callbacks import CustomJS
from bokeh.plotting import show


T = TypeVar("T", bound="EffectiveSampleSize")


class EffectiveSampleSize:
    """Effective Sample Size (ESS) diagnostic tool."""

    def __init__(self: T, idata: az.InferenceData) -> None:
        """Initialize."""
        self.idata = idata
        self.rv_identifiers = list(self.idata["posterior"].data_vars)
        self.rv_names = sorted(
            [str(rv_identifier) for rv_identifier in self.rv_identifiers],
        )
        self.num_chains = self.idata["posterior"].dims["chain"]

    def modify_doc(self: T, doc: Any) -> None:
        """Modify the Jupyter document in order to display the tool."""
        # Initialize the widgets.
        rv_name = self.rv_names[0]
        rv_identifier = self.rv_identifiers[self.rv_names.index(rv_name)]

        # Compute the initial data displayed in the tool.
        rv_data = self.idata["posterior"][rv_identifier].values
        computed_data = tool.compute_data(rv_data)

        # Create the Bokeh source(s).
        sources = tool.create_sources(computed_data)

        # Create the figure(s).
        figures = tool.create_figures()

        # Create the glyph(s) and attach them to the figure(s).
        glyphs = tool.create_glyphs()
        tool.add_glyphs(figures, glyphs, sources)

        # Create the annotation(s) and attache them to the figure(s).
        annotations = tool.create_annotations(figures)
        annotations["ess"]["legend"].click_policy = LegendClickPolicy.hide
        tool.add_annotations(figures, annotations)

        # Create the tool tip(s) and attach them to the figure(s).
        tooltips = tool.create_tooltips(figures)
        tool.add_tooltips(figures, tooltips)

        # Create the widget(s) for the tool.
        widgets = tool.create_widgets(rv_name, self.rv_names)

        # Create the callback(s) for the widget(s).
        def update_rv_select(attr: Any, old: str, new: str) -> None:
            rv_name = new
            rv_identifier = self.rv_identifiers[self.rv_names.index(rv_name)]
            rv_data = self.idata["posterior"][rv_identifier].values
            tool.update(rv_data, sources)

        widgets["rv_select"].on_change("value", update_rv_select)
        # NOTE: We are using Bokeh's CustomJS model in order to reset the ranges of the
        #       figures.
        widgets["rv_select"].js_on_change(
            "value",
            CustomJS(args={"p": list(figures.values())[0]}, code="p.reset.emit()"),
        )

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
