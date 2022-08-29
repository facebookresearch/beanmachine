"""Autocorrelation diagnostic tool for a Bean Machine model."""
from typing import Any, Tuple, TypeVar

import arviz as az

import beanmachine.ppl.diagnostics.tools.helpers.autocorrelation as tool
from bokeh.models.callbacks import CustomJS
from bokeh.plotting import show

T = TypeVar("T", bound="Autocorrelation")


class Autocorrelation:
    """Autocorrelation diagnostic tool."""

    def __init__(self: T, idata: az.InferenceData) -> None:
        """Initialize."""
        self.idata = idata
        self.rv_identifiers = list(self.idata["posterior"].data_vars)
        self.rv_names = sorted(
            [str(rv_identifier) for rv_identifier in self.rv_identifiers],
        )
        self.num_chains = self.idata["posterior"].dims["chain"]
        self.num_draws = self.idata["posterior"].dims["draw"]

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
        figures = tool.create_figures(self.num_chains)

        # Create the glyph(s) and attach them to the figure(s).
        glyphs = tool.create_glyphs(self.num_chains)
        tool.add_glyphs(figures, glyphs, sources)

        # Create the annotation(s) and attache them to the figure(s).
        annotations = tool.create_annotations(computed_data)
        tool.add_annotations(figures, annotations)

        # Create the tool tip(s) and attach them to the figure(s).
        tooltips = tool.create_tooltips(figures)
        tool.add_tooltips(figures, tooltips)

        # Create the widget(s) for the tool.
        widgets = tool.create_widgets(rv_name, self.rv_names, self.num_draws)

        # Create the callback(s) for the widget(s).
        def update_rv_select(attr: Any, old: str, new: str) -> None:
            rv_name = new
            rv_identifier = self.rv_identifiers[self.rv_names.index(rv_name)]
            rv_data = self.idata["posterior"][rv_identifier].values
            tool.update(rv_data, sources)
            end = 10 if self.num_draws <= 2 * 100 else 100
            widgets["range_slider"].value = (0, end)

        def update_range_slider(
            attr: Any,
            old: Tuple[int, int],
            new: Tuple[int, int],
        ) -> None:
            fig = figures[list(figures.keys())[0]]
            fig.x_range.start, fig.x_range.end = new

        widgets["rv_select"].on_change("value", update_rv_select)
        # NOTE: We are using Bokeh's CustomJS model in order to reset the ranges of the
        #       figures.
        widgets["rv_select"].js_on_change(
            "value",
            CustomJS(args={"p": list(figures.values())[0]}, code="p.reset.emit()"),
        )
        widgets["range_slider"].on_change("value", update_range_slider)

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
