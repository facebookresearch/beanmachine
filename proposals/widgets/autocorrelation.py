"""Autocorrelation widget."""
from typing import Dict, List

import arviz as az
import numpy as np
from bokeh.layouts import column, row
from bokeh.models import Annotation, Glyph
from bokeh.models.annotations import BoxAnnotation
from bokeh.models.callbacks import CustomJS
from bokeh.models.glyphs import Quad
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets import RangeSlider, Select
from bokeh.plotting import show
from bokeh.plotting.figure import figure, Figure

from . import utils


class AutocorrelationWidget:
    def __init__(self, idata: az.InferenceData) -> None:
        """Autocorrelation widget.

        Args:
            idata (az.InferenceData): ArviZ `InferenceData` object.
        """
        self.idata = idata

        self.rv_identifiers = list(self.idata["posterior"].data_vars)
        self.rv_names = [str(rv_identifier) for rv_identifier in self.rv_identifiers]
        self.num_chains = self.idata["posterior"].dims["chain"]
        self.num_draws_single_chain = self.idata["posterior"].dims["draw"]
        self.num_draws_all_chains = self.num_chains * self.num_draws_single_chain

    def compute(self, rv_name: str) -> Dict[str, Dict[str, List[float]]]:
        """Compute data for the widget.

        Args:
            rv_name (str): Bean Machine string representation of an `RVIdentifier`.

        Returns:
            Dict[str, Dict[str, List[float]]]: Data used for the widget.
        """
        rv_identifier = self.rv_identifiers[self.rv_names.index(rv_name)]
        data = self.idata["posterior"][rv_identifier].values

        # Calculate the autocorrelation for each chain, and the assumed Normal
        # autocorrelation.
        autocorr = az.stats.stats_utils.autocorr(data)
        output = {}
        for chain in range(self.num_chains):
            chain_data = autocorr[chain, :]
            bins = np.arange(len(chain_data) + 1)
            left = bins[:-1]
            top = chain_data
            right = bins[1:]
            bottom = np.zeros(len(chain_data))
            output[f"chain{chain}"] = {
                "left": left.tolist(),
                "top": top.tolist(),
                "right": right.tolist(),
                "bottom": bottom.tolist(),
                "chain": [chain + 1] * len(top),
                "label": chain_data.tolist(),
            }
        return output

    def create_sources(self, rv_name: str) -> Dict[str, ColumnDataSource]:
        """Create Bokeh `ColumnDataSource` objects.

        Args:
            rv_name (str): Bean Machine string representation of an `RVIdentifier`.

        Returns:
            Dict[str, ColumnDataSource]: A dictionary containing Bokeh
                `ColumnDataSource` objects.
        """
        data = self.compute(rv_name=rv_name)

        output = {}
        for chain in range(self.num_chains):
            output[f"chain{chain}"] = ColumnDataSource(data[f"chain{chain}"])
        return output

    def create_figures(self, rv_name: str) -> Dict[str, Figure]:
        """Create Bokeh `Figure` objects.

        Args:
            rv_name (str): Bean Machine string representation of an `RVIdentifier`.

        Returns:
            Dict[str, Figure]: Dictionary of Bokeh `Figure` objects.
        """
        figs = {}

        for chain in range(self.num_chains):
            fig = figure(
                plot_width=500,
                plot_height=300,
                outline_line_color="black",
                title=f"{rv_name} autocorrelation chain {chain + 1}",
                x_axis_label="Draw",
                y_axis_label="Autocorrelation",
            )
            utils.style_figure(fig)
            fig.x_range.update_from_json({"end": 100})
            fig.y_range.update_from_json({"start": -1.2})
            fig.y_range.update_from_json({"end": 1.2})
            figs[f"chain{chain}"] = fig
        keys = list(figs.keys())
        first_key = keys[0]
        first_fig = figs[first_key]
        x_range = first_fig.x_range
        y_range = first_fig.y_range
        for _, fig in figs.items():
            fig.x_range = x_range
            fig.y_range = y_range
        return figs

    def create_glyphs(self) -> Dict[str, Dict[str, Glyph]]:
        """Create Bokeh `Glyph` objects.

        Returns:
            Dict[str, Dict[str, Glyph]]: Dictionary of Bokeh `Glyph` objects.
        """
        palette = utils.choose_palette(n=self.num_chains)

        output = {}
        for chain in range(self.num_chains):
            color = palette[chain]
            glyph = Quad(
                left="left",
                top="top",
                right="right",
                bottom="bottom",
                fill_color=color,
                line_color="white",
                fill_alpha=1,
                name=f"autocorr_chain{chain}",
            )
            hover_glyph = Quad(
                left="left",
                top="top",
                right="right",
                bottom="bottom",
                fill_color=color,
                line_color="black",
                line_width=2,
                fill_alpha=1,
                name=f"autocorr_hover_chain{chain}",
            )
            muted_glyph = Quad(
                left="left",
                top="top",
                right="right",
                bottom="bottom",
                fill_color=color,
                line_color="white",
                fill_alpha=0.1,
                name=f"autocorr_muted_chain{chain}",
            )
            output[f"chain{chain}"] = {
                "glyph": glyph,
                "hover_glyph": hover_glyph,
                "muted_glyph": muted_glyph,
            }
        return output

    def create_annotations(self) -> Dict[str, Annotation]:
        """Create Bokeh `Annotation` objects.

        Returns:
            Dict[str, Annotation]: Dictionary of Bokeh `Annotation` objects.
        """
        confidence_interval = (1.96 / self.num_draws_single_chain) ** 0.5

        palette = utils.choose_palette(n=self.num_chains)
        output = {}
        for chain in range(self.num_chains):
            color = palette[chain]
            box = BoxAnnotation(
                bottom=-confidence_interval,
                top=confidence_interval,
                fill_color=color,
                fill_alpha=0.2,
            )
            output[f"chain{chain}"] = box
        return output

    def update_figures(
        self,
        figs: Dict[str, Figure],
        rv_name: str,
        old_sources: Dict[str, ColumnDataSource],
    ) -> None:
        """Update the figures in the widget.

        Args:
            figs (Dict[str, Figure]): Dictionary of Bokeh `Figure` objects.
            rv_name (str): Bean Machine string representation of an `RVIdentifier`.
            old_sources (Dict[str, ColumnDataSource]): Dictionary of Bokeh
                `ColumnDataSource` objects. These are the old sources we will be
                updating with new data based on user interactions.

        Returns:
            None: Directly updates the given figures with new data.
        """
        new_sources = self.create_sources(rv_name=rv_name)

        for key, new_cds in new_sources.items():
            old_sources[key].data = dict(new_cds.data)
            fig = figs[key]
            fig.title.text = " ".join([rv_name] + fig.title.text.split()[1:])

    def modify_doc(self, doc) -> None:
        """Modify the document by adding the widget."""
        # Set the initial view.
        rv_name = self.rv_names[0]

        # Create data sources for the figures.
        sources = self.create_sources(rv_name)

        # Create the figures.
        figs = self.create_figures(rv_name)

        # Create glyphs and add them to the figure.
        glyphs = self.create_glyphs()
        for key, glyph_dict in glyphs.items():
            fig = figs[key]
            source = sources[key]
            utils.add_glyph_to_figure(fig=fig, source=source, glyph_dict=glyph_dict)

        # Create annotations for the figure.
        annotations = self.create_annotations()
        for key, annotation in annotations.items():
            utils.add_annotation_to_figure(fig=figs[key], annotation=annotation)

        # Create tooltips for the figure.
        for key, fig in figs.items():
            tips = HoverTool(
                renderers=[
                    renderer
                    for renderer in fig.renderers
                    if renderer.name == f"autocorr_{key}"
                ],
                tooltips=[("Autocorrelation", "@top{0.000}")],  # noqa FS003 f-string
            )
            fig.add_tools(tips)

        # Widgets
        rv_select = Select(title="Query", value=rv_name, options=self.rv_names)
        range_slider = RangeSlider(
            start=0,
            end=self.num_draws_single_chain,
            value=(0, 100),
            step=100,
            title="Autocorrelation range",
        )

        def update_rv(attr, old, new):
            self.update_figures(figs=figs, rv_name=new, old_sources=sources)
            range_slider.value = (0, 100)

        def update_range(attr, old, new):
            f = figs[list(figs.keys())[0]]
            start, end = new
            f.x_range.start = start
            f.x_range.end = end

        rv_select.on_change("value", update_rv)
        # NOTE: We are using Bokeh's CustomJS model in order to reset the ranges of the
        #       figures.
        rv_select.js_on_change(
            "value",
            CustomJS(args={"p": list(figs.values())[0]}, code="p.reset.emit()"),
        )
        range_slider.on_change("value", update_range)
        range_slider.js_on_change(
            "value",
            CustomJS(args={"p": list(figs.values())[0]}, code="p.reset.emit()"),
        )

        rows = [
            row(*fig_row) for fig_row in np.array(list(figs.values())).reshape((2, -1))
        ]
        layout = column(
            rv_select,
            range_slider,
            *rows,
        )
        doc.add_root(layout)

    def show_widget(self):
        """Display the widget."""
        show(self.modify_doc)
