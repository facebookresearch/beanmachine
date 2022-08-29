# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Marginal 2D diagnostic tool for a Bean Machine model."""
from typing import Any, TypeVar

import arviz as az

import beanmachine.ppl.diagnostics.tools.helpers.marginal2d as tool
from bokeh.models.callbacks import CustomJS
from bokeh.plotting import show


T = TypeVar("T", bound="Marginal2d")


class Marginal2d:
    """Marginal2d diagnostic tool."""

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
        x_rv_name = self.rv_names[0]
        y_rv_name = self.rv_names[1]
        x_hdi_probability = 0.89  # az.rcParams["stats.hdi_prob"]
        y_hdi_probability = 0.89  # az.rcParams["stats.hdi_prob"]
        bw_factor = 1.0
        x_rv_identifier = self.rv_identifiers[self.rv_names.index(x_rv_name)]
        y_rv_identifier = self.rv_identifiers[self.rv_names.index(y_rv_name)]

        # Compute the initial data displayed in the tool.
        x_rv_data = self.idata["posterior"][x_rv_identifier].values
        y_rv_data = self.idata["posterior"][y_rv_identifier].values
        computed_data = tool.compute_data(
            x_data=x_rv_data,
            y_data=y_rv_data,
            x_label=x_rv_name,
            y_label=y_rv_name,
            x_hdi_probability=x_hdi_probability,
            y_hdi_probability=y_hdi_probability,
            bw_factor=bw_factor,
        )
        x_bw = computed_data["x"]["distribution"]["bandwidth"]
        y_bw = float(computed_data["y"]["distribution"]["bandwidth"])

        # Create the Bokeh source(s).
        sources = tool.create_sources(computed_data)

        # Create the figure(s).
        figures = tool.create_figures(x_rv_name, y_rv_name)

        # Create the glyph(s) and attach them to the figure(s).
        glyphs = tool.create_glyphs(computed_data)
        tool.add_glyphs(figures, glyphs, sources)

        # Create the annotation(s) and attache them to the figure(s).
        annotations = tool.create_annotations(sources)
        tool.add_annotations(figures, annotations)

        # Create the tool tip(s) and attach them to the figure(s).
        tooltips = tool.create_tooltips(x_rv_name, y_rv_name, figures)
        tool.add_tooltips(figures, tooltips)

        # Create the widget(s) for the tool.
        widgets = tool.create_widgets(
            x_rv_name,
            y_rv_name,
            self.rv_names,
            bw_factor,
            x_bw,
            y_bw,
        )

        # Create the callback(s) for the widget(s).
        def update_x_rv_select(attr: Any, old: str, new: str) -> None:
            x_rv_name = new
            y_rv_name = widgets["y_rv_select"].value
            bw_factor = 1.0
            x_hdi_probability = 0.89
            y_hdi_probability = 0.89
            x_rv_identifier = self.rv_identifiers[self.rv_names.index(x_rv_name)]
            y_rv_identifier = self.rv_identifiers[self.rv_names.index(y_rv_name)]
            x_rv_data = self.idata["posterior"][x_rv_identifier].values
            y_rv_data = self.idata["posterior"][y_rv_identifier].values
            x_bw, y_bw = tool.update(
                x_rv_data=x_rv_data,
                y_rv_data=y_rv_data,
                x_rv_name=x_rv_name,
                y_rv_name=y_rv_name,
                sources=sources,
                figures=figures,
                tooltips=tooltips,
                x_hdi_probability=x_hdi_probability,
                y_hdi_probability=y_hdi_probability,
                bw_factor=bw_factor,
            )
            widgets["x_hdi_probability"] = 100 * x_hdi_probability
            widgets["y_hdi_probability"] = 100 * y_hdi_probability
            widgets["bw_factor"] = bw_factor
            widgets["x_bw_div"].text = f"Bandwidth {x_rv_name}: {x_bw * bw_factor}"
            widgets["y_bw_div"].text = f"Bandwidth {y_rv_name}: {y_bw * bw_factor}"

        def update_y_rv_select(attr: Any, old: str, new: str) -> None:
            x_rv_name = widgets["x_rv_select"].value
            y_rv_name = new
            bw_factor = 1.0
            x_hdi_probability = 0.89
            y_hdi_probability = 0.89
            x_rv_identifier = self.rv_identifiers[self.rv_names.index(x_rv_name)]
            y_rv_identifier = self.rv_identifiers[self.rv_names.index(y_rv_name)]
            x_rv_data = self.idata["posterior"][x_rv_identifier].values
            y_rv_data = self.idata["posterior"][y_rv_identifier].values
            x_bw, y_bw = tool.update(
                x_rv_data=x_rv_data,
                y_rv_data=y_rv_data,
                x_rv_name=x_rv_name,
                y_rv_name=y_rv_name,
                sources=sources,
                figures=figures,
                tooltips=tooltips,
                x_hdi_probability=x_hdi_probability,
                y_hdi_probability=y_hdi_probability,
                bw_factor=bw_factor,
            )
            widgets["x_hdi_probability"] = 100 * x_hdi_probability
            widgets["y_hdi_probability"] = 100 * y_hdi_probability
            widgets["bw_factor"] = bw_factor
            widgets["x_bw_div"].text = f"Bandwidth {x_rv_name}: {x_bw * bw_factor}"
            widgets["y_bw_div"].text = f"Bandwidth {y_rv_name}: {y_bw * bw_factor}"

        def update_x_hdi_slider(attr: Any, old: int, new: int) -> None:
            x_rv_name = widgets["x_rv_select"].value
            y_rv_name = widgets["y_rv_select"].value
            bw_factor = 1.0
            x_hdi_probability = new
            y_hdi_probability = widgets["y_hdi_slider"].value
            x_rv_identifier = self.rv_identifiers[self.rv_names.index(x_rv_name)]
            y_rv_identifier = self.rv_identifiers[self.rv_names.index(y_rv_name)]
            x_rv_data = self.idata["posterior"][x_rv_identifier].values
            y_rv_data = self.idata["posterior"][y_rv_identifier].values
            x_bw, y_bw = tool.update(
                x_rv_data=x_rv_data,
                y_rv_data=y_rv_data,
                x_rv_name=x_rv_name,
                y_rv_name=y_rv_name,
                sources=sources,
                figures=figures,
                tooltips=tooltips,
                x_hdi_probability=x_hdi_probability / 100,
                y_hdi_probability=y_hdi_probability / 100,
                bw_factor=bw_factor,
            )
            widgets["x_hdi_probability"] = x_hdi_probability
            widgets["y_hdi_probability"] = y_hdi_probability
            widgets["bw_factor"] = bw_factor
            widgets["x_bw_div"].text = f"Bandwidth {x_rv_name}: {x_bw * bw_factor}"
            widgets["y_bw_div"].text = f"Bandwidth {y_rv_name}: {y_bw * bw_factor}"

        def update_y_hdi_slider(attr: Any, old: int, new: int) -> None:
            x_rv_name = widgets["x_rv_select"].value
            y_rv_name = widgets["y_rv_select"].value
            bw_factor = 1.0
            x_hdi_probability = widgets["x_hdi_slider"].value
            y_hdi_probability = new
            x_rv_identifier = self.rv_identifiers[self.rv_names.index(x_rv_name)]
            y_rv_identifier = self.rv_identifiers[self.rv_names.index(y_rv_name)]
            x_rv_data = self.idata["posterior"][x_rv_identifier].values
            y_rv_data = self.idata["posterior"][y_rv_identifier].values
            x_bw, y_bw = tool.update(
                x_rv_data=x_rv_data,
                y_rv_data=y_rv_data,
                x_rv_name=x_rv_name,
                y_rv_name=y_rv_name,
                sources=sources,
                figures=figures,
                tooltips=tooltips,
                x_hdi_probability=x_hdi_probability / 100,
                y_hdi_probability=y_hdi_probability / 100,
                bw_factor=bw_factor,
            )
            widgets["x_hdi_probability"] = x_hdi_probability
            widgets["y_hdi_probability"] = y_hdi_probability
            widgets["bw_factor"] = bw_factor
            widgets["x_bw_div"].text = f"Bandwidth {x_rv_name}: {x_bw * bw_factor}"
            widgets["y_bw_div"].text = f"Bandwidth {y_rv_name}: {y_bw * bw_factor}"

        widgets["x_rv_select"].on_change("value", update_x_rv_select)
        widgets["y_rv_select"].on_change("value", update_y_rv_select)
        # NOTE: We are using Bokeh's CustomJS model in order to reset the ranges of the
        #       figures.
        cjs = CustomJS(args={"p": list(figures.values())[0]}, code="p.reset.emit()")
        widgets["x_rv_select"].js_on_change("value", cjs)
        widgets["y_rv_select"].js_on_change("value", cjs)
        widgets["x_hdi_slider"].on_change("value", update_x_hdi_slider)
        widgets["y_hdi_slider"].on_change("value", update_y_hdi_slider)

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
