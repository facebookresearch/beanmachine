"""Trace plot widget."""
from typing import Dict, List, Union

import arviz as az
import numpy as np
from bokeh.layouts import column, row
from bokeh.models import Annotation, Glyph
from bokeh.models.annotations import Legend, LegendItem, Span
from bokeh.models.callbacks import CustomJS
from bokeh.models.glyphs import Circle, Line, Quad
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets import Div, Panel, Select, Slider, Tabs
from bokeh.plotting import show
from bokeh.plotting.figure import figure, Figure

from . import utils


class TracePlotWidget:

    bw_cache = {}

    def __init__(self, idata: az.InferenceData) -> None:
        """Trace plot widget.

        This widget also contains rank plots.

        Args:
            idata (az.InferenceData): ArviZ `InferenceData` object.
        """
        self.idata = idata

        self.rv_identifiers = list(self.idata["posterior"].data_vars)
        self.rv_names = [str(rv_identifier) for rv_identifier in self.rv_identifiers]
        self.num_chains = self.idata["posterior"].dims["chain"]
        self.hdi_prob = az.rcParams["stats.hdi_prob"]

    def compute(
        self,
        rv_name: str,
        bw_fct: float,
        hdi: float,
    ) -> Dict[str, Dict[str, List[Union[int, float]]]]:
        """Compute data for the widget.

        Args:
            rv_name (str): Bean Machine string representation of an `RVIdentifier`.
            bw_fct (float): Bandwidth multiplicative factor.
            hdi (float): Highest Density Interval.

        Returns:
            Dict[str, Dict[str, List[Union[int, float]]]]: Data used for the widget.
        """
        rv_identifier = self.rv_identifiers[self.rv_names.index(rv_name)]
        output = {"rank": {}, "trace": {}}
        rank_output = {}
        trace_output = {}
        marginal_output = {}
        forest_output = {}
        trace_data = self.idata["posterior"][rv_identifier].values
        rank_data = az.plots.plot_utils.compute_ranks(trace_data)
        n_bins = int(np.ceil(2 * np.log2(rank_data.shape[1])) + 1)
        bins = np.histogram_bin_edges(rank_data, bins=n_bins, range=(0, rank_data.size))
        rank_output["bins"] = bins.tolist()
        for chain in range(self.num_chains):
            # Rank data
            _, h, _ = az.stats.density_utils.histogram(rank_data[chain, :], bins=n_bins)
            rank_output[f"chain{chain}_hist"] = h.tolist()
            normed_hist = h / h.max()
            rank_mean = float(normed_hist.mean())
            N = len(bins)
            rank_output[f"chain{chain}_normed_hist"] = normed_hist.tolist()
            rank_output[f"chain{chain}_left"] = bins[:-1].tolist()
            rank_output[f"chain{chain}_top"] = (normed_hist + chain).tolist()
            rank_output[f"chain{chain}_right"] = bins[1:].tolist()
            rank_output[f"chain{chain}_bottom"] = (np.zeros(N - 1) + chain).tolist()
            rank_output[f"chain{chain}_bin_label"] = [
                f"{int(b[0]):0,}–{int(b[1]):0,}" for b in zip(bins[:-1], bins[1:])
            ]
            rank_output[f"chain{chain}_chain_label"] = [chain + 1] * (N - 1)
            rank_output[f"chain{chain}_rank_label"] = normed_hist.tolist()
            x = np.linspace(start=bins[0], stop=bins[-1], num=N - 1)
            rank_output[f"chain{chain}_mean_x"] = x.tolist()
            y = chain + (np.ones(N - 1) * normed_hist.mean())
            rank_output[f"chain{chain}_mean_y"] = y.tolist()
            rank_output[f"chain{chain}_mean_chain_label"] = [chain + 1] * len(x)
            rank_output[f"chain{chain}_mean_label"] = [rank_mean] * len(x)

            # Trace data
            y = trace_data[chain, :]
            trace_output[f"chain{chain}_x"] = np.arange(len(y)).tolist()
            trace_output[f"chain{chain}_y"] = y.tolist()
            trace_output[f"chain{chain}_mean_label"] = [float(y.mean())] * len(y)
            trace_output[f"chain{chain}_chain_label"] = [chain + 1] * len(y)

            # Marginal data
            pdf = az.stats.density_utils._kde_linear(y, bw_return=True, bw_fct=bw_fct)
            pdf_x = pdf[0]
            pdf_y = pdf[1] / pdf[1].max()
            marginal_mean = float(pdf_x.mean())
            if rv_name not in self.bw_cache:
                self.bw_cache[rv_name] = pdf[2]
            else:
                self.bw_cache[rv_name] = pdf[2]
            marginal_output[f"chain{chain}_x"] = pdf_x.tolist()
            marginal_output[f"chain{chain}_y"] = pdf_y.tolist()
            marginal_output[f"chain{chain}_chain_label"] = [chain + 1] * len(pdf_y)
            marginal_output[f"chain{chain}_mean_label"] = [marginal_mean] * len(pdf_y)

            # Forest data
            hdi_x = list(az.stats.hdi(y, hdi_prob=hdi))
            hdi_y = 2 * [chain + 1]
            chain_mean_x = float(y.mean())
            chain_mean_y = hdi_y[0]
            forest_output[f"chain{chain}_x"] = hdi_x
            forest_output[f"chain{chain}_y"] = hdi_y
            forest_output[f"chain{chain}_chain_mean_x"] = [chain_mean_x]
            forest_output[f"chain{chain}_chain_mean_y"] = [chain_mean_y]
            forest_output[f"chain{chain}_chain_label"] = [chain + 1]
            forest_output[f"chain{chain}_mean_label"] = [chain_mean_x]

        output["rank"] = rank_output
        output["trace"] = trace_output
        output["marginal"] = marginal_output
        output["forest"] = forest_output
        return output

    def create_sources(
        self,
        rv_name: str,
        bw_fct: float = 1.0,
        hdi: float = az.rcParams["stats.hdi_prob"],
    ) -> Dict[str, Dict[str, Dict[str, ColumnDataSource]]]:
        """Create Bokeh `ColumnDataSource` objects.

        Args:
            rv_name (str): Bean Machine string representation of an `RVIdentifier`.
            bw_fct (float): Bandwidth multiplicative factor.
            hdi (float): Highest Density Interval.

        Returns:
            Dict[str, Dict[str, Dict[str, ColumnDataSource]]]: A dictionary containing
                Bokeh `ColumnDataSource` objects.
        """
        data = self.compute(rv_name=rv_name, bw_fct=bw_fct, hdi=hdi)

        output = {"rank": {}, "trace": {}, "marginal": {}, "forest": {}}
        rank_output = {}
        trace_output = {}
        marginal_output = {}
        forest_output = {}
        for chain in range(self.num_chains):
            key = f"chain{chain}"

            # Rank sources
            rank_histogram = ColumnDataSource(
                {
                    "left": data["rank"][f"{key}_left"],
                    "top": data["rank"][f"{key}_top"],
                    "right": data["rank"][f"{key}_right"],
                    "bottom": data["rank"][f"{key}_bottom"],
                    "bin_label": data["rank"][f"{key}_bin_label"],
                    "chain_label": data["rank"][f"{key}_chain_label"],
                    "rank_label": data["rank"][f"{key}_rank_label"],
                }
            )
            rank_mean = ColumnDataSource(
                {
                    "x": data["rank"][f"{key}_mean_x"],
                    "y": data["rank"][f"{key}_mean_y"],
                    "chain_label": data["rank"][f"{key}_mean_chain_label"],
                    "mean_label": data["rank"][f"{key}_mean_label"],
                }
            )
            rank_output[f"chain{chain}"] = {
                "rank": rank_histogram,
                "rank_mean": rank_mean,
            }

            # Trace sources
            trace = ColumnDataSource(
                {
                    "x": data["trace"][f"{key}_x"],
                    "y": data["trace"][f"{key}_y"],
                    "chain_label": data["trace"][f"{key}_chain_label"],
                    "mean_label": data["trace"][f"{key}_mean_label"],
                }
            )
            trace_output[f"chain{chain}"] = {"trace": trace}

            # Marginal sources
            marginal = ColumnDataSource(
                {
                    "x": data["marginal"][f"{key}_x"],
                    "y": data["marginal"][f"{key}_y"],
                    "chain_label": data["marginal"][f"{key}_chain_label"],
                    "mean_label": data["marginal"][f"{key}_mean_label"],
                }
            )
            marginal_output[f"chain{chain}"] = {"marginal": marginal}

            # Forest sources
            forest_line = ColumnDataSource(
                {
                    "x": data["forest"][f"{key}_x"],
                    "y": data["forest"][f"{key}_y"],
                }
            )
            forest_circle = ColumnDataSource(
                {
                    "x": data["forest"][f"{key}_chain_mean_x"],
                    "y": data["forest"][f"{key}_chain_mean_y"],
                    "chain_label": data["forest"][f"{key}_chain_label"],
                    "mean_label": data["forest"][f"{key}_mean_label"],
                }
            )
            forest_output[f"chain{chain}"] = {
                "forest_line": forest_line,
                "forest_circle": forest_circle,
            }

        output["rank"] = rank_output
        output["trace"] = trace_output
        output["marginal"] = marginal_output
        output["forest"] = forest_output
        return output

    def create_figures(self, rv_name: str) -> Dict[str, Figure]:
        """Create Bokeh `Figure` objects.

        Args:
            rv_name (str): Bean Machine string representation of an `RVIdentifier`.

        Returns:
            Dict[str, Figure]: Dictionary of Bokeh `Figure` objects.
        """
        plot_width = 800

        plot_height = 500

        # Rank figure
        rank_fig = figure(
            plot_width=plot_width,
            plot_height=plot_height,
            x_axis_label="Rank from all chains",
            y_axis_label="Chain",
            outline_line_color="black",
            title=f"{rv_name} rank histograms for all chains for all draws",
        )
        utils.style_figure(rank_fig)
        rank_ticker = list(range(1, self.num_chains + 1))
        rank_fig.yaxis.ticker = rank_ticker

        # Trace figure
        trace_fig = figure(
            plot_width=plot_width,
            plot_height=plot_height,
            x_axis_label="Draw from single chain",
            y_axis_label=rv_name,
            outline_line_color="black",
            title=f"{rv_name} trace plots for all chains for all draws",
        )
        utils.style_figure(trace_fig)

        # Marginal figure
        marginal_fig = figure(
            plot_width=500,
            plot_height=500,
            x_axis_label=rv_name,
            outline_line_color="black",
            title=f"{rv_name} chain marginals",
        )
        utils.style_figure(marginal_fig)
        marginal_fig.yaxis.visible = False

        # Forest figure
        forest_fig = figure(
            plot_width=500,
            plot_height=500,
            x_axis_label=rv_name,
            outline_line_color="black",
            title=f"{rv_name} chain marginal forest",
            x_range=marginal_fig.x_range,
        )
        utils.style_figure(forest_fig)
        forest_ticker = list(range(1, self.num_chains + 1))
        forest_fig.yaxis.ticker = forest_ticker
        output = {
            "rank": rank_fig,
            "trace": trace_fig,
            "marginal": marginal_fig,
            "forest": forest_fig,
        }
        return output

    def create_glyphs(self) -> Dict[str, Dict[str, Dict[str, Glyph]]]:
        """Create Bokeh `Glyph` objects.

        Returns:
            Dict[str, Dict[str, Dict[str, Glyph]]]: Dictionary of Bokeh `Glyph` objects.
        """
        line_width = 1

        line_dash = "solid"
        palette = utils.choose_palette(self.num_chains)
        output = {}
        rank_output = {}
        trace_output = {}
        marginal_output = {}
        forest_output = {}
        for chain in range(self.num_chains):
            color = palette[chain]

            key = f"chain{chain}"
            rank_output[key] = {}
            trace_output[key] = {}
            marginal_output[key] = {}
            forest_output[key] = {}

            rank_chain_output = {}
            trace_chain_output = {}
            marginal_chain_output = {}
            forest_chain_output = {}

            # Rank glyphs
            rank_glyph = Quad(
                left="left",
                top="top",
                right="right",
                bottom="bottom",
                fill_color=color,
                line_color="white",
                fill_alpha=0.6,
                name=f"{key}_rank",
            )
            rank_hover_glyph = Quad(
                left="left",
                top="top",
                right="right",
                bottom="bottom",
                fill_color=color,
                line_color="black",
                line_width=2,
                fill_alpha=1,
                name=f"{key}_rank_hover",
            )
            rank_muted_glyph = Quad(
                left="left",
                top="top",
                right="right",
                bottom="bottom",
                fill_color=color,
                line_color="white",
                fill_alpha=0.1,
                name=f"{key}_rank_muted",
            )
            rank_mean_glyph = Line(
                x="x",
                y="y",
                line_dash="dashed",
                line_color="black",
                line_width=2,
                line_alpha=0.5,
                name=f"{key}_rank_mean",
            )
            rank_mean_hover_glyph = Line(
                x="x",
                y="y",
                line_dash="solid",
                line_color="black",
                line_width=4,
                line_alpha=1,
                name=f"{key}_rank_mean_hover",
            )
            rank_mean_muted_glyph = Line(
                x="x",
                y="y",
                line_dash="dashed",
                line_color="black",
                line_width=2,
                line_alpha=0.1,
                name=f"{key}_rank_mean_muted",
            )
            rank_chain_output["rank"] = {
                "glyph": rank_glyph,
                "hover_glyph": rank_hover_glyph,
                "muted_glyph": rank_muted_glyph,
            }
            rank_chain_output["rank_mean"] = {
                "glyph": rank_mean_glyph,
                "hover_glyph": rank_mean_hover_glyph,
                "muted_glyph": rank_mean_muted_glyph,
            }
            rank_output[key] = rank_chain_output

            # Trace glyphs
            trace_glyph = Line(
                x="x",
                y="y",
                line_dash=line_dash,
                line_color=color,
                line_width=line_width,
                line_alpha=0.5,
                name=f"{key}_trace",
            )
            trace_hover_glyph = Line(
                x="x",
                y="y",
                line_dash=line_dash,
                line_color=color,
                line_width=line_width,
                line_alpha=0.5,
                name=f"{key}_trace_hover",
            )
            trace_muted_glyph = Line(
                x="x",
                y="y",
                line_dash=line_dash,
                line_color=color,
                line_width=line_width,
                line_alpha=0.1,
                name=f"{key}_trace_muted",
            )
            trace_chain_output["trace"] = {
                "glyph": trace_glyph,
                "hover_glyph": trace_hover_glyph,
                "muted_glyph": trace_muted_glyph,
            }
            trace_output[key] = trace_chain_output

            # Marginal glyphs
            marginal_glyph = Line(
                x="x",
                y="y",
                line_dash=line_dash,
                line_color=color,
                line_width=3,
                line_alpha=0.8,
                name=f"{key}_marginal",
            )
            marginal_hover_glyph = Line(
                x="x",
                y="y",
                line_dash=line_dash,
                line_color=color,
                line_width=3,
                line_alpha=1.0,
                name=f"{key}_marginal_hover",
            )
            marginal_muted_glyph = Line(
                x="x",
                y="y",
                line_dash=line_dash,
                line_color=color,
                line_width=1,
                line_alpha=0.1,
                name=f"{key}_marginal_muted",
            )
            marginal_chain_output["marginal"] = {
                "glyph": marginal_glyph,
                "hover_glyph": marginal_hover_glyph,
                "muted_glyph": marginal_muted_glyph,
            }
            marginal_output[key] = marginal_chain_output

            # Forest glyphs
            forest_line_glyph = Line(
                x="x",
                y="y",
                line_dash=line_dash,
                line_color=color,
                line_width=3,
                line_alpha=1,
                name=f"{key}_forest_line",
            )
            forest_line_hover_glyph = Line(
                x="x",
                y="y",
                line_dash=line_dash,
                line_color=color,
                line_width=3,
                line_alpha=1,
                name=f"{key}_forest_hover_line",
            )
            forest_line_muted_glyph = Line(
                x="x",
                y="y",
                line_dash=line_dash,
                line_color=color,
                line_width=3,
                line_alpha=0.1,
                name=f"{key}_forest_muted_line",
            )
            forest_circle_glyph = Circle(
                x="x",
                y="y",
                fill_color=color,
                line_color="white",
                fill_alpha=1,
                line_alpha=1,
                size=10,
                name=f"{key}_forest_circle",
            )
            forest_circle_hover_glyph = Circle(
                x="x",
                y="y",
                fill_color="orange",
                line_color="black",
                fill_alpha=1,
                line_alpha=1,
                size=10,
                name=f"{key}_forest_circle_hover",
            )
            forest_circle_muted_glyph = Circle(
                x="x",
                y="y",
                fill_color=color,
                line_color="white",
                fill_alpha=0.1,
                line_alpha=0.1,
                size=10,
                name=f"{key}_forest_circle_muted",
            )

            forest_chain_output["forest_line"] = {
                "glyph": forest_line_glyph,
                "hover_glyph": forest_line_hover_glyph,
                "muted_glyph": forest_line_muted_glyph,
            }
            forest_chain_output["forest_circle"] = {
                "glyph": forest_circle_glyph,
                "hover_glyph": forest_circle_hover_glyph,
                "muted_glyph": forest_circle_muted_glyph,
            }
            forest_output[key] = forest_chain_output

        output["rank"] = rank_output
        output["trace"] = trace_output
        output["marginal"] = marginal_output
        output["forest"] = forest_output
        return output

    def create_annotations(
        self,
        figs: Dict[str, Figure],
        rv_name: str,
    ) -> Dict[str, Annotation]:
        """Create Bokeh `Annotation` objects.

        Args:
            figs (Dict[str, Figure]): Dictionary of Bokeh `Figure` objects.
            rv_name (str): Bean Machine string representation of an `RVIdentifier`.

        Returns:
            Dict[str, Annotation]: Dictionary of Bokeh `Annotation` objects.
        """
        rv_identifier = self.rv_identifiers[self.rv_names.index(rv_name)]

        forest_mean = float(
            self.idata["posterior"][rv_identifier].values.flatten().mean()
        )
        forest_span = Span(
            location=forest_mean,
            dimension="height",
            line_color="grey",
            line_dash="dashed",
            line_width=3,
            level="underlay",
            line_alpha=0.4,
        )
        legend_items = [
            LegendItem(
                label=f"Chain {chain + 1}",
                renderers=[
                    renderer
                    for renderer in figs["rank"].renderers
                    if renderer.name == f"chain{chain}_rank"
                ]
                + [
                    renderer
                    for renderer in figs["rank"].renderers
                    if renderer.name == f"chain{chain}_rank_mean"
                ]
                + [
                    renderer
                    for renderer in figs["trace"].renderers
                    if renderer.name == f"chain{chain}_trace"
                ]
                + [
                    renderer
                    for renderer in figs["marginal"].renderers
                    if renderer.name == f"chain{chain}_marginal"
                ]
                + [
                    renderer
                    for renderer in figs["forest"].renderers
                    if renderer.name
                    in [
                        f"chain{chain}_forest_line",
                        f"chain{chain}_forest_circle",
                    ]
                ],
            )
            for chain in range(self.num_chains)
        ]
        legend = Legend(
            items=legend_items,
            orientation="horizontal",
            border_line_color="black",
            click_policy="hide",
        )
        return {"legend": legend, "forest_span": forest_span}

    def add_tools(self, figs: Dict[str, Figure]) -> None:
        """Add Bokeh `HoverTool` objects to the figures.

        Args:
            figs (Dict[str, Figure]): Dictionary of Bokeh `Figure` objects.

        Returns:
            None: Directly adds tools to the given figures.
        """
        # Rank tools
        rank_bin_tips = HoverTool(
            renderers=[
                renderer
                for renderer in figs["rank"].renderers
                if renderer.name.endswith("rank")
            ],
            tooltips=[
                ("Chain", "@chain_label"),
                ("Draws", "@bin_label"),
                ("Rank", "@rank_label"),
            ],
        )
        figs["rank"].add_tools(rank_bin_tips)
        rank_mean_tips = HoverTool(
            renderers=[
                renderer
                for renderer in figs["rank"].renderers
                if renderer.name.endswith("rank_mean")
            ],
            tooltips=[("Chain", "@chain_label"), ("Mean", "@mean_label")],
        )
        figs["rank"].add_tools(rank_mean_tips)

        # Trace tools
        rv_name = figs["trace"].title.text.split()[0]
        trace_tips = HoverTool(
            renderers=[
                renderer
                for renderer in figs["trace"].renderers
                if renderer.name.endswith("trace")
            ],
            tooltips=[
                ("Chain", "@chain_label"),
                ("Mean", "@mean_label"),
                (f"{rv_name}", "@y"),
            ],
        )
        figs["trace"].add_tools(trace_tips)

        # marginal tools
        marginal_tips = HoverTool(
            renderers=[
                renderer
                for renderer in figs["marginal"].renderers
                if renderer.name.endswith("marginal")
            ],
            tooltips=[
                ("Chain", "@chain_label"),
                ("Mean", "@mean_label"),
                (f"{rv_name}", "@x"),
            ],
        )
        figs["marginal"].add_tools(marginal_tips)

        # Forest tools
        forest_tips = HoverTool(
            renderers=[
                renderer
                for renderer in figs["forest"].renderers
                if renderer.name.endswith("forest_circle")
            ],
            tooltips=[
                ("Chain", "@chain_label"),
                ("Mean", "@mean_label"),
            ],
        )
        figs["forest"].add_tools(forest_tips)

    def update_figure(
        self,
        rv_name: str,
        old_sources: Dict[str, Dict[str, Dict[str, ColumnDataSource]]],
        figs: Dict[str, Figure],
        bw_fct: float,
        hdi: float,
    ) -> None:
        """Update the figures in the widget.

        Args:
            rv_name (str): Bean Machine string representation of an `RVIdentifier`.
            old_sources (Dict[str, Dict[str, Dict[str, ColumnDataSource]]]): old_sources
            figs (Dict[str, Figure]): Dictionary of Bokeh `Figure` objects.
            bw_fct (float): Bandwidth multiplicative factor.
            hdi (float): Highest Density Interval.

        Returns:
            None: Directly updates the given figures with new data.
        """
        new_sources = self.create_sources(rv_name, bw_fct=bw_fct, hdi=hdi)

        fig_key = ""
        fig = figure()
        for fig_key, chain_dict in new_sources.items():
            fig = figs[fig_key]
            fig.title.text = " ".join([rv_name] + fig.title.text.split()[1:])
            if fig_key == "trace":
                fig.yaxis.axis_label = rv_name
            if fig_key in ["marginal", "forest"]:
                fig.xaxis.axis_label = rv_name
            for chain_key, source_dict in chain_dict.items():
                for source_key, source in source_dict.items():
                    old_sources[fig_key][chain_key][source_key].data = dict(source.data)

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
        forest_x_range_starts = []
        forest_x_range_ends = []
        for key, chain_dict in glyphs.items():
            fig = figs[key]
            for chain_key, glyph_source_dict in chain_dict.items():
                for source_key, glyph_dict in glyph_source_dict.items():
                    source = sources[key][chain_key][source_key]
                    if source_key == "forest_line":
                        forest_x_range_starts.append(source.data["x"][0])
                        forest_x_range_ends.append(source.data["x"][1])
                    utils.add_glyph_to_figure(
                        fig=fig,
                        source=source,
                        glyph_dict=glyph_dict,
                    )

        # Create annotations for the figure.
        annotations = self.create_annotations(figs=figs, rv_name=rv_name)
        figs["rank"].add_layout(annotations["legend"], "below")
        figs["trace"].add_layout(annotations["legend"], "below")
        figs["forest"].add_layout(annotations["forest_span"])

        # Create tooltips for the figure.
        self.add_tools(figs=figs)

        # Widgets
        rv_select = Select(title="Query", value=rv_name, options=self.rv_names)
        bw_slider = Slider(
            start=0.01,
            end=2.0,
            value=1.0,
            step=0.01,
            title="Marginal bandwidth factor",
            width=500,
        )
        bw_div = Div(text=f"Bandwidth: {self.bw_cache[rv_name] * bw_slider.value}")
        hdi_slider = Slider(
            start=1,
            end=99,
            value=100 * self.hdi_prob,
            step=1,
            title="HDI interval",
            width=500,
        )

        def update_rv(attr, old, new):
            rv_identifier = self.rv_identifiers[self.rv_names.index(new)]
            forest_mean = float(
                self.idata["posterior"][rv_identifier].values.flatten().mean()
            )
            span = [
                item
                for item in figs["forest"]._property_values["center"]
                if isinstance(item, Span)
            ][0]
            span.location = forest_mean
            self.update_figure(
                rv_name=new,
                old_sources=sources,
                figs=figs,
                bw_fct=1.0,
                hdi=self.hdi_prob,
            )
            bw_slider.value = 1.0
            bw_div.text = f"Bandwidth: {self.bw_cache[new]}"
            hdi_slider.value = 100 * self.hdi_prob

        def update_bw(attr, old, new):
            rv_name = str(rv_select.value)
            self.update_figure(
                figs=figs,
                rv_name=rv_name,
                old_sources=sources,
                bw_fct=new,
                hdi=self.hdi_prob,
            )
            bw_div.text = f"Bandwidth: {self.bw_cache[rv_name] * bw_slider.value}"

        def update_hdi(attr, old, new):
            rv_name = str(rv_select.value)
            self.update_figure(
                figs=figs,
                rv_name=rv_name,
                old_sources=sources,
                bw_fct=float(bw_slider.value),
                hdi=new / 100,
            )

        rv_select.on_change("value", update_rv)
        bw_slider.on_change("value", update_bw)
        hdi_slider.on_change("value", update_hdi)
        # NOTE: We are using Bokeh's CustomJS model in order to reset the ranges of the
        #       figures.
        rv_select.js_on_change(
            "value",
            CustomJS(
                args={
                    "rank_fig": figs["rank"],
                    "trace_fig": figs["trace"],
                    "forest_fig": figs["forest"],
                },
                code=(
                    "rank_fig.reset.emit();\n"
                    "trace_fig.reset.emit();\n"
                    "forest_fig.reset.emit()\n"
                ),
            ),
        )

        rank_tab = Panel(child=figs["rank"], title="Rank")
        trace_tab = Panel(child=figs["trace"], title="Trace")
        rank_trace_tabs = Tabs(tabs=[rank_tab, trace_tab])
        marginal_tab = Panel(
            child=column(figs["marginal"], bw_slider, bw_div),
            title="Marginal",
        )
        forest_tab = Panel(child=column(figs["forest"], hdi_slider), title="Forest")
        marginal_forest_tabs = Tabs(tabs=[marginal_tab, forest_tab])
        layout = column(
            rv_select,
            row(marginal_forest_tabs, rank_trace_tabs),
        )

        doc.add_root(layout)

    def show_widget(self) -> None:
        """Display the widget."""
        show(self.modify_doc)
