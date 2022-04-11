# flake8: noqa
"""Effective sample size widget."""
from typing import Dict, List, Union

import arviz as az
import numpy as np
from bokeh.layouts import column
from bokeh.models import Annotation, Glyph
from bokeh.models.annotations import Legend, LegendItem
from bokeh.models.callbacks import CustomJS
from bokeh.models.glyphs import Circle, Line
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.markups import Div
from bokeh.models.widgets.panels import Panel, Tabs
from bokeh.plotting import show
from bokeh.plotting.figure import figure, Figure

from . import utils


class EffectiveSampleSizeWidget:
    def __init__(self, idata: az.InferenceData) -> None:
        """Effective sample size widget.

        Args:
            idata (az.InferenceData): ArviZ `InferenceData` object.
        """
        self.idata = idata

        self.rv_identifiers = list(self.idata["posterior"].data_vars)
        self.rv_names = [str(rv_identifier) for rv_identifier in self.rv_identifiers]
        self.num_chains = self.idata["posterior"].dims["chain"]
        self.num_draws_single_chain = self.idata["posterior"].dims["draw"]
        self.num_draws_all_chains = self.num_chains * self.num_draws_single_chain
        self.first_draw = self.idata["posterior"].draw.values[0]
        self.rule_of_thumb = 100 * self.num_chains

    def compute(
        self,
        rv_name: str,
        num_points: int = 20,
    ) -> Dict[str, List[Union[int, float]]]:
        """Compute data for the widget.

        Args:
            rv_name (str): Bean Machine string representation of an `RVIdentifier`.
            num_points (int): num_points

        Returns:
            Dict[str, List[Union[int, float]]]: Data used for the widget.
        """
        # Extract the correct data from the ArviZ InferenceData object.
        rv_identifier = self.rv_identifiers[self.rv_names.index(rv_name)]

        # Calculate the effective sample size evolution.
        ess_x = np.linspace(
            start=self.num_draws_all_chains / num_points,
            stop=self.num_draws_all_chains,
            num=num_points,
        )
        draw_divisions = np.linspace(
            start=self.num_draws_single_chain // num_points,
            stop=self.num_draws_single_chain,
            num=num_points,
            dtype=np.integer,
        )
        data = self.idata["posterior"].data_vars[rv_identifier].values

        # Generate data for the rule-of-thumb visual line.
        rot_x = np.linspace(start=0, stop=self.num_draws_all_chains, num=num_points)
        rot_y = self.rule_of_thumb * np.ones(num_points)
        rot_label = [self.rule_of_thumb] * len(ess_x)

        output = {
            "ess_x": ess_x.tolist(),
            "ess_y_bulk": [
                az.stats.diagnostics._ess_bulk(data[:, self.first_draw : draw_div])
                for draw_div in draw_divisions
            ],
            "ess_y_tail": [
                az.stats.diagnostics._ess_tail(data[:, self.first_draw : draw_div])
                for draw_div in draw_divisions
            ],
            "rot_x": rot_x.tolist(),
            "rot_y": rot_y.tolist(),
            "rot_label": rot_label,
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

        output = {
            "ess_bulk": ColumnDataSource({"x": data["ess_x"], "y": data["ess_y_bulk"]}),
            "ess_tail": ColumnDataSource({"x": data["ess_x"], "y": data["ess_y_tail"]}),
            "rot": ColumnDataSource(
                {
                    "x": data["rot_x"],
                    "y": data["rot_y"],
                    "label": data["rot_label"],
                }
            ),
        }
        return output

    def create_figure(self, rv_name: str) -> Figure:
        """Create Bokeh `Figure` objects.

        Args:
            rv_name (str): Bean Machine string representation of an `RVIdentifier`.

        Returns:
            Figure: Bokeh `Figure` object.
        """
        fig = figure(
            plot_width=700,
            plot_height=500,
            outline_line_color="black",
            title=f"{rv_name} effective sample size",
            x_axis_label="Total number of draws",
            y_axis_label="ESS",
        )
        utils.style_figure(fig)
        return fig

    def create_glyphs(self) -> Dict[str, Dict[str, Glyph]]:
        """Create Bokeh `Glyph` objects.

        Returns:
            Dict[str, Dict[str, Glyph]]: Dictionary of Bokeh `Glyph` objects.
        """
        ess_line_dash = "solid"

        rot_line_dash = "dashed"

        ess_line_width = 2
        rot_line_width = 4
        ess_circle_size = 10

        palette = utils.choose_palette(n=4)
        ess_bulk_color = palette[0]
        ess_tail_color = palette[1]
        rot_color = palette[-1]

        non_muted_alpha = 0.6
        muted_alpha = 0.1

        # Bulk
        ess_bulk_line_glyph = Line(
            x="x",
            y="y",
            line_dash=ess_line_dash,
            line_color=ess_bulk_color,
            line_width=ess_line_width,
            line_alpha=non_muted_alpha,
            name="ess_bulk_line",
        )
        ess_bulk_circle_glyph = Circle(
            x="x",
            y="y",
            size=ess_circle_size,
            fill_color=ess_bulk_color,
            fill_alpha=1,
            line_color="white",
            line_alpha=1,
            line_width=1,
            name="ess_bulk_circle",
        )
        ess_bulk_circle_hover_glyph = Circle(
            x="x",
            y="y",
            size=ess_circle_size,
            fill_color=ess_bulk_color,
            fill_alpha=1,
            line_color="black",
            line_alpha=1,
            line_width=2,
            name="ess_bulk_circle_hover",
        )
        ess_bulk_circle_muted_glyph = Circle(
            x="x",
            y="y",
            size=ess_circle_size,
            fill_color=ess_bulk_color,
            fill_alpha=1,
            line_color="white",
            line_alpha=1,
            line_width=1,
            name="ess_bulk_circle_muted",
        )

        # Tail
        ess_tail_line_glyph = Line(
            x="x",
            y="y",
            line_dash=ess_line_dash,
            line_color=ess_tail_color,
            line_width=ess_line_width,
            line_alpha=non_muted_alpha,
            name="ess_tail_line",
        )
        ess_tail_circle_glyph = Circle(
            x="x",
            y="y",
            size=ess_circle_size,
            fill_color=ess_tail_color,
            fill_alpha=1,
            line_color="white",
            line_alpha=1,
            line_width=1,
            name="ess_tail_circle",
        )
        ess_tail_circle_hover_glyph = Circle(
            x="x",
            y="y",
            size=ess_circle_size,
            fill_color=ess_tail_color,
            fill_alpha=1,
            line_color="black",
            line_alpha=1,
            line_width=2,
            name="ess_tail_circle_hover",
        )
        ess_tail_circle_muted_glyph = Circle(
            x="x",
            y="y",
            size=ess_circle_size,
            fill_color=ess_tail_color,
            fill_alpha=1,
            line_color="white",
            line_alpha=1,
            line_width=1,
            name="ess_tail_circle_muted",
        )

        # Rule of thumb
        rot_line_glyph = Line(
            x="x",
            y="y",
            line_dash=rot_line_dash,
            line_color=rot_color,
            line_width=rot_line_width,
            line_alpha=non_muted_alpha,
            name="rot_line",
        )
        rot_line_hover_glyph = Line(
            x="x",
            y="y",
            line_dash="solid",
            line_color=rot_color,
            line_width=rot_line_width,
            line_alpha=1.0,
            name="rot_line_hover",
        )
        rot_line_muted_glyph = Line(
            x="x",
            y="y",
            line_dash=rot_line_dash,
            line_color=rot_color,
            line_width=rot_line_width,
            line_alpha=muted_alpha,
            name="rot_line_muted",
        )

        output = {
            "ess_bulk_line": {
                "glyph": ess_bulk_line_glyph,
                "hover_glyph": None,
                "muted_glyph": None,
            },
            "ess_bulk_circle": {
                "glyph": ess_bulk_circle_glyph,
                "hover_glyph": ess_bulk_circle_hover_glyph,
                "muted_glyph": ess_bulk_circle_muted_glyph,
            },
            "ess_tail_line": {
                "glyph": ess_tail_line_glyph,
                "hover_glyph": None,
                "muted_glyph": None,
            },
            "ess_tail_circle": {
                "glyph": ess_tail_circle_glyph,
                "hover_glyph": ess_tail_circle_hover_glyph,
                "muted_glyph": ess_tail_circle_muted_glyph,
            },
            "rot": {
                "glyph": rot_line_glyph,
                "hover_glyph": rot_line_hover_glyph,
                "muted_glyph": rot_line_muted_glyph,
            },
        }
        return output

    def create_annotations(self, fig: Figure) -> Annotation:
        """Create Bokeh `Annotation` objects.

        Args:
            fig (Figure): Bokeh `Figure` object.

        Returns:
            Annotation: Bokeh `Annotation` object.
        """
        ess_bulk_legend_item = LegendItem(
            label="Bulk",
            renderers=[
                renderer
                for renderer in fig.renderers
                if renderer.name in ["ess_bulk_line", "ess_bulk_circle"]
            ],
        )
        ess_tail_legend_item = LegendItem(
            label="Tail",
            renderers=[
                renderer
                for renderer in fig.renderers
                if renderer.name in ["ess_tail_line", "ess_tail_circle"]
            ],
        )
        rot_legend_item = LegendItem(
            label="Rule-of-thumb",
            renderers=[
                renderer for renderer in fig.renderers if renderer.name == "rot_line"
            ],
        )
        legend = Legend(
            items=[ess_bulk_legend_item, ess_tail_legend_item, rot_legend_item],
            location="top_left",
            orientation="vertical",
            border_line_color="black",
            background_fill_alpha=0.5,
        )
        return legend

    def update_figure(
        self,
        fig: Figure,
        rv_name: str,
        old_sources: Dict[str, ColumnDataSource],
    ) -> None:
        """Update the figures in the widget.

        Args:
            fig (Figure): fig
            rv_name (str): Bean Machine string representation of an `RVIdentifier`.
            old_sources (Dict[str, ColumnDataSource]): old_sources

        Returns:
            None: Directly updates the given figures with new data.
        """
        new_sources = self.create_sources(rv_name=rv_name)

        for key, new_cds in new_sources.items():
            old_sources[key].data = dict(new_cds.data)
        fig.title.text = " ".join([rv_name] + fig.title.text.split()[1:])
        fig.xaxis.axis_label = rv_name

    def help_page(self):
        text = """
            <h2>
              Effective sample size diagnostic
            </h2>
            <p style="margin-bottom: 10px">
              MCMC samplers do not draw truly independent samples from the target
              distribution, which means that our samples are correlated. In an ideal
              situation all samples would be independent, but we do not have that
              luxury. We can, however, measure the number of <em>effectively
              independent</em> samples we draw, which is called the effective sample
              size. You can read more about how this value is calculated in the Vehtari
              <em>et al</em> paper. In brief, it is a measure that combines information
              from the \(\hat{R}\) value with the autocorrelation estimates within the
              chains.
            </p>
            <p style="margin-bottom: 10px">
              ESS estimates come in two variants, <em>ess_bulk</em> and
              <em>ess_tail</em>. The former is the default, but the latter can be useful
              if you need good estimates of the tails of your posterior distribution.
              The rule of thumb for <em>ess_bulk</em> is for this value to be greater
              than 100 per chain on average. The <em>ess_tail</em> is an estimate for
              effectively independent samples considering the more extreme values of the
              posterior. This is not the number of samples that landed in the tails of
              the posterior, but rather a measure of the number of effectively
              independent samples if we sampled the tails of the posterior. The rule of
              thumb for this value is also to be greater than 100 per chain on average.
            </p>
            <p style="margin-bottom: 10px">
              When the model is converging properly, both the bulk and tail lines should
              be <em>roughly</em> linear.
            </p>
            <ul>
              <li>
                Vehtari A, Gelman A, Simpson D, Carpenter B, Bürkner PC (2021)
                <b>
                  Rank-normalization, folding, and localization: An improved \(\hat{R}\)
                  for assessing convergence of MCMC (with discussion)
                </b>.
                <em>Bayesian Analysis</em> 16(2)
                667–718.
                <a href=https://dx.doi.org/10.1214/20-BA1221 style="color: blue">
                  doi: 10.1214/20-BA1221
                </a>.
              </li>
            </ul>
        """
        div = Div(text=text, disable_math=False, min_width=800)
        return div

    def modify_doc(self, doc) -> None:
        """Modify the document by adding the widget."""
        # Set the initial view.
        rv_name = self.rv_names[0]

        # Create data sources for the figure.
        sources = self.create_sources(rv_name)

        # Create the figure.
        fig = self.create_figure(rv_name)

        # Create glyphs and add them to the figure.
        glyphs = self.create_glyphs()
        for key, glyph_dict in glyphs.items():
            source_key = ""
            if key.startswith("ess_bulk"):
                source_key = "ess_bulk"
            elif key.startswith("ess_tail"):
                source_key = "ess_tail"
            elif key == "rot":
                source_key = "rot"
            source = sources[source_key]
            utils.add_glyph_to_figure(fig=fig, source=source, glyph_dict=glyph_dict)

        # Create annotations for the figure and add them.
        annotations = self.create_annotations(fig=fig)
        fig.add_layout(annotations)

        # Create tooltips for the figure.
        bulk_tips = HoverTool(
            renderers=[
                renderer
                for renderer in fig.renderers
                if renderer.name == "ess_bulk_circle"
            ],
            tooltips=[
                ("Bulk draws", "@x{0,}"),  # noqa FS003 f-string
                ("ESS", "@y{0,}"),  # noqa FS003 flake8 f-string
            ],
        )
        fig.add_tools(bulk_tips)
        tail_tips = HoverTool(
            renderers=[
                renderer
                for renderer in fig.renderers
                if renderer.name == "ess_tail_circle"
            ],
            tooltips=[
                ("Tail draws", "@x{0,}"),  # noqa FS003 f-string
                ("ESS", "@y{0,}"),  # noqa FS003 f-string
            ],
        )
        fig.add_tools(tail_tips)
        rot_tips = HoverTool(
            renderers=[
                renderer for renderer in fig.renderers if renderer.name == "rot_line"
            ],
            tooltips=[("Rule of thumb", "@label")],
        )
        fig.add_tools(rot_tips)

        # Widgets
        rv_select = Select(title="Query", value=rv_name, options=self.rv_names)

        def update_rv(attr, old, new):
            self.update_figure(fig=fig, rv_name=new, old_sources=sources)

        rv_select.on_change("value", update_rv)
        # NOTE: We are using Bokeh's CustomJS model in order to reset the ranges of the
        #       figures.
        rv_select.js_on_change(
            "value",
            CustomJS(args={"p": fig}, code="p.reset.emit()"),
        )

        widget_tab = Panel(child=column(rv_select, fig), title="ESS")
        help_tab = Panel(child=self.help_page(), title="Help")
        tabs = Tabs(tabs=[widget_tab, help_tab])
        layout = column(tabs)
        doc.add_root(layout)

    def show_widget(self):
        """Display the widget."""
        show(self.modify_doc)
