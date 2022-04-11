"""Posterior plot widget."""
from typing import Dict, List

import arviz as az
import numpy as np
from bokeh.layouts import column, row
from bokeh.models import Annotation, Glyph
from bokeh.models.annotations import Band, LabelSet
from bokeh.models.callbacks import CustomJS
from bokeh.models.glyphs import Circle, Line
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.markups import Div
from bokeh.models.widgets.panels import Panel, Tabs
from bokeh.models.widgets.sliders import Slider
from bokeh.plotting import show
from bokeh.plotting.figure import figure, Figure
from scipy import interpolate

from . import utils


class PosteriorWidget:

    bw_cache = {}

    def __init__(self, idata: az.InferenceData) -> None:
        """Posterior widget.

        Args:
            idata (az.InferenceData): ArviZ `InferenceData` object.
        """
        self.idata = idata
        self.rv_identifiers = list(self.idata["posterior"].data_vars)
        self.rv_names = [str(rv_identifier) for rv_identifier in self.rv_identifiers]
        self.hdi_prob = az.rcParams["stats.hdi_prob"]

    def compute(self, rv_name: str, hdi: float, **kwargs) -> Dict[str, List[float]]:
        """Compute data for the widget.

        Args:
            rv_name (str): Bean Machine string representation of an `RVIdentifier`.
            hdi (float): Highest Density Interval. Can be (0, 1).
            kwargs: See the ArviZ documentation for `_kde_linear`.

        Returns:
            Dict[str, List[float]]: Data used for the widget.
        """
        # Extract the correct data from the ArviZ InferenceData object.
        rv_identifier = self.rv_identifiers[self.rv_names.index(rv_name)]
        data = self.idata["posterior"][rv_identifier].values.flatten()

        # Calculate the density for the RV.
        pdf = az.stats.density_utils._kde_linear(data, bw_return=True, **kwargs)
        pdf_x = pdf[0]
        pdf_y = pdf[1] / pdf[1].max()
        if rv_name not in self.bw_cache:
            self.bw_cache[rv_name] = pdf[2]
        else:
            self.bw_cache[rv_name] = pdf[2]
        kde = interpolate.interp1d(pdf_x, pdf_y)
        pdf_mean_x = float(pdf_x.mean())
        pdf_mean_y = float(kde(pdf_mean_x))

        # Calculate the cumulative distribution for the RV.
        cdf = az.stats.density_utils._kde_linear(data, cumulative=True, **kwargs)
        cdf_x = cdf[0]
        cdf_y = cdf[1] / cdf[1].max()
        cdf_intrp = interpolate.interp1d(cdf_x, cdf_y)
        cdf_mean_y = cdf_intrp(pdf_mean_x)

        # Calculate the HDI interval and where the values exist on the PDF.
        hdi_interval = list(az.stats.hdi(data, hdi_prob=hdi))
        lower_bound_hdi_pdf_x = lower_bound_hdi_cdf_x = hdi_interval[0]
        upper_bound_hdi_pdf_x = upper_bound_hdi_cdf_x = hdi_interval[1]
        lower_bound_hdi_pdf_y = float(kde(lower_bound_hdi_pdf_x))
        upper_bound_hdi_pdf_y = float(kde(upper_bound_hdi_pdf_x))
        lower_bound_hdi_cdf_y = float(cdf_intrp(lower_bound_hdi_cdf_x))
        upper_bound_hdi_cdf_y = float(cdf_intrp(upper_bound_hdi_cdf_x))
        pdf_mask = np.ix_(
            np.logical_and(
                pdf_x >= lower_bound_hdi_pdf_x, pdf_x <= upper_bound_hdi_pdf_x
            )
        )[0]
        hdi_pdf_x = pdf_x[pdf_mask]
        hdi_lower_pdf_y = np.zeros(len(hdi_pdf_x))
        hdi_upper_pdf_y = pdf_y[pdf_mask]
        cdf_mask = np.ix_(
            np.logical_and(
                cdf_x >= lower_bound_hdi_pdf_x, cdf_x <= upper_bound_hdi_pdf_x
            )
        )[0]
        hdi_cdf_x = cdf_x[cdf_mask]
        hdi_lower_cdf_y = np.zeros(len(hdi_cdf_x))
        hdi_upper_cdf_y = cdf_y[cdf_mask]

        output = {
            "pdf_x": pdf_x.tolist(),
            "pdf_y": pdf_y.tolist(),
            "cdf_x": cdf_x.tolist(),
            "cdf_y": cdf_y.tolist(),
            "hdi_pdf_x": list(hdi_pdf_x),
            "hdi_lower_pdf_y": hdi_lower_pdf_y.tolist(),
            "hdi_upper_pdf_y": list(hdi_upper_pdf_y),
            "hdi_cdf_x": list(hdi_cdf_x),
            "hdi_lower_cdf_y": hdi_lower_cdf_y.tolist(),
            "hdi_upper_cdf_y": list(hdi_upper_cdf_y),
            "point_stats_pdf_x": [
                lower_bound_hdi_pdf_x,
                pdf_mean_x,
                upper_bound_hdi_pdf_x,
            ],
            "point_stats_pdf_y": [
                lower_bound_hdi_pdf_y,
                pdf_mean_y,
                upper_bound_hdi_pdf_y,
            ],
            "point_stats_cdf_x": [
                lower_bound_hdi_cdf_x,
                pdf_mean_x,
                upper_bound_hdi_cdf_x,
            ],
            "point_stats_cdf_y": [
                lower_bound_hdi_cdf_y,
                cdf_mean_y,
                upper_bound_hdi_cdf_y,
            ],
        }
        return output

    def create_sources(
        self,
        rv_name: str,
        bw_fct: float = 1.0,
        hdi: float = 0.89,
    ) -> Dict[str, ColumnDataSource]:
        """Create Bokeh `ColumnDataSource` objects.

        Args:
            rv_name (str): Bean Machine string representation of an `RVIdentifier`.
            bw_fct (float): Multiplicative bandwidth factor.
            hdi (float): Highest Density Interval. Can be (0, 1).

        Returns:
            Dict[str, ColumnDataSource]: A dictionary containing Bokeh
                `ColumnDataSource` objects.
        """
        data = self.compute(rv_name=rv_name, bw_fct=bw_fct, hdi=hdi)
        output = {
            "posterior": ColumnDataSource({"x": data["pdf_x"], "y": data["pdf_y"]}),
            "cdf": ColumnDataSource({"x": data["cdf_x"], "y": data["cdf_y"]}),
            "hdi_pdf": ColumnDataSource(
                {
                    "base": data["hdi_pdf_x"],
                    "lower": data["hdi_lower_pdf_y"],
                    "upper": data["hdi_upper_pdf_y"],
                }
            ),
            "hdi_cdf": ColumnDataSource(
                {
                    "base": data["hdi_cdf_x"],
                    "lower": data["hdi_lower_cdf_y"],
                    "upper": data["hdi_upper_cdf_y"],
                }
            ),
            "point_stats_pdf": ColumnDataSource(
                {
                    "x": data["point_stats_pdf_x"],
                    "y": data["point_stats_pdf_y"],
                    "label": ["Lower HDI", "Mean", "Upper HDI"],
                    "label_set": [
                        f"Lower HDI: {data['point_stats_pdf_x'][0]:.3f}",
                        f"Mean: {data['point_stats_pdf_x'][1]:.3f}",
                        f"Upper HDI: {data['point_stats_pdf_x'][2]:.3f}",
                    ],
                    "justification": ["right", "center", "left"],
                    "x_offset": [-5, 0, 5],
                    "y_offset": [0, 10, 0],
                }
            ),
            "point_stats_cdf": ColumnDataSource(
                {
                    "x": data["point_stats_cdf_x"],
                    "y": data["point_stats_cdf_y"],
                    "label": ["Lower HDI", "Mean", "Upper HDI"],
                    "label_set": [
                        f"Lower HDI: {data['point_stats_cdf_x'][0]:.3f}",
                        f"Mean: {data['point_stats_cdf_x'][1]:.3f}",
                        f"Upper HDI: {data['point_stats_cdf_x'][2]:.3f}",
                    ],
                    "justification": ["right", "center", "left"],
                    "x_offset": [-5, 0, 5],
                    "y_offset": [0, 10, 0],
                }
            ),
        }
        return output

    def create_pdf_figure(self, rv_name: str) -> Figure:
        """Create Bokeh `Figure` objects.

        Args:
            rv_name (str): Bean Machine string representation of an `RVIdentifier`.

        Returns:
            Figure: Bokeh `Figure` object.
        """
        fig = figure(
            plot_width=500,
            plot_height=500,
            x_axis_label=rv_name,
            outline_line_color="black",
            title=f"{rv_name} posterior",
        )
        utils.style_figure(fig)
        fig.yaxis.visible = False
        return fig

    def create_cdf_figure(self, rv_name: str) -> Figure:
        """Create Bokeh `Figure` objects.

        Args:
            rv_name (str): Bean Machine string representation of an `RVIdentifier`.

        Returns:
            Figure: Bokeh `Figure` object.
        """
        fig = figure(
            plot_width=500,
            plot_height=500,
            x_axis_label=rv_name,
            outline_line_color="black",
            title=f"{rv_name} CDF",
        )
        utils.style_figure(fig)
        fig.yaxis.visible = False
        return fig

    def create_glyphs(self) -> Dict[str, Dict[str, Glyph]]:
        """Create Bokeh `Glyph` objects.

        Returns:
            Dict[str, Dict[str, Glyph]]: Dictionary of Bokeh `Glyph` objects.
        """
        line_dash = "solid"
        line_width = 2
        palette = utils.choose_palette(n=1)
        color = palette[0]
        muted_alpha = 0.1

        # Posterior
        posterior_glyph = Line(
            x="x",
            y="y",
            line_dash=line_dash,
            line_color=color,
            line_width=line_width,
            line_alpha=0.7,
            name="posterior",
        )
        posterior_hover_glyph = Line(
            x="x",
            y="y",
            line_dash=line_dash,
            line_color="orange",
            line_width=line_width,
            line_alpha=1,
            name="posterior_hover",
        )
        posterior_muted_glyph = Line(
            x="x",
            y="y",
            line_dash=line_dash,
            line_color=color,
            line_width=line_width,
            line_alpha=muted_alpha,
            name="posterior_muted",
        )

        # CDF
        cdf_glyph = Line(
            x="x",
            y="y",
            line_dash=line_dash,
            line_color=color,
            line_width=line_width,
            line_alpha=0.7,
            name="cdf",
        )
        cdf_hover_glyph = Line(
            x="x",
            y="y",
            line_dash=line_dash,
            line_color="orange",
            line_width=line_width,
            line_alpha=1,
            name="cdf_hover",
        )
        cdf_muted_glyph = Line(
            x="x",
            y="y",
            line_dash=line_dash,
            line_color=color,
            line_width=line_width,
            line_alpha=muted_alpha,
            name="cdf_muted",
        )

        # Point statistics
        point_stats_pdf_glyph = Circle(
            x="x",
            y="y",
            fill_color=color,
            line_color="white",
            fill_alpha=1,
            line_alpha=1,
            size=10,
            name="point_stats_pdf",
        )
        point_stats_pdf_hover_glyph = Circle(
            x="x",
            y="y",
            fill_color="orange",
            line_color="black",
            fill_alpha=1,
            line_alpha=1,
            size=10,
            name="point_stats_pdf_hover",
        )
        point_stats_pdf_muted_glyph = Circle(
            x="x",
            y="y",
            fill_color=color,
            line_color="white",
            fill_alpha=muted_alpha,
            line_alpha=muted_alpha,
            size=10,
            name="point_stats_pdf_muted",
        )
        point_stats_cdf_glyph = Circle(
            x="x",
            y="y",
            fill_color=color,
            line_color="white",
            fill_alpha=1,
            line_alpha=1,
            size=10,
            name="point_stats_cdf",
        )
        point_stats_cdf_hover_glyph = Circle(
            x="x",
            y="y",
            fill_color="orange",
            line_color="black",
            fill_alpha=1,
            line_alpha=1,
            size=10,
            name="point_stats_cdf_hover",
        )
        point_stats_cdf_muted_glyph = Circle(
            x="x",
            y="y",
            fill_color=color,
            line_color="white",
            fill_alpha=muted_alpha,
            line_alpha=muted_alpha,
            size=10,
            name="point_stats_cdf_muted",
        )

        output = {
            "posterior": {
                "glyph": posterior_glyph,
                "hover_glyph": posterior_hover_glyph,
                "muted_glyph": posterior_muted_glyph,
            },
            "cdf": {
                "glyph": cdf_glyph,
                "hover_glyph": cdf_hover_glyph,
                "muted_glyph": cdf_muted_glyph,
            },
            "point_stats_pdf": {
                "glyph": point_stats_pdf_glyph,
                "hover_glyph": point_stats_pdf_hover_glyph,
                "muted_glyph": point_stats_pdf_muted_glyph,
            },
            "point_stats_cdf": {
                "glyph": point_stats_cdf_glyph,
                "hover_glyph": point_stats_cdf_hover_glyph,
                "muted_glyph": point_stats_cdf_muted_glyph,
            },
        }
        return output

    def create_annotations(
        self,
        sources: Dict[str, ColumnDataSource],
    ) -> Dict[str, Annotation]:
        """Create Bokeh `Annotation` objects.

        Args:
            sources (Dict[str, ColumnDataSource]): A dictionary containing Bokeh
                `ColumnDataSource` objects.

        Returns:
            Dict[str, Annotation]: Dictionary of Bokeh `Annotation` objects.
        """
        palette = utils.choose_palette(n=1)
        color = palette[0]

        # HDI colored region.
        hdi_pdf_cds = sources["hdi_pdf"]
        hdi_pdf = Band(
            base="base",
            lower="lower",
            upper="upper",
            source=hdi_pdf_cds,
            level="underlay",
            fill_color=color,
            fill_alpha=0.2,
            line_width=1,
            line_color="white",
            name="hdi_pdf",
        )
        hdi_cdf_cds = sources["hdi_cdf"]
        hdi_cdf = Band(
            base="base",
            lower="lower",
            upper="upper",
            source=hdi_cdf_cds,
            level="underlay",
            fill_color=color,
            fill_alpha=0.2,
            line_width=1,
            line_color="white",
            name="hdi_cdf",
        )

        # Point statistics labels.
        point_stats_pdf_cds = sources["point_stats_pdf"]
        point_stats_pdf_labels = LabelSet(
            x="x",
            y="y",
            text="label_set",
            text_align="justification",
            x_offset="x_offset",
            y_offset="y_offset",
            source=point_stats_pdf_cds,
            name="point_stats_pdf_labels",
        )
        point_stats_cdf_cds = sources["point_stats_cdf"]
        point_stats_cdf_labels = LabelSet(
            x="x",
            y="y",
            text="label_set",
            text_align="justification",
            x_offset="x_offset",
            y_offset="y_offset",
            source=point_stats_cdf_cds,
            name="point_stats_cdf_labels",
        )

        output = {
            "hdi_pdf": hdi_pdf,
            "hdi_cdf": hdi_cdf,
            "point_stats_pdf_labels": point_stats_pdf_labels,
            "point_stats_cdf_labels": point_stats_cdf_labels,
        }
        return output

    def update_figure(
        self,
        fig: Figure,
        rv_name: str,
        old_sources: Dict[str, ColumnDataSource],
        bw_fct: float,
        hdi: int,
    ) -> None:
        """Update the figures in the widget.

        Directly manipulates the Bokeh `ColumnDataSource` objects by updating the data
        they contain.

        Args:
            fig (Figure): Bokeh `Figure` object.
            rv_name (str): Bean Machine string representation of an `RVIdentifier`.
            old_sources (Dict[str, ColumnDataSource]): Old Bokeh `ColumnDataSource`
                objects before the widget interaction change.
            bw_fct (float): Multiplicative bandwidth factor.
            hdi (float): Highest Density Interval. Can be (0, 1).

        Returns:
            None: Directly updates the given figures with new data.
        """
        new_sources = self.create_sources(rv_name, bw_fct=bw_fct, hdi=hdi)
        for key, new_cds in new_sources.items():
            old_sources[key].data = dict(new_cds.data)
        fig.title.text = " ".join([rv_name] + fig.title.text.split()[1:])
        fig.xaxis.axis_label = rv_name

    def help_page(self):
        text = """
            <h2>
              Highest density interval
            </h2>
            <p style="margin-bottom: 10px">
              The highest density interval region is not equal tailed like a typical
              equal tailed interval of 2.5%. Thus it will include the mode(s) of the
              posterior distribution.
            </p>
            <p style="margin-bottom: 10px">
              There is nothing particularly specific about having a default HDI of 89%.
              If fact, the only remarkable thing about defaulting to 89% is that it is
              the highest prime number that does not exceed the unstable 95% threshold.
              See the link to McElreath's book below for further discussion.
            </p>
            <ul>
              <li>
                McElreath R (2020)
                <b>
                  Statistical Rethinking: A Bayesian Course with Examples in R and Stan
                  2nd edition.
                </b>
                <em>Chapman and Hall/CRC</em>
                <a href=https://dx.doi.org/10.1201/9780429029608 style="color: blue">
                  doi: 10.1201/9780429029608
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

        # Create data sources for the figures.
        sources = self.create_sources(rv_name)

        # Create the figures.
        pdf_fig = self.create_pdf_figure(rv_name)
        cdf_fig = self.create_cdf_figure(rv_name)
        cdf_fig.x_range = pdf_fig.x_range
        cdf_fig.y_range = pdf_fig.y_range

        # Create glyphs and add them to the figure.
        glyphs = self.create_glyphs()
        utils.add_glyph_to_figure(
            fig=pdf_fig,
            source=sources["posterior"],
            glyph_dict=glyphs["posterior"],
        )
        utils.add_glyph_to_figure(
            fig=pdf_fig,
            source=sources["point_stats_pdf"],
            glyph_dict=glyphs["point_stats_pdf"],
        )
        utils.add_glyph_to_figure(
            fig=cdf_fig,
            source=sources["cdf"],
            glyph_dict=glyphs["cdf"],
        )
        utils.add_glyph_to_figure(
            fig=cdf_fig,
            source=sources["point_stats_cdf"],
            glyph_dict=glyphs["point_stats_cdf"],
        )

        # Create annotations for the figure.
        annotations = self.create_annotations(sources=sources)
        utils.add_annotation_to_figure(
            fig=pdf_fig,
            annotation=annotations["point_stats_pdf_labels"],
        )
        utils.add_annotation_to_figure(
            fig=pdf_fig,
            annotation=annotations["hdi_pdf"],
        )
        utils.add_annotation_to_figure(
            fig=cdf_fig,
            annotation=annotations["point_stats_cdf_labels"],
        )
        utils.add_annotation_to_figure(
            fig=cdf_fig,
            annotation=annotations["hdi_cdf"],
        )

        # Create tooltips for the figure.
        posterior_tips = HoverTool(
            renderers=[
                renderer
                for renderer in pdf_fig.renderers
                if renderer.name == "posterior"
            ],
            tooltips=[("", "@x")],
        )
        pdf_fig.add_tools(posterior_tips)
        cdf_tips = HoverTool(
            renderers=[
                renderer for renderer in cdf_fig.renderers if renderer.name == "cdf"
            ],
            tooltips=[("", "@x")],
        )
        cdf_fig.add_tools(cdf_tips)
        point_stats_pdf_tips = HoverTool(
            renderers=[
                renderer
                for renderer in pdf_fig.renderers
                if renderer.name == "point_stats_pdf"
            ],
            tooltips=[("", "@label_set")],
        )
        pdf_fig.add_tools(point_stats_pdf_tips)
        point_stats_cdf_tips = HoverTool(
            renderers=[
                renderer
                for renderer in cdf_fig.renderers
                if renderer.name == "point_stats_cdf"
            ],
            tooltips=[("", "@label_set")],
        )
        cdf_fig.add_tools(point_stats_cdf_tips)

        # Widgets
        rv_select = Select(title="Query", value=rv_name, options=self.rv_names)
        bw_slider = Slider(
            start=0.01,
            end=2.0,
            value=1.0,
            step=0.01,
            title="Posterior bandwidth factor",
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

        def update_bw(attr, old, new):
            rv_name = pdf_fig.title.text.split()[0]
            self.update_figure(
                fig=pdf_fig,
                rv_name=rv_name,
                old_sources=sources,
                bw_fct=bw_slider.value,
                hdi=hdi_slider.value / 100,
            )
            bw_div.text = f"Bandwidth: {self.bw_cache[rv_name] * bw_slider.value}"

        def update_rv(attr, old, new):
            self.update_figure(
                fig=pdf_fig,
                rv_name=new,
                old_sources=sources,
                bw_fct=1.0,
                hdi=self.hdi_prob,
            )
            self.update_figure(
                fig=cdf_fig,
                rv_name=new,
                old_sources=sources,
                bw_fct=1.0,
                hdi=self.hdi_prob,
            )
            bw_slider.value = 1.0
            bw_div.text = f"Bandwidth: {self.bw_cache[new]}"
            hdi_slider.value = 100 * self.hdi_prob

        def update_hdi(attr, old, new):
            rv_name = pdf_fig.title.text.split()[0]
            self.update_figure(
                fig=pdf_fig,
                rv_name=rv_name,
                old_sources=sources,
                bw_fct=bw_slider.value,
                hdi=new / 100,
            )

        rv_select.on_change("value", update_rv)
        # NOTE: We are using Bokeh's CustomJS model in order to reset the ranges of the
        #       figures.
        rv_select.js_on_change(
            "value",
            CustomJS(args={"p": pdf_fig}, code="p.reset.emit()"),
        )
        bw_slider.on_change("value", update_bw)
        hdi_slider.on_change("value", update_hdi)

        widget_layout = column(
            rv_select,
            row(pdf_fig, cdf_fig),
            bw_slider,
            bw_div,
            hdi_slider,
        )
        widget_panel = Panel(child=widget_layout, title="Posterior")
        help_panel = Panel(child=self.help_page(), title="Help")
        tabs = Tabs(tabs=[widget_panel, help_panel])
        layout = column(tabs)

        doc.add_root(layout)

    def show_widget(self):
        """Display the widget."""
        show(self.modify_doc)
