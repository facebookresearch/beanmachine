"""Joint plot widget."""
from typing import Dict, List, Union

import arviz as az
from bokeh.layouts import column, row
from bokeh.models import Glyph
from bokeh.models.callbacks import CustomJS
from bokeh.models.glyphs import Circle, Image, Line
from bokeh.models.mappers import LinearColorMapper
from bokeh.models.sources import ColumnDataSource
from bokeh.models.widgets.groups import RadioButtonGroup
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.markups import Div
from bokeh.models.widgets.panels import Panel, Tabs
from bokeh.plotting import gridplot, show
from bokeh.plotting.figure import figure, Figure

from . import utils


class JointPlotWidget:
    def __init__(self, idata: az.InferenceData) -> None:
        """Joint plot widget.

        Args:
            idata (az.InferenceData): ArviZ `InferenceData` object.
        """
        self.idata = idata

        self.rv_identifiers = list(self.idata["posterior"].data_vars)
        self.rv_names = [str(rv_identifier) for rv_identifier in self.rv_identifiers]

    def compute(
        self,
        rv_names: List[str],
        **kwargs,
    ) -> Dict[str, Dict[str, Union[List[float], List[List[float]], float, str]]]:
        """Compute data for the widget.

        Args:
            rv_name (List[str]): Bean Machine string representation of an
                `RVIdentifier`.
            kwargs:

        Returns:
            Dict[str, Dict[str, Union[List[float], List[List[float]], float, str]]]:
                Data used for the widget.
        """
        rv_identifiers = []

        for rv_name in rv_names:
            rv_identifiers.append(self.rv_identifiers[self.rv_names.index(rv_name)])
        output = {
            "marginal_x": {"x": [], "y": [], "label": [], "data": []},
            "marginal_y": {"x": [], "y": [], "label": [], "data": []},
            "joint": {},
        }
        # 1D marginals
        marginal_data = []
        for i, rv_identifier in enumerate(rv_identifiers):
            key = ""
            x_key = ""
            y_key = ""
            if i == 0:
                key = "marginal_x"
                x_key = "x"
                y_key = "y"
            elif i == 1:
                key = "marginal_y"
                x_key = "y"
                y_key = "x"
            locals()[f"{key}_data"] = self.idata["posterior"][
                rv_identifier
            ].values.flatten()
            output[key]["data"] = locals()[f"{key}_data"].tolist()
            marginal_data.append(locals()[f"{key}_data"])
            marginal_pdf = az.stats.density_utils._kde_linear(
                locals()[f"{key}_data"],
                **kwargs,
            )
            marginal_pdf_x = marginal_pdf[0]
            marginal_pdf_y = marginal_pdf[1] / marginal_pdf[1].max()
            output[key][x_key] = marginal_pdf_x.tolist()
            output[key][y_key] = marginal_pdf_y.tolist()
            output[key]["label"] = [rv_names[i]] * len(output[key]["x"])
        # 2D marginal
        density, xmin, xmax, ymin, ymax = az.stats.density_utils._fast_kde_2d(
            x=marginal_data[0],
            y=marginal_data[1],
        )
        output["joint"] = {
            "image": [density.T.tolist()],
            "xmin": [xmin],
            "xmax": [xmax],
            "ymin": [ymin],
            "ymax": [ymax],
            "dw": [xmax - xmin],
            "dh": [ymax - ymin],
            "palette": ["Viridis256"],
            "label": ["/".join(rv_names)],
        }
        return output

    def create_sources(self, rv_names: List[str]) -> Dict[str, ColumnDataSource]:
        """Create Bokeh `ColumnDataSource` objects.

        Args:
            rv_name (List[str]): Bean Machine string representation of an
                `RVIdentifier`.

        Returns:
            Dict[str, ColumnDataSource]: A dictionary containing Bokeh
                `ColumnDataSource` objects.
        """
        data = self.compute(rv_names=rv_names)

        output = {
            "marginal_x": ColumnDataSource(
                {
                    "x": data["marginal_x"]["x"],
                    "y": data["marginal_x"]["y"],
                    "label": data["marginal_x"]["label"],
                }
            ),
            "marginal_y": ColumnDataSource(
                {
                    "x": data["marginal_y"]["x"],
                    "y": data["marginal_y"]["y"],
                    "label": data["marginal_y"]["label"],
                }
            ),
            "joint": ColumnDataSource(
                {
                    "image": data["joint"]["image"],
                    "x": data["joint"]["xmin"],
                    "y": data["joint"]["ymin"],
                    "xmax": data["joint"]["xmax"],
                    "ymax": data["joint"]["ymax"],
                    "dw": data["joint"]["dw"],
                    "dh": data["joint"]["dh"],
                    "label": data["joint"]["label"],
                    "palette": data["joint"]["palette"],
                }
            ),
            "scatter": ColumnDataSource(
                {
                    "x": data["marginal_x"]["data"],
                    "y": data["marginal_y"]["data"],
                }
            ),
        }
        return output

    def create_figures(self, sources: Dict[str, ColumnDataSource]) -> Dict[str, Figure]:
        """Create Bokeh `Figure` objects.

        Args:
            sources (Dict[str, ColumnDataSource]): sources

        Returns:
            Dict[str, Figure]: Dictionary of Bokeh `Figure` objects.
        """
        plot_width = 500

        plot_height = 500
        min_border = 0
        rv_names = [source.data.get("label", [""])[0] for _, source in sources.items()]
        rv_names = [rv_name for rv_name in rv_names if rv_name]
        marginal_x_name = rv_names[0]
        marginal_y_name = rv_names[1]
        joint_dict = sources["joint"].data
        xmin = joint_dict["x"][0]
        xmax = joint_dict["xmax"][0]
        ymin = joint_dict["y"][0]
        ymax = joint_dict["ymax"][0]

        # Marginal x-axis
        marginal_x_fig = figure(
            plot_width=plot_width,
            plot_height=100,
            outline_line_color=None,
            x_range=[xmin, xmax],
            x_axis_location=None,
            min_border=min_border,
        )
        marginal_x_fig.yaxis.visible = False
        marginal_x_fig.xaxis.visible = False
        marginal_x_fig.grid.visible = False

        # Marginal y-axis
        marginal_y_fig = figure(
            plot_width=100,
            plot_height=plot_height,
            outline_line_color=None,
            y_range=[ymin, ymax],
            y_axis_location=None,
            min_border=min_border,
        )
        marginal_y_fig.yaxis.visible = False
        marginal_y_fig.xaxis.visible = False
        marginal_y_fig.grid.visible = False

        # Central figure
        joint_fig = figure(
            plot_width=plot_width,
            plot_height=plot_height,
            outline_line_color="black",
            min_border=min_border,
            x_axis_label=marginal_x_name,
            y_axis_label=marginal_y_name,
            match_aspect=True,
            background_fill_color="#440154",
            x_range=marginal_x_fig.x_range,
            y_range=marginal_y_fig.y_range,
        )
        joint_fig.grid.visible = False

        # Scatter figure
        scatter_fig = figure(
            plot_width=plot_width,
            plot_height=plot_height,
            outline_line_color="black",
            min_border=min_border,
            x_axis_label=marginal_x_name,
            y_axis_label=marginal_y_name,
            match_aspect=True,
            x_range=marginal_x_fig.x_range,
            y_range=marginal_y_fig.y_range,
        )
        utils.style_figure(scatter_fig)

        output = {
            "marginal_x": marginal_x_fig,
            "marginal_y": marginal_y_fig,
            "joint": joint_fig,
            "scatter": scatter_fig,
        }
        return output

    def create_glyphs(self) -> Dict[str, Dict[str, Glyph]]:
        """Create Bokeh `Glyph` objects.

        Returns:
            Dict[str, Dict[str, Glyph]]: Dictionary of Bokeh `Glyph` objects.
        """
        line_width = 2

        hover_line_width = 3
        muted_line_width = 1
        line_alpha = 0.7
        hover_line_alpha = 1
        muted_line_alpha = 0.1
        palette = utils.choose_palette(n=1)
        color = palette[0]

        marginal_x_glyph = Line(
            x="x",
            y="y",
            line_color=color,
            line_width=line_width,
            line_alpha=line_alpha,
            name="marginal_x",
        )
        marginal_x_hover_glyph = Line(
            x="x",
            y="y",
            line_color=color,
            line_width=hover_line_width,
            line_alpha=hover_line_alpha,
            name="marginal_x_hover",
        )
        marginal_x_muted_glyph = Line(
            x="x",
            y="y",
            line_color=color,
            line_width=muted_line_width,
            line_alpha=muted_line_alpha,
            name="marginal_x_muted",
        )

        marginal_y_glyph = Line(
            x="x",
            y="y",
            line_color=color,
            line_width=line_width,
            line_alpha=line_alpha,
            name="marginal_y",
        )
        marginal_y_hover_glyph = Line(
            x="x",
            y="y",
            line_color=color,
            line_width=hover_line_width,
            line_alpha=hover_line_alpha,
            name="marginal_y_hover",
        )
        marginal_y_muted_glyph = Line(
            x="x",
            y="y",
            line_color=color,
            line_width=muted_line_width,
            line_alpha=muted_line_alpha,
            name="marginal_y_muted",
        )

        # Joint
        joint_glyph = Image(
            image="image",
            x="x",
            y="y",
            dw="dw",
            dw_units="data",
            dh="dh",
            dh_units="data",
            color_mapper=LinearColorMapper(palette="Viridis256"),
        )

        # Scatter
        scatter_glyph = Circle(
            x="x",
            y="y",
            size=10,
            fill_color=color,
            line_color="white",
            fill_alpha=0.6,
            line_alpha=0.6,
            name="scatter",
        )
        scatter_hover_glyph = Circle(
            x="x",
            y="y",
            size=10,
            fill_color=color,
            line_color="black",
            fill_alpha=1,
            line_alpha=1,
            name="scatter_hover",
        )
        scatter_muted_glyph = Circle(
            x="x",
            y="y",
            size=10,
            fill_color=color,
            line_color="white",
            fill_alpha=0.1,
            line_alpha=0.1,
            name="scatter_hover",
        )

        output = {
            "marginal_x": {
                "glyph": marginal_x_glyph,
                "hover_glyph": marginal_x_hover_glyph,
                "muted_glyph": marginal_x_muted_glyph,
            },
            "marginal_y": {
                "glyph": marginal_y_glyph,
                "hover_glyph": marginal_y_hover_glyph,
                "muted_glyph": marginal_y_muted_glyph,
            },
            "joint": {"glyph": joint_glyph},
            "scatter": {
                "glyph": scatter_glyph,
                "hover_glyph": scatter_hover_glyph,
                "muted_glyph": scatter_muted_glyph,
            },
        }
        return output

    def update_figure(
        self,
        figs: Dict[str, Figure],
        rv_names: List[str],
        old_sources: Dict[str, ColumnDataSource],
    ) -> None:
        """Update the figures in the widget.

        Args:
            figs (Dict[str, Figure]): Dictionary of Bokeh `Figure` objects.
            rv_name (List[str]): Bean Machine string representation of an
                `RVIdentifier`.
            old_sources (Dict[str, ColumnDataSource]): old_sources

        Returns:
            None: Directly updates the given figures with new data.
        """
        new_sources = self.create_sources(rv_names=rv_names)

        # Update the x-axis
        figs["marginal_x"].renderers[0].data_source.data = dict(
            new_sources["marginal_x"].data
        )
        figs["marginal_x"].x_range.start = new_sources["joint"].data["x"][0]
        figs["marginal_x"].x_range.end = new_sources["joint"].data["xmax"][0]
        old_sources["joint"].data["image"] = new_sources["joint"].data["image"]
        old_sources["joint"].data["x"] = new_sources["joint"].data["x"]
        old_sources["joint"].data["dw"] = new_sources["joint"].data["dw"]
        figs["joint"].xaxis.axis_label = rv_names[0]

        # Update the y-axis
        figs["marginal_y"].renderers[0].data_source.data = dict(
            new_sources["marginal_y"].data
        )
        figs["marginal_y"].y_range.start = new_sources["joint"].data["y"][0]
        figs["marginal_y"].y_range.end = new_sources["joint"].data["ymax"][0]
        old_sources["joint"].data["image"] = new_sources["joint"].data["image"]
        old_sources["joint"].data["y"] = new_sources["joint"].data["y"]
        old_sources["joint"].data["dh"] = new_sources["joint"].data["dh"]
        figs["joint"].yaxis.axis_label = rv_names[1]

        # Update the scatter plot
        old_sources["scatter"].data = dict(new_sources["scatter"].data)

    def help_page(self):
        text = """
            <h2>
              Joint plot
            </h2>
            <p style="margin-bottom: 10px">
              A joint plot shows univariate marginals along the x and y axes. The
              central figure shows the bivariate marginal of both random variables.
            </p>
        """
        div = Div(text=text, disable_math=False, min_width=800)
        return div

    def modify_doc(self, doc) -> None:
        """Modify the document by adding the widget."""
        # Set the initial view.
        rv_names = self.rv_names[0:2]

        # Create data sources for the figures.
        sources = self.create_sources(rv_names=rv_names)

        # Create the figures.
        figs = self.create_figures(sources=sources)

        # Create glyphs and add them to the figure.
        glyphs = self.create_glyphs()
        for key, glyph_dict in glyphs.items():
            fig = figs[key]
            source = sources[key]
            utils.add_glyph_to_figure(fig=fig, source=source, glyph_dict=glyph_dict)

        # Create tooltips for the figure.

        # Widgets
        marginal_x_select = Select(
            title="Marginal x-axis",
            value=rv_names[0],
            options=self.rv_names,
        )
        marginal_y_select = Select(
            title="Marginal y-axis",
            value=rv_names[1],
            options=self.rv_names,
        )
        button_group = RadioButtonGroup(labels=["image", "data"], active=0)

        def update_marginal_x_rv(attr, old, new):
            rv_names = [new, marginal_y_select.value]
            self.update_figure(
                figs=figs,
                rv_names=rv_names,
                old_sources=sources,
            )

        def update_marginal_y_rv(attr, old, new):
            rv_names = [marginal_x_select.value, new]
            self.update_figure(
                figs=figs,
                rv_names=rv_names,
                old_sources=sources,
            )

        figures = [[None, None], [None, None]]
        if button_group.active == 0:
            figures = [[figs["marginal_x"], None], [figs["joint"], figs["marginal_y"]]]
        elif button_group.active == 1:
            figures = [
                [figs["marginal_x"], None],
                [figs["scatter"], figs["marginal_y"]],
            ]
        figures_grid = gridplot(figures)

        def update_button(attr, old, new):
            if new == 1:
                layout.children[-1].children[-1].children[1] = (figs["scatter"], 1, 0)
            if new == 0:
                layout.children[-1].children[-1].children[1] = (figs["joint"], 1, 0)

        marginal_x_select.on_change("value", update_marginal_x_rv)
        marginal_y_select.on_change("value", update_marginal_y_rv)
        button_group.on_change("active", update_button)
        # NOTE: We are using Bokeh's CustomJS model in order to reset the ranges of the
        #       figures.
        marginal_x_select.js_on_change(
            "value",
            CustomJS(
                args={
                    "marginal_x": figs["marginal_x"],
                    "marginal_y": figs["marginal_y"],
                },
                code="marginal_x.reset.emit()",
            ),
        )
        marginal_y_select.js_on_change(
            "value",
            CustomJS(
                args={
                    "marginal_x": figs["marginal_x"],
                    "marginal_y": figs["marginal_y"],
                },
                code="marginal_x.reset.emit(); marginal_y.reset.emit()",
            ),
        )

        widget_panel = Panel(
            child=column(
                row(marginal_x_select, marginal_y_select),
                button_group,
                figures_grid,
            ),
            title="Joint plot",
        )
        help_panel = Panel(child=self.help_page(), title="Help")
        tabs = Tabs(tabs=[widget_panel, help_panel])
        layout = column(tabs)
        doc.add_root(layout)

    def show_widget(self) -> None:
        """Display the widget."""
        show(self.modify_doc)
