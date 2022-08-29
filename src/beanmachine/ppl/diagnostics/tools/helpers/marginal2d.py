"""Methods used to generate the diagnostic tool."""
from typing import List, Optional

import arviz as az
import beanmachine.ppl.diagnostics.tools.typing.marginal2d as typing
import numpy as np
import numpy.typing as npt

from beanmachine.ppl.diagnostics.tools.helpers.marginal1d import (
    compute_data as m1d_data,
    compute_stats,
    hdi_data as compute_hdi_data,
)
from beanmachine.ppl.diagnostics.tools.helpers.plotting import (
    choose_palette,
    create_toolbar,
    filter_renderers,
    style_figure,
)
from bokeh.models.annotations import Band
from bokeh.models.glyphs import Circle, Image, Line
from bokeh.models.layouts import Column, GridBox, Row
from bokeh.models.mappers import LinearColorMapper
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets import Div
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.panels import Panel, Tabs
from bokeh.models.widgets.sliders import Slider
from bokeh.plotting.figure import figure


MARGINAL1D_PLOT_WIDTH = 500
MARGINAL1D_PLOT_HEIGHT = 100
MARGINAL2D_PLOT_WIDTH = 500
MARGINAL2D_PLOT_HEIGHT = 500
FIGURE_NAMES = ["x", "y", "xy"]


def compute_xy_data(
    x: npt.NDArray,
    y: npt.NDArray,
    x_label: str,
    y_label: str,
    x_stats: List[float],
    y_stats: List[float],
) -> typing.XYData:
    """Compute the two-dimensional marginal.

    Parameters
    ----------
    x : npt.NDArray
        The random variable data for the x-axis.
    y : npt.NDArray
        The random variable data for the y-axis.
    x_label : str
        The label for the x-axis.
    y_label : str
        The label for the y-axis.
    x_hdi_probability : float
        The HDI probability to use when calculating the HDI bounds for the x-axis.
    y_hdi_probability : float
        The HDI probability to use when calculating the HDI bounds for the y-axis.
    x_stats : List[float]
        Statistics for the x-axis; HDI bounds and the mean.
    y_stats : List[float]
        Statistics for the y-axis; HDI bounds and the mean.

    Returns
    -------
    typing.XYData
        The x-y joint marginal data object.
    """
    density, xmin, xmax, ymin, ymax = az.stats.density_utils._fast_kde_2d(x=x, y=y)
    distribution = {
        "image": [density.T.tolist()],
        "x": [xmin],
        "y": [ymin],
        "ymax": [ymax],
        "dw": [xmax - xmin],
        "dh": [ymax - ymin],
    }
    x = np.linspace(start=xmin, stop=xmax)
    y = np.linspace(start=ymin, stop=ymax)
    hdi = {
        "x": {
            "lower": {"x": [x_stats[0]] * len(y), "y": y.tolist()},
            "upper": {"x": [x_stats[2]] * len(y), "y": y.tolist()},
        },
        "y": {
            "lower": {"x": x.tolist(), "y": [y_stats[0]] * len(x)},
            "upper": {"x": x.tolist(), "y": [y_stats[2]] * len(x)},
        },
    }
    stats = {"x": [x_stats[1]], "y": [y_stats[1]]}
    labels = {"mean": [f"{x_label}/{y_label}"]}
    return {"distribution": distribution, "hdi": hdi, "stats": stats, "labels": labels}


def compute_y_data(
    data: npt.NDArray,
    bw_factor: float,
    hdi_probability: float,
) -> typing.YData:
    """Compute the marginal for the y-axis.

    In order to correctly display the HDI region in the figure, we need to separate it
    into a top and bottom Bokeh annotation. This is why the y data is handled
    differently than the x-axis data.

    Parameters
    ----------
    data : npt.NDArray
        Random variable data for the y-axis.
    bw_factor : float
        Multiplicative factor used when calculating the kernel density estimate.
    hdi_probability : float
        The HDI probability to use when calculating the HDI bounds.

    Returns
    -------
    typing.YData
        The data for the y-axis. This data looks very similar to the x-axis data, except
        where we need to split the HDI into a top and bottom portion.
    """
    hdi_bounds = az.stats.hdi(data, hdi_prob=hdi_probability)
    kde_x, kde_y, bandwidth = az.stats.density_utils._kde_linear(
        data,
        bw_return=True,
        bw_fct=bw_factor,
    )
    kde_y /= kde_y.max()
    hdi_data = compute_hdi_data(data, kde_x, kde_y, hdi_probability)
    hdi_x = hdi_data["base"]
    n = len(hdi_x)
    half_index = n // 2
    x_at_half_index = hdi_x[half_index]
    bottom_base = [0]
    bottom_lower = [hdi_bounds[0]]
    for i in range(len(kde_x)):
        if kde_x[i] <= x_at_half_index and kde_x[i] >= hdi_bounds[0]:
            bottom_base.append(kde_y[i])
            bottom_lower.append(kde_x[i])
    bottom_upper = [x_at_half_index] * len(bottom_base)
    top_base = [0]
    top_upper = [hdi_bounds[1]]
    for i in range(len(kde_x))[::-1]:
        if kde_x[i] >= x_at_half_index and kde_x[i] <= hdi_bounds[1]:
            top_base.append(kde_y[i])
            top_upper.append(kde_x[i])
    top_lower = [x_at_half_index] * len(top_base)
    stats_and_labels = compute_stats(
        data,
        kde_x,
        kde_y,
        hdi_probability,
        return_labels=True,
    )
    stats = stats_and_labels["stats"]
    stats = {"x": stats["y"], "y": stats["x"], "text": stats["text"]}
    labels = stats_and_labels["labels"]
    labels = {
        "x": labels["y"],
        "y": labels["x"],
        "text": labels["text"],
        "text_align": labels["text_align"],
        "x_offset": labels["y_offset"],
        "y_offset": labels["x_offset"],
    }
    return {
        "distribution": {
            "x": kde_y.tolist(),
            "y": kde_x.tolist(),
            "bandwidth": bandwidth,
        },
        "hdi": {
            "top": {"base": top_base, "lower": top_lower, "upper": top_upper},
            "bottom": {
                "base": bottom_base,
                "lower": bottom_lower,
                "upper": bottom_upper,
            },
        },
        "stats": stats,
        "labels": stats_and_labels["labels"],
    }


def compute_data(
    x_data: npt.NDArray,
    y_data: npt.NDArray,
    x_label: str,
    y_label: str,
    x_hdi_probability: float,
    y_hdi_probability: float,
    bw_factor: Optional[float] = None,
    bins: Optional[List[int]] = None,
) -> typing.Data:
    """Compute effective sample size estimates using the given data.

    Parameters
    ----------
    x_data : npt.NDArray
        The random variable data for the x-axis.
    y_data : npt.NDArray
        The random variable data for the y-axis.
    x_label : str
        The label for the x-axis.
    y_label : str
        The label for the y-axis.
    x_hdi_probability : float
        The HDI probability to use when calculating the HDI bounds for the x-axis.
    y_hdi_probability : float
        The HDI probability to use when calculating the HDI bounds for the y-axis.
    bw_factor : float, optional, default is 1.0 if None is given
        Multiplicative factor used when calculating the kernel density estimate.
    bins : List[int], optional, default is [128, 128] if None is given
        The grid points to use when calculating the two-dimensional KDE.

    Returns
    -------
    typing.Data
        A dictionary of data used for the tool.
    """
    if bw_factor is None:
        bw_factor = 1.0
    if bins is None:
        bins = [128, 128]

    x_data = x_data.reshape(-1)
    y_data = y_data.reshape(-1)

    x = m1d_data(x_data, bw_factor, x_hdi_probability)["marginal"]
    x_stats = [float(value) for value in x["stats"]["x"]]
    y = compute_y_data(y_data, bw_factor, y_hdi_probability)
    y_stats = [float(value) for value in y["stats"]["y"]]
    xy = compute_xy_data(
        x=x_data,
        y=y_data,
        x_label=x_label,
        y_label=y_label,
        x_stats=x_stats,
        y_stats=y_stats,
    )
    return {"x": x, "y": y, "xy": xy}


def create_sources(data: typing.Data) -> typing.Sources:
    """Create Bokeh sources from the given data that will be bound to glyphs.

    Parameters
    ----------
    data : typing.Data
        A dictionary of data used for the tool.

    Returns
    -------
    typing.Sources
        A dictionary of Bokeh ColumnDataSource objects.
    """
    output = {}
    for figure_name, figure_data in data.items():
        output[figure_name] = {}
        for glyph_name, glyph_data in figure_data.items():
            # Remove the bandwidth key so we can use the data directly in a source.
            if figure_name in ["x", "y"] and glyph_name == "distribution":
                glyph_data.pop("bandwidth")
                output[figure_name][glyph_name] = ColumnDataSource(data=glyph_data)
            elif figure_name == "y" and glyph_name == "hdi":
                output[figure_name][glyph_name] = {}
                for glyph_type in ["top", "bottom"]:
                    output[figure_name][glyph_name].update(
                        {glyph_type: ColumnDataSource(data=glyph_data[glyph_type])},
                    )
            elif figure_name == "xy" and glyph_name == "hdi":
                output[figure_name][glyph_name] = {}
                for axis in ["x", "y"]:
                    output[figure_name][glyph_name][axis] = {}
                    for glyph_type in ["lower", "upper"]:
                        output[figure_name][glyph_name][axis].update(
                            {
                                glyph_type: ColumnDataSource(
                                    data=glyph_data[axis][glyph_type],
                                ),
                            },
                        )
            else:
                output[figure_name][glyph_name] = ColumnDataSource(data=glyph_data)
    return output


def create_figures(x_rv_name: str, y_rv_name: str) -> typing.Figures:
    """Create the Bokeh figures used for the tool.

    Parameters
    ----------
    x_rv_name : str
        The x-axis label.
    y_rv_name : str
        The y-axis label.

    Returns
    -------
    typing.Figures
        A dictionary of Bokeh Figure objects.
    """
    output = {}
    for figure_name in FIGURE_NAMES:
        fig = figure()
        if figure_name == "x":
            fig = figure(
                width=MARGINAL1D_PLOT_WIDTH,
                height=MARGINAL1D_PLOT_HEIGHT,
                outline_line_color=None,
                min_border=None,
            )
            fig.yaxis.visible = False
            fig.xaxis.visible = False
            fig.grid.visible = False
        elif figure_name == "y":
            fig = figure(
                width=MARGINAL1D_PLOT_HEIGHT,
                height=MARGINAL1D_PLOT_WIDTH,
                outline_line_color=None,
                min_border=None,
            )
            fig.yaxis.visible = False
            fig.xaxis.visible = False
            fig.grid.visible = False
        elif figure_name == "xy":
            fig = figure(
                width=MARGINAL2D_PLOT_WIDTH,
                height=MARGINAL2D_PLOT_HEIGHT,
                outline_line_color="black",
                min_border=0,
                x_axis_label=x_rv_name,
                y_axis_label=y_rv_name,
                match_aspect=True,
                background_fill_color="#440154",
            )
            style_figure(fig)
            fig.grid.visible = False
        output[figure_name] = fig
    output["x"].x_range = output["xy"].x_range
    output["y"].y_range = output["xy"].y_range
    return output


def create_glyphs(data: typing.Data) -> typing.Glyphs:
    """Create the glyphs used for the figures of the tool.

    Parameters
    ----------
    data : typing.Data
        A dictionary of data used for the tool.

    Returns
    -------
    typing.Glyphs
        A dictionary of Bokeh Glyphs objects.
    """
    palette = choose_palette(4)
    glyph_color = palette[0]
    hover_glyph_color = palette[1]
    mean_color = palette[3]
    output = {}
    for figure_name, _ in data.items():
        output[figure_name] = {}
        if figure_name in ["x", "y"]:
            output[figure_name] = {
                "distribution": {
                    "glyph": Line(
                        x="x",
                        y="y",
                        line_color=glyph_color,
                        line_alpha=1.0,
                        line_width=2.0,
                        name=f"{figure_name}DistributionGlyph",
                    ),
                    "hover_glyph": Line(
                        x="x",
                        y="y",
                        line_color=hover_glyph_color,
                        line_alpha=1.0,
                        line_width=2.0,
                        name=f"{figure_name}DistributionHoverGlyph",
                    ),
                },
                "stats": {
                    "glyph": Circle(
                        x="x",
                        y="y",
                        size=10,
                        line_color="white",
                        fill_color=glyph_color,
                        fill_alpha=1.0,
                        name=f"{figure_name}StatsGlyph",
                    ),
                    "hover_glyph": Circle(
                        x="x",
                        y="y",
                        size=10,
                        line_color="white",
                        fill_color=hover_glyph_color,
                        fill_alpha=1.0,
                        name=f"{figure_name}StatsHoverGlyph",
                    ),
                },
            }
        if figure_name == "xy":
            output[figure_name] = {
                "distribution": Image(
                    image="image",
                    x="x",
                    y="y",
                    dw="dw",
                    dw_units="data",
                    dh="dh",
                    dh_units="data",
                    color_mapper=LinearColorMapper(palette="Viridis256"),
                    name=f"{figure_name}Glyph",
                ),
                "hdi": {
                    "x": {
                        "lower": {
                            "glyph": Line(
                                x="x",
                                y="y",
                                line_color="white",
                                line_alpha=0.7,
                                name=f"{figure_name}XLowerBoundHDIGlyph",
                            ),
                            "hover_glyph": Line(
                                x="x",
                                y="y",
                                line_color="white",
                                line_alpha=1.0,
                                name=f"{figure_name}XLowerBoundHDIHoverGlyph",
                            ),
                        },
                        "upper": {
                            "glyph": Line(
                                x="x",
                                y="y",
                                line_color="white",
                                line_alpha=0.7,
                                name=f"{figure_name}XUpperBoundHDIGlyph",
                            ),
                            "hover_glyph": Line(
                                x="x",
                                y="y",
                                line_color="white",
                                line_alpha=1.0,
                                name=f"{figure_name}XUpperBoundHDIHoverGlyph",
                            ),
                        },
                    },
                    "y": {
                        "lower": {
                            "glyph": Line(
                                x="x",
                                y="y",
                                line_color="white",
                                line_alpha=0.7,
                                name=f"{figure_name}YLowerBoundHDIGlyph",
                            ),
                            "hover_glyph": Line(
                                x="x",
                                y="y",
                                line_color="white",
                                line_alpha=1.0,
                                name=f"{figure_name}YLowerBoundHDIHoverGlyph",
                            ),
                        },
                        "upper": {
                            "glyph": Line(
                                x="x",
                                y="y",
                                line_color="white",
                                line_alpha=0.7,
                                name=f"{figure_name}YUpperBoundHDIGlyph",
                            ),
                            "hover_glyph": Line(
                                x="x",
                                y="y",
                                line_color="white",
                                line_alpha=1.0,
                                name=f"{figure_name}YUpperBoundHDIHoverGlyph",
                            ),
                        },
                    },
                },
                "stats": {
                    "glyph": Circle(
                        x="x",
                        y="y",
                        size=10,
                        line_color="red",
                        fill_color=mean_color,
                        fill_alpha=1.0,
                        name=f"{figure_name}MeanGlyph",
                    ),
                    "hover_glyph": Circle(
                        x="x",
                        y="y",
                        size=10,
                        line_color="white",
                        fill_color=mean_color,
                        fill_alpha=1.0,
                        name=f"{figure_name}MeanHoverGlyph",
                    ),
                },
            }
    return output


def add_glyphs(
    figures: typing.Figures,
    glyphs: typing.Glyphs,
    sources: typing.Sources,
) -> None:
    """Bind source data to glyphs and add the glyphs to the given figures.

    Parameters
    ----------
    figures : typing.Figures
        A dictionary of Bokeh Figure objects.
    glyphs : typing.Glyphs
        A dictionary of Bokeh Glyphs objects.
    sources : typing.Sources
        A dictionary of Bokeh ColumnDataSource objects.

    Returns
    -------
    None
        Adds data bound glyphs to the given figures directly.
    """
    for figure_name, fig in figures.items():
        figure_source = sources[figure_name]
        figure_glyphs = glyphs[figure_name]
        if figure_name in ["x", "y"]:
            for glyph_type in ["distribution", "stats"]:
                glyph_source = figure_source[glyph_type]
                figure_glyph = figure_glyphs[glyph_type]
                fig.add_glyph(
                    source_or_glyph=glyph_source,
                    glyph=figure_glyph["glyph"],
                    hover_glyph=figure_glyph["hover_glyph"],
                    name=figure_glyph["glyph"].name,
                )
        if figure_name == "xy":
            fig.add_glyph(
                source_or_glyph=figure_source["distribution"],
                glyph=figure_glyphs["distribution"],
                name=figure_glyphs["distribution"].name,
            )
            fig.add_glyph(
                source_or_glyph=figure_source["hdi"]["x"]["lower"],
                glyph=figure_glyphs["hdi"]["x"]["lower"]["glyph"],
                hover_glyph=figure_glyphs["hdi"]["x"]["lower"]["hover_glyph"],
                name=figure_glyphs["hdi"]["x"]["lower"]["glyph"].name,
            )
            fig.add_glyph(
                source_or_glyph=figure_source["hdi"]["x"]["upper"],
                glyph=figure_glyphs["hdi"]["x"]["upper"]["glyph"],
                hover_glyph=figure_glyphs["hdi"]["x"]["upper"]["hover_glyph"],
                name=figure_glyphs["hdi"]["x"]["upper"]["glyph"].name,
            )
            fig.add_glyph(
                source_or_glyph=figure_source["hdi"]["y"]["lower"],
                glyph=figure_glyphs["hdi"]["y"]["lower"]["glyph"],
                hover_glyph=figure_glyphs["hdi"]["y"]["lower"]["hover_glyph"],
                name=figure_glyphs["hdi"]["y"]["lower"]["glyph"].name,
            )
            fig.add_glyph(
                source_or_glyph=figure_source["hdi"]["y"]["upper"],
                glyph=figure_glyphs["hdi"]["y"]["upper"]["glyph"],
                hover_glyph=figure_glyphs["hdi"]["y"]["upper"]["hover_glyph"],
                name=figure_glyphs["hdi"]["y"]["upper"]["glyph"].name,
            )
            fig.add_glyph(
                source_or_glyph=figure_source["stats"],
                glyph=figure_glyphs["stats"]["glyph"],
                hover_glyph=figure_glyphs["stats"]["hover_glyph"],
                name=figure_glyphs["stats"]["glyph"].name,
            )


def create_annotations(sources: typing.Sources) -> typing.Annotations:
    """Create any annotations for the figures of the tool.

    Parameters
    ----------
    source : typing.Sources
        A dictionary of Bokeh ColumnDataSource objects.

    Returns
    -------
    typing.Annotations
        A dictionary of Bokeh Annotation objects.
    """
    palette = choose_palette(1)
    color = palette[0]
    return {
        "x": Band(
            base="base",
            lower="lower",
            upper="upper",
            source=sources["x"]["hdi"],
            level="underlay",
            fill_color=color,
            fill_alpha=0.2,
            line_color=None,
            name="xHDIAnnotation",
        ),
        "y": {
            "top": Band(
                base="base",
                lower="lower",
                upper="upper",
                source=sources["y"]["hdi"]["top"],
                level="underlay",
                fill_color=color,
                fill_alpha=0.2,
                line_color=None,
                name="yTopHDIAnnotation",
            ),
            "bottom": Band(
                base="base",
                lower="lower",
                upper="upper",
                source=sources["y"]["hdi"]["bottom"],
                level="underlay",
                fill_color=color,
                fill_alpha=0.2,
                line_color=None,
                name="yBottomHDIAnnotation",
            ),
        },
    }


def add_annotations(figures: typing.Figures, annotations: typing.Annotations) -> None:
    """Add the given annotations to the given figures of the tool.

    Parameters
    ----------
    figures : typing.Figures
        A dictionary of Bokeh Figure objects.
    annotations : typing.Annotations
        A dictionary of Bokeh Annotation objects.

    Returns
    -------
    None
        Adds annotations directly to the given figures.
    """
    figures["x"].add_layout(annotations["x"])
    figures["y"].add_layout(annotations["y"]["top"])
    figures["y"].add_layout(annotations["y"]["bottom"])


def create_tooltips(
    x_rv_name: str,
    y_rv_name: str,
    figures: typing.Figures,
) -> typing.Tooltips:
    """Create hover tools for the glyphs used in the figures of the tool.

    Parameters
    ----------
    x_rv_name : str
        The name of the random variable along the x-axis.
    y_rv_name : str
        The name of the random variable along the y-axis.
    figures : typing.Figures
        A dictionary of Bokeh Figure objects.

    Returns
    -------
    typing.Tooltips
        A dictionary of Bokeh HoverTools objects.
    """
    output = {}
    for figure_name, fig in figures.items():
        output[figure_name] = {}
        distribution_tips = []
        stats_tips = []
        distribution_renderers = None
        stats_renderers = None
        if figure_name in ["x", "y"]:
            distribution_renderers = filter_renderers(
                figure=fig,
                search="DistributionGlyph",
                substring=True,
            )
            stats_renderers = filter_renderers(
                figure=fig,
                search="StatsGlyph",
                substring=True,
            )
            if figure_name == "x":
                distribution_tips = [(x_rv_name, "@x")]
                stats_tips = [("", "@text")]
            if figure_name == "y":
                distribution_tips = [(y_rv_name, "@y")]
                stats_tips = [("", "@text")]
            output[figure_name] = {
                "distribution": HoverTool(
                    renderers=distribution_renderers,
                    tooltips=distribution_tips,
                ),
                "stats": HoverTool(renderers=stats_renderers, tooltips=stats_tips),
            }
        if figure_name == "xy":
            stats_renderers = filter_renderers(
                figure=fig,
                search="MeanGlyph",
                substring=True,
            )
            x_lower = filter_renderers(figure=fig, search="XLowerBound", substring=True)
            x_upper = filter_renderers(figure=fig, search="XUpperBound", substring=True)
            y_lower = filter_renderers(figure=fig, search="YLowerBound", substring=True)
            y_upper = filter_renderers(figure=fig, search="YUpperBound", substring=True)
            output[figure_name]["mean"] = HoverTool(
                renderers=stats_renderers,
                tooltips=[(x_rv_name, "@x"), (y_rv_name, "@y")],
            )
            output[figure_name]["hdi"] = {
                "x": {
                    "lower": HoverTool(renderers=x_lower, tooltips=[(x_rv_name, "@x")]),
                    "upper": HoverTool(renderers=x_upper, tooltips=[(x_rv_name, "@x")]),
                },
                "y": {
                    "lower": HoverTool(renderers=y_lower, tooltips=[(y_rv_name, "@y")]),
                    "upper": HoverTool(renderers=y_upper, tooltips=[(y_rv_name, "@y")]),
                },
            }
    return output


def add_tooltips(figures: typing.Figures, tooltips: typing.Tooltips) -> None:
    """Add the given tools to the figures.

    Parameters
    ----------
    figures : typing.Figures
        A dictionary of Bokeh Figure objects.
    tooltips : typing.Tooltips
        A dictionary of Bokeh HoverTools objects.

    Returns
    -------
    None
        Adds the tooltips directly to the given figures.
    """
    figures["xy"].add_tools(tooltips["xy"]["mean"])
    figures["xy"].add_tools(tooltips["xy"]["hdi"]["x"]["lower"])
    figures["xy"].add_tools(tooltips["xy"]["hdi"]["x"]["upper"])
    figures["xy"].add_tools(tooltips["xy"]["hdi"]["y"]["lower"])
    figures["xy"].add_tools(tooltips["xy"]["hdi"]["y"]["upper"])
    figures["x"].add_tools(tooltips["x"]["distribution"])
    figures["x"].add_tools(tooltips["x"]["stats"])
    figures["y"].add_tools(tooltips["y"]["distribution"])
    figures["y"].add_tools(tooltips["y"]["stats"])


def create_widgets(
    x_rv_name: str,
    y_rv_name: str,
    rv_names: List[str],
    bw_factor: float,
    x_bw: float,
    y_bw: float,
) -> typing.Widgets:
    """Create the widgets used in the tool.

    Parameters
    ----------
    x_rv_name : str
        The name of the random variable along the x-axis.
    y_rv_name : str
        The name of the random variable along the y-axis.
    rv_names : List[str]
        A list of all available random variable names.
    bw_factor : float
        Multiplicative factor used when calculating the kernel density estimate.
    x_bw : float
        The bandwidth used to calculate the KDE along the x-axis.
    y_bw : float
        The bandwidth used to calculate the KDE along the y-axis.

    Returns
    -------
    typing.Widgets
        A dictionary of Bokeh widget objects.
    """
    return {
        "x_rv_select": Select(value=x_rv_name, options=rv_names, title="Query (x)"),
        "y_rv_select": Select(value=y_rv_name, options=rv_names, title="Query (y)"),
        "x_hdi_slider": Slider(start=1, end=99, step=1, value=89, title="HDI (x)"),
        "y_hdi_slider": Slider(start=1, end=99, step=1, value=89, title="HDI (y)"),
        "x_bw_div": Div(text=f"Bandwidth {x_rv_name}: {bw_factor * x_bw}"),
        "y_bw_div": Div(text=f"Bandwidth {y_rv_name}: {bw_factor * y_bw}"),
    }


def help_page() -> Div:
    """Help tab for the tool.

    Returns
    -------
    Div
        Bokeh Div widget containing the help tab information.
    """
    text = """
    <h2>
      Joint plot
    </h2>
    <p style="margin-bottom: 10px">
      A joint plot shows univariate marginals along the x and y axes. The
      central figure shows the bivariate marginal of both random variables.
    </p>
    """
    return Div(
        text=text,
        disable_math=False,
        min_width=MARGINAL1D_PLOT_WIDTH + MARGINAL2D_PLOT_WIDTH,
    )


def create_figure_grid(figures: typing.Figures) -> Row:
    """Layout the given figures in a grid, and make one toolbar.

    Parameters
    ----------
    figures : typing.Figures
        A dictionary of Bokeh Figure objects.

    Returns
    -------
    Row
        A Bokeh layout object.
    """
    toolbar = create_toolbar(figures)
    grid_box = GridBox(
        children=[
            [figures["x"], 0, 0],
            [figures["xy"], 1, 0],
            [figures["y"], 1, 1],
        ],
        sizing_mode=None,
    )
    return Row(children=[grid_box, toolbar])


def create_view(widgets: typing.Widgets, figures: typing.Figures) -> Tabs:
    """Create the tool view.

    Parameters
    ----------
    widgets : typing.Widgets
        A dictionary of Bokeh widget objects.
    figures : typing.Figures
        A dictionary of Bokeh Figure objects.

    Returns
    -------
    Tabs
        Bokeh Tabs objects.
    """
    help_panel = Panel(child=help_page(), title="Help", name="helpPanel")
    figure_grid = create_figure_grid(figures)
    tool_panel = Panel(
        child=Row(
            children=[
                Column(
                    children=[
                        Row(
                            children=[
                                widgets["x_rv_select"],
                                widgets["y_rv_select"],
                            ],
                        ),
                        figure_grid,
                    ],
                ),
                Column(
                    children=[
                        widgets["x_hdi_slider"],
                        widgets["y_hdi_slider"],
                        widgets["x_bw_div"],
                        widgets["y_bw_div"],
                    ],
                ),
            ],
        ),
        title="Marginal 2D",
        name="marginalToolPanel",
    )
    return Tabs(tabs=[tool_panel, help_panel])


def update(
    x_rv_data: npt.NDArray,
    y_rv_data: npt.NDArray,
    x_rv_name: str,
    y_rv_name: str,
    sources: typing.Sources,
    figures: typing.Figures,
    tooltips: typing.Tooltips,
    x_hdi_probability: float,
    y_hdi_probability: float,
    bw_factor: float,
) -> tuple[float, float]:
    """Update the tool based on user interaction.

    Parameters
    ----------
    x_rv_data : npt.NDArray
        A 2D NumPy array where the length of the first dimension is the number of chains
        of the model, and the length of the second dimension is the number of draws of
        the model.
    y_rv_data : npt.NDArray
        A 2D NumPy array where the length of the first dimension is the number of chains
        of the model, and the length of the second dimension is the number of draws of
        the model.
    x_rv_name : str
        The name of the random variable along the x-axis.
    y_rv_name : str
        The name of the random variable along the y-axis.
    sources : typing.Sources
        A dictionary of Bokeh ColumnDataSource objects.
    figures : typing.Figures
        A dictionary of Bokeh Figure objects.
    tooltips : typing.Tooltips
        A dictionary of Bokeh HoverTools objects.
    x_hdi_probability : float
        The HDI probability to use when calculating the HDI bounds for the x-axis.
    y_hdi_probability : float
        The HDI probability to use when calculating the HDI bounds for the y-axis.
    bw_factor : float
        Multiplicative factor used when calculating the kernel density estimate.

    Returns
    -------
    None
        Updates Bokeh ColumnDataSource objects.
    """
    computed_data = compute_data(
        x_rv_data,
        y_rv_data,
        x_rv_name,
        y_rv_name,
        x_hdi_probability,
        y_hdi_probability,
        bw_factor,
    )
    x_bw = computed_data["x"]["distribution"].pop("bandwidth")
    y_bw = computed_data["y"]["distribution"].pop("bandwidth")
    for figure_name, figure_data in computed_data.items():
        figure_sources = sources[figure_name]
        if figure_name == "x":
            dist = figure_sources["distribution"]
            dist.data = figure_data["distribution"]
            figure_sources["hdi"].data = figure_data["hdi"]
            figure_sources["stats"].data = figure_data["stats"]
        if figure_name == "y":
            dist = figure_sources["distribution"]
            hdi_top = figure_sources["hdi"]["top"]
            hdi_bottom = figure_sources["hdi"]["bottom"]
            dist.data = figure_data["distribution"]
            hdi_top.data = figure_data["hdi"]["top"]
            hdi_bottom.data = figure_data["hdi"]["bottom"]
            figure_sources["stats"].data = figure_data["stats"]
        if figure_name == "xy":
            dist = figure_sources["distribution"]
            hdi_x_lower = figure_sources["hdi"]["x"]["lower"]
            hdi_x_upper = figure_sources["hdi"]["x"]["upper"]
            hdi_y_lower = figure_sources["hdi"]["y"]["lower"]
            hdi_y_upper = figure_sources["hdi"]["y"]["upper"]
            dist.data = figure_data["distribution"]
            hdi_x_lower.data = figure_data["hdi"]["x"]["lower"]
            hdi_x_upper.data = figure_data["hdi"]["x"]["upper"]
            hdi_y_lower.data = figure_data["hdi"]["y"]["lower"]
            hdi_y_upper.data = figure_data["hdi"]["y"]["upper"]
            figure_sources["stats"].data = figure_data["stats"]
    figures["xy"].xaxis.axis_label = x_rv_name
    figures["xy"].yaxis.axis_label = y_rv_name
    tooltips["xy"]["mean"].tooltips = [
        (x_rv_name, "@x"),
        (y_rv_name, "@y"),
    ]
    tooltips["xy"]["hdi"]["x"]["lower"].tooltips = [(x_rv_name, "@x")]
    tooltips["xy"]["hdi"]["x"]["upper"].tooltips = [(x_rv_name, "@x")]
    tooltips["xy"]["hdi"]["y"]["lower"].tooltips = [(y_rv_name, "@y")]
    tooltips["xy"]["hdi"]["y"]["upper"].tooltips = [(y_rv_name, "@y")]
    return (x_bw, y_bw)
