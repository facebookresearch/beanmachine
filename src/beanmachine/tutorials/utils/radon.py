# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Data ETL for the radon tutorial."""
from typing import Dict

import arviz as az
import numpy as np
import pandas as pd
import statsmodels.api as sm
from beanmachine.tutorials.utils import etl
from bokeh.layouts import layout
from bokeh.models import ColumnDataSource, HoverTool, Whisker
from bokeh.models.tickers import FixedTicker
from bokeh.plotting import figure, gridplot
from bokeh.plotting.figure import Figure
from pandas import DataFrame
from scipy import stats


class ExtractRadonTutorialData(etl.Extract):
    """Extract data for the radon tutorial."""

    _SCHEME = "http"
    _NETLOC = "stat.columbia.edu"
    _PATH = "~gelman/arm/examples/radon"
    _RADON_FILENAME = "srrs2.dat"
    _COUNTY_FILENAME = "cty.dat"

    def __init__(self) -> None:
        self.radon_data_url = self._build_url(self._RADON_FILENAME)
        self.county_data_url = self._build_url(self._COUNTY_FILENAME)
        self.extracted_data = self._extract()

    def _build_url(self, filename: str) -> str:
        return self._SCHEME + "://" + self._NETLOC + "/" + self._PATH + "/" + filename

    def _extract(self) -> Dict[str, DataFrame]:
        radon_df = pd.read_csv(self.radon_data_url, skipinitialspace=True)
        county_df = pd.read_csv(self.county_data_url, skipinitialspace=True)
        return {"radon": radon_df, "county": county_df}


class TransformRadonTutorialData(etl.Transform):
    """Transform radon data for the tutorial."""

    extractor = ExtractRadonTutorialData
    counties = None

    def _tx_radon_df(self, df: DataFrame) -> DataFrame:
        # Select only counties in Minnesota.
        df = df[df["state"] == "MN"].copy()

        # Fix the spelling mistakes.
        misspellings = {
            "ST LOUIS": "ST. LOUIS",
            "SHAKOPEE-MDEWAKANTO": "SHAKOPEE MDEWAKANTON SIOUX",
            "MILLIE LACS": "MILLE LACS",
        }
        df["county"] = df["county"].str.strip().replace(misspellings)

        # Sort the data.
        df = df.sort_values(by="county").reset_index(drop=True)

        # Create an index for the counties.
        self.counties = sorted(df["county"].unique())
        counties_dict = {county: i for i, county in enumerate(self.counties)}
        df["county_index"] = df["county"].map(counties_dict)

        # We only need a few columns for this analysis.
        df = df[["county_index", "county", "floor", "activity"]]

        # Calculate the logarithm of the activity data.
        df["log_activity"] = np.log(df["activity"].values + 0.1)

        return df

    def _tx_cty_df(self, df: DataFrame) -> DataFrame:
        # Filter for Minnesota.
        df = df[df["st"] == "MN"].copy()

        # Fix spelling errors.
        df["cty"] = df["cty"].replace(
            {
                "BIGSTONE": "BIG STONE",
                "BLUEEARTH": "BLUE EARTH",
                "CROWWING": "CROW WING",
                "LACQUIPARLE": "LAC QUI PARLE",
                "LAKEOFTHEWOODS": "LAKE OF THE WOODS",
                "LESUEUR": "LE SUEUR",
                "MILLELACS": "MILLE LACS",
                "OTTERTAIL": "OTTER TAIL",
                "STLOUIS": "ST. LOUIS",
                "YELLOWMEDICINE": "YELLOW MEDICINE",
            }
        )

        # Drop counties in the `cty` data not found in the `srrs` data.
        df = df.drop(df[~df["cty"].isin(self.counties)].index)

        # Drop duplicates.
        df = df.drop(df[df["cty"].duplicated()].index)

        # We only need a few columns for this analysis.
        df = df[["cty", "Uppm"]].sort_values(by="cty").copy()
        df = df.rename(columns={"cty": "county"})

        # Sort the data
        df = df.sort_values(by="county").reset_index(drop=True)

        # Calculate the logarithm of the uranium concentration.
        df["log_Uppm"] = np.log(df["Uppm"].values)

        return df

    def _tx_data(self, radon: DataFrame, county: DataFrame) -> DataFrame:
        radon_df = self._tx_radon_df(radon)
        county_df = self._tx_cty_df(county)
        df = pd.merge(
            left=radon_df,
            right=county_df,
            left_on="county",
            right_on="county",
            how="left",
        )
        return df

    def _transform(self) -> DataFrame:
        """Transform the data."""
        radon_df = self.extracted_data["radon"]
        county_df = self.extracted_data["county"]
        return self._tx_data(radon_df, county_df)


class LoadRadonTutorialData(etl.Load):
    """Load the transformed radon data."""

    transformer = TransformRadonTutorialData
    filename = "radon.csv"

    def _load(self) -> DataFrame:
        """Load transformed data."""
        return self.transformed_data


def load_data() -> DataFrame:
    """Load the radon data."""
    loader = LoadRadonTutorialData()
    return loader.load()


def log_plot_comparison(data: pd.Series, nbins: int = 40):
    """Compare data plot with the log(data) plot."""
    # Take the log of the given data.
    log_data = np.log(data + 0.01)

    # Determine histograms for the data.
    histogram, bins = np.histogram(data, bins=nbins)
    log_histogram, log_bins = np.histogram(log_data, bins=nbins)

    # Estimate the densities and scale them to their histograms.
    kde = sm.nonparametric.KDEUnivariate(data)
    kde.fit()
    scaled_density = (kde.density / kde.density.max()) * histogram.max()
    log_kde = sm.nonparametric.KDEUnivariate(log_data)
    log_kde.fit()
    log_scaled_density = (log_kde.density / log_kde.density.max()) * log_histogram.max()

    # Create the plots.
    plot = figure(
        plot_width=500,
        plot_height=500,
        title=f"Histogram of {data.name}",
        y_axis_label="Counts",
        x_axis_label=data.name,
    )
    log_plot = figure(
        plot_width=500,
        plot_height=500,
        title=f"Histogram of log({data.name})",
        y_axis_label="Counts",
        x_axis_label=f"log({data.name})",
    )

    # Bind data to the plots.
    density_source = ColumnDataSource({"x": kde.support, "y": scaled_density})
    density_glyph = plot.line(
        x="x",
        y="y",
        source=density_source,
        line_color="black",
        line_width=2.0,
        line_alpha=0.7,
        hover_line_color="brown",
        hover_line_width=3.0,
        hover_line_alpha=1.0,
        legend_label="Kernel density estimation",
    )
    density_tooltips = HoverTool(
        renderers=[density_glyph],
        tooltips=[
            ("Density", ""),
            ("Count", "@y"),
            (f"{data.name.title()}", "@x"),
        ],
    )
    plot.add_tools(density_tooltips)
    histogram_source = ColumnDataSource(
        {
            "left": bins[:-1],
            "right": bins[1:],
            "top": histogram,
            "bottom": np.zeros(histogram.shape[0]),
            "activity": [
                f"{item[0]:.3f} - {item[1]:.3f}" for item in zip(bins[:-1], bins[1:])
            ],
        }
    )
    histogram_glyph = plot.quad(
        left="left",
        right="right",
        top="top",
        bottom="bottom",
        source=histogram_source,
        fill_color="steelblue",
        fill_alpha=0.7,
        line_color="white",
        line_width=1.0,
        hover_color="orange",
        hover_alpha=1.0,
        hover_line_color="black",
        hover_line_width=2.0,
        legend_label="Histogram",
    )
    histogram_tooltips = HoverTool(
        renderers=[histogram_glyph],
        tooltips=[
            ("Histogram", ""),
            ("Counts", "@top"),
            (f"{data.name.title()}", "@activity"),
        ],
    )
    plot.add_tools(histogram_tooltips)

    log_density_source = ColumnDataSource(
        {"x": log_kde.support, "y": log_scaled_density}
    )
    log_density_glyph = log_plot.line(
        x="x",
        y="y",
        source=log_density_source,
        line_color="black",
        line_width=2.0,
        line_alpha=0.7,
        hover_line_color="brown",
        hover_line_width=3.0,
        hover_line_alpha=1.0,
        legend_label="Kernel density estimation",
    )
    log_density_tooltips = HoverTool(
        renderers=[log_density_glyph],
        tooltips=[
            ("Density", ""),
            ("Count", "@y"),
            (f"log({data.name})", "@x"),
        ],
    )
    log_plot.add_tools(log_density_tooltips)
    log_histogram_source = ColumnDataSource(
        {
            "left": log_bins[:-1],
            "right": log_bins[1:],
            "top": log_histogram,
            "bottom": np.zeros(log_histogram.shape[0]),
            "activity": [
                f"{item[0]:.3f} - {item[1]:.3f}"
                for item in zip(log_bins[:-1], log_bins[1:])
            ],
        }
    )
    log_histogram_glyph = log_plot.quad(
        left="left",
        right="right",
        top="top",
        bottom="bottom",
        source=log_histogram_source,
        fill_color="steelblue",
        fill_alpha=0.7,
        line_color="white",
        line_width=1.0,
        hover_color="orange",
        hover_alpha=1.0,
        hover_line_color="black",
        hover_line_width=2.0,
        legend_label="Histogram",
    )
    log_histogram_tooltips = HoverTool(
        renderers=[log_histogram_glyph],
        tooltips=[
            ("Histogram", ""),
            ("Counts", "@top"),
            (f"log({data.name.title()})", "@activity"),
        ],
    )
    log_plot.add_tools(log_histogram_tooltips)

    # Style the plots.
    plot.outline_line_color = "black"
    plot.grid.grid_line_color = "grey"
    plot.grid.grid_line_alpha = 0.2
    plot.grid.grid_line_width = 0.3
    log_plot.outline_line_color = "black"
    log_plot.grid.grid_line_color = "grey"
    log_plot.grid.grid_line_alpha = 0.2
    log_plot.grid.grid_line_width = 0.3

    return gridplot([[plot, log_plot]])


def floor_plot(df):
    # Create the plot.
    radon_floor_plot = figure(
        plot_width=500,
        plot_height=500,
        title="log(radon) measurement vs floor",
        y_range=[-6, 6],
        x_range=[-0.5, 1.5],
        x_axis_label="Floor",
        y_axis_label="log(radon)",
    )

    # Prepare data for the plot.
    basement_floor_data = df[df["floor"] == 0]["log_activity"].values
    basement_floor_kde = sm.nonparametric.KDEUnivariate(basement_floor_data)
    basement_floor_kde.fit()
    ground_floor_data = df[df["floor"] == 1]["log_activity"].values
    ground_floor_kde = sm.nonparametric.KDEUnivariate(ground_floor_data)
    ground_floor_kde.fit()
    radon_floor_source = ColumnDataSource(
        {
            "x": (df["floor"].values + np.random.normal(scale=0.02, size=df.shape[0])),
            "y": df["log_activity"].values,
            "county": df["county"].values,
            "color": df["floor"].apply(
                lambda floor: "orange" if floor == 1 else "steelblue"
            ),
        }
    )

    # Bind data to the plot.
    radon_floor_glyph = radon_floor_plot.circle(
        x="x",
        y="y",
        source=radon_floor_source,
        size=5,
        fill_color="color",
        line_color="white",
        alpha=0.7,
    )
    radon_floor_tooltips = HoverTool(
        renderers=[radon_floor_glyph],
        tooltips=[
            ("County", "@county"),
            ("log(radon)", "@y{0.000}"),
        ],
    )
    radon_floor_plot.add_tools(radon_floor_tooltips)
    x = [-0.25, 0.25]
    y = [basement_floor_kde.support[np.argmax(basement_floor_kde.density)]] * 2
    radon_floor_plot.line(
        y=y,
        x=x,
        line_color="steelblue",
        line_dash="dashed",
        line_width=4.0,
        alpha=0.5,
    )
    x = 0.25 * (basement_floor_kde.density / basement_floor_kde.density.max())
    radon_floor_plot.line(
        y=basement_floor_kde.support,
        x=x,
        line_color="steelblue",
        alpha=0.7,
    )
    radon_floor_plot.line(
        y=basement_floor_kde.support,
        x=-x,
        line_color="steelblue",
        alpha=0.7,
    )
    x = 0.25 * (ground_floor_kde.density / ground_floor_kde.density.max())
    radon_floor_plot.line(
        y=ground_floor_kde.support,
        x=1 + x,
        line_color="orange",
        alpha=0.7,
    )
    radon_floor_plot.line(
        y=ground_floor_kde.support,
        x=1 - x,
        line_color="orange",
        alpha=0.7,
    )
    radon_floor_plot.line(
        y=[ground_floor_kde.support[np.argmax(ground_floor_kde.density)]] * 2,
        x=[0.75, 1.25],
        line_color="orange",
        line_dash="dashed",
        line_width=4.0,
        alpha=0.5,
    )

    # Style the plot.
    radon_floor_plot.xaxis.ticker = FixedTicker(ticks=[0, 1])
    radon_floor_plot.xaxis.major_label_overrides = {
        0: "Basement",
        1: "First",
    }
    radon_floor_plot.grid.grid_line_color = None
    radon_floor_plot.outline_line_color = "black"
    radon_floor_plot.output_backend = "svg"

    return radon_floor_plot


def sample_of_priors():
    half_cauchys = []
    normals = []
    x = np.linspace(0, 100, 10000)
    X = np.linspace(-100, 100, 10000)
    for i in range(1, 6):
        half_cauchys.append(stats.halfcauchy(loc=0, scale=i).pdf(x))
        normals.append(stats.norm(0, i).pdf(X))

    cauchy_plot = figure(
        plot_width=500,
        plot_height=500,
        title="Half Cauchy priors",
        x_range=[1e-2, 100],
        x_axis_type="log",
    )
    normal_plot = figure(
        plot_width=500,
        plot_height=500,
        title="Normal priors",
        x_range=[-10, 10],
    )
    colors = ["steelblue", "magenta", "black", "orange", "brown"]

    for i, half_cauchy in enumerate(half_cauchys):
        cauchy_source = ColumnDataSource({"x": x, "y": half_cauchy})
        cauchy_plot.line(
            x="x",
            y="y",
            source=cauchy_source,
            line_width=2,
            color=colors[i],
            legend_label=f"μ = 0; γ = {i + 1}",
        )

        normal_source = ColumnDataSource({"x": X, "y": normals[i]})
        normal_plot.line(
            x="x",
            y="y",
            source=normal_source,
            line_width=2,
            color=colors[i],
            legend_label=f"μ = 0; σ = {i + 1}",
        )

    cauchy_plot.outline_line_color = normal_plot.outline_line_color = "black"
    cauchy_plot.grid.grid_line_alpha = normal_plot.grid.grid_line_alpha = 0.2
    cauchy_plot.grid.grid_line_color = normal_plot.grid.grid_line_color = "grey"
    cauchy_plot.grid.grid_line_width = normal_plot.grid.grid_line_width = 0.2

    priors_plot = gridplot([[cauchy_plot, normal_plot]])
    return priors_plot


def plot_trace_ranks(keys, values, samples):
    parameters = dict(zip(keys, values))
    plots = []
    colors = ["#2a2eec", "#fa7c17", "#328c06", "#c10c90"]
    for title, parameter in parameters.items():
        data = {title: samples.get(parameter).numpy()}
        trace = az.plot_trace(data, show=False, kind="rank_bars").reshape(-1)
        for i, p in enumerate(trace):
            if i == 0:
                p.plot_width = 300
                for j, renderer in enumerate(p.renderers):
                    renderer._property_values["glyph"].line_color = colors[j]
                    renderer._property_values["glyph"].line_dash = "solid"
                    renderer._property_values["glyph"].line_width = 2
                    renderer._property_values["glyph"].line_alpha = 0.6
            else:
                p.plot_width = 600
            p.plot_height = 300
            p.outline_line_color = "black"
            p.grid.grid_line_alpha = 0.2
            p.grid.grid_line_color = "grey"
            p.grid.grid_line_width = 0.2
        plots.append(layout([[trace[0], trace[1]]]))
    return gridplot([plots])


def sample_county_trace_ranks(sample_counties: Dict[int, str], alphas) -> Figure:
    colors = ["#2a2eec", "#fa7c17", "#328c06", "#c10c90"]
    plots = []
    for index, county in sample_counties.items():
        data = {f"α[{county}]": alphas[:, :, index]}
        trace = az.plot_trace(data, show=False, kind="rank_bars").reshape(-1)
        for i, p in enumerate(trace):
            if i == 0:
                p.plot_width = 300
                for j, renderer in enumerate(p.renderers):
                    renderer._property_values["glyph"].line_color = colors[j]
                    renderer._property_values["glyph"].line_dash = "solid"
                    renderer._property_values["glyph"].line_width = 2
                    renderer._property_values["glyph"].line_alpha = 0.6
            else:
                p.plot_width = 600
            p.plot_height = 300
            p.outline_line_color = "black"
            p.grid.grid_line_alpha = 0.2
            p.grid.grid_line_color = "grey"
            p.grid.grid_line_width = 0.2
        plots.append(layout([[trace[0], trace[1]]]))
    return gridplot([plots])


def uranium(summary_df: DataFrame, df: DataFrame) -> Figure:
    """
    Plot uranium linear regression.

    :param summary_df: The dataframe output from arviz.
    :param df: The original dataframe data.
    :returns plot: A bokeh Figure object.
    """
    alpha_hat_df = (
        summary_df[["mean", "sd"]]
        .loc[summary_df.index.astype(str).str.startswith("alpha_hat"), :]
        .reset_index(drop=True)
        .copy()
    )
    alpha_hat_df["log_Uppm"] = df["log_Uppm"].values
    alpha_hat_df["county"] = df["county"].values
    alpha_hat_df = alpha_hat_df.drop_duplicates().reset_index(drop=True)
    alpha_hat_df["lower"] = alpha_hat_df["mean"] - alpha_hat_df["sd"]
    alpha_hat_df["upper"] = alpha_hat_df["mean"] + alpha_hat_df["sd"]

    intercepts_source = ColumnDataSource(
        {
            "x": alpha_hat_df["log_Uppm"].values,
            "y": alpha_hat_df["mean"].values,
            "lower": alpha_hat_df["lower"].values,
            "upper": alpha_hat_df["upper"].values,
            "county": alpha_hat_df["county"].values,
        }
    )

    plot = figure(
        plot_width=1000,
        plot_height=500,
        title="Partial-pooling with individual and group level predictors",
        x_axis_label="log(uranium)",
        y_axis_label="Intercept estimate (log(radon activity))",
        y_range=[0.5, 2.2],
        x_range=[-1, 0.6],
    )
    markers = plot.circle(
        x="x",
        y="y",
        source=intercepts_source,
        size=10,
        fill_color="steelblue",
        line_color="white",
        fill_alpha=0.7,
        line_alpha=0.7,
        hover_fill_color="orange",
        hover_line_color="black",
        hover_fill_alpha=1.0,
        legend_label="County",
    )
    tooltips = HoverTool(
        renderers=[markers],
        tooltips=[
            ("County", "@county"),
            ("Estimated α", "@y{0.000}"),
        ],
    )
    plot.add_tools(tooltips)
    whiskers = Whisker(
        base="x",
        upper="upper",
        lower="lower",
        source=intercepts_source,
        line_color="steelblue",
    )
    whiskers.upper_head.line_color = "steelblue"
    whiskers.lower_head.line_color = "steelblue"
    plot.add_layout(whiskers)

    x = np.array([-1, 1])
    a = summary_df.loc[
        summary_df.index.astype(str).str.startswith("mu_alpha"), "mean"
    ].values
    g = summary_df.loc[
        summary_df.index.astype(str).str.startswith("gamma"), "mean"
    ].values
    y = a + g * x
    plot.line(
        x=x,
        y=y,
        line_color="black",
        line_alpha=0.3,
        line_width=3,
        legend_label="Estimated linear regression",
        level="underlay",
    )

    plot.outline_line_color = "black"
    plot.grid.grid_line_alpha = 0.2
    plot.grid.grid_line_color = "grey"
    plot.grid.grid_line_width = 0.2
    plot.legend.location = "top_left"
    plot.output_backend = "svg"

    return plot
