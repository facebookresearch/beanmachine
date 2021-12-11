# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Helper module for the hierarchical model tutorial."""
from typing import Dict, List

import arviz as az
import numpy as np
import pandas as pd
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.tutorials.utils import plots
from bokeh.models import Arrow, Band, HoverTool, VeeHead, Whisker
from bokeh.models.sources import ColumnDataSource
from bokeh.plotting import figure
from bokeh.plotting.figure import Figure
from scipy.stats import beta, pareto, uniform


COLORS = ["#2a2eec", "#fa7c17", "#328c06", "#c10c90"]


def plot_current_hits(df: pd.DataFrame) -> Figure:
    """
    Plot ``current hits`` data.

    :param df: Dataframe of the model data.
    :type df: pd.DataFrame
    :return: Bokeh figure of the current hits data.
    :rtype: Figure
    """
    # Prepare data for the figure.
    y = df["Current hits"].values[::-1]
    x = np.linspace(0, max(y) + 1, len(y))
    names = df["Name"].values[::-1]

    # Create the figure data source.
    source = ColumnDataSource({"x": x, "y": y, "name": names})

    # Add labels to the figure.
    figure_kwargs = {
        "title": "Current hits",
        "y_axis_label": "Hits",
        "x_axis_label": "Player",
    }

    # Add tooltips to the figure.
    tips = [("Name", "@name"), ("Hits", "@y")]

    # Create the figure.
    p = plots.scatter_plot(
        plot_source=source,
        figure_kwargs=figure_kwargs,
        tooltips=tips,
    )

    # Style the figure.
    plots.style(p)
    p.xaxis.major_tick_line_color = None
    p.xaxis.minor_tick_line_color = None
    p.xaxis.major_label_text_font_size = "0pt"

    return p


def plot_complete_pooling_priors() -> Figure:
    """
    Plot a family of priors for the complete-pooling model.

    :return: Bokeh figure of the priors.
    :rtype: Figure
    """
    # Support for the priors.
    N_ = int(1e4)
    x = np.linspace(0, 1, N_)

    # Prior PDFs.
    beta_samples = beta(1, 1).pdf(x)
    uniform_samples = uniform(0, 1).pdf(x)

    # Create the figure data sources.
    beta_source = ColumnDataSource({"x": x, "y": beta_samples})
    uniform_source = ColumnDataSource({"x": x, "y": uniform_samples})

    # Create the figure.
    plot = figure(
        plot_width=400,
        plot_height=400,
        title="Beta(1, 1) vs Uniform",
        x_axis_label="Support",
        x_range=[0, 1],
        y_range=[0.8, 1.2],
    )

    # Create glyphs on the figure.
    plot.line(
        x="x",
        y="y",
        source=beta_source,
        line_color="steelblue",
        line_alpha=0.6,
        line_width=6,
        legend_label="Beta(1, 1)",
    )
    plot.line(
        x="x",
        y="y",
        source=uniform_source,
        line_color="orange",
        line_alpha=1,
        line_width=2,
        legend_label="Uniform distribution",
    )

    # Style the figure.
    plots.style(plot)
    plot.yaxis.major_tick_line_color = None
    plot.yaxis.minor_tick_line_color = None

    return plot


def plot_complete_pooling_diagnostics(samples: MonteCarloSamples) -> List[Figure]:
    """
    Plot the complete-pooling diagnostics.

    :param samples: Bean Machine inference object.
    :type samples: MonteCarloSamples
    :return: Bokeh figure of some visual diagnostics.
    :rtype: List[Figure]
    """
    # Prepare the data for the figure.
    diagnostics_data = {
        key.__dict__["wrapper"].__name__: value.values
        for key, value in samples.to_xarray().data_vars.items()
    }

    # Cycle through each query and create the diagnostics plots using arviz.
    diagnostics_plots = []
    for key, value in diagnostics_data.items():
        ac_plot = az.plot_autocorr({key: value}, show=False)[0].tolist()
        tr_plot = az.plot_trace(
            {key: value},
            kind="rank_bars",
            show=False,
        )[0].tolist()
        for i, p in enumerate(tr_plot):
            # Style the plots from arviz.
            if i == 0:
                p.plot_width = 300
                for j, renderer in enumerate(p.renderers):
                    renderer._property_values["glyph"].line_color = COLORS[j]
                    renderer._property_values["glyph"].line_dash = "solid"
                    renderer._property_values["glyph"].line_width = 2
                    renderer._property_values["glyph"].line_alpha = 0.6
            else:
                p.plot_width = 600
            p.plot_height = 300
            plots.style(p)
        for p in ac_plot:
            p.plot_width = 300
            p.plot_height = 300
            plots.style(p)
        diagnostics_plots = [tr_plot[0], tr_plot[1], *ac_plot]

    return diagnostics_plots


def plot_complete_pooling_model(
    df: pd.DataFrame,
    samples: MonteCarloSamples,
    query: RVIdentifier,
) -> Figure:
    """
    Complete-pooling model plot.

    :param df: Dataframe of model data.
    :type df: pd.DataFrame
    :param samples: Bean Machine inference object.
    :type samples: MonteCarloSamples
    :param query: Bean Machine query object.
    :type query: RVIdentifier
    :return: Bokeh figure of the model.
    :rtype: Figure
    """
    # Calculate the HDIs for the complete-pooling model.
    data = {"φ": samples.get(query).numpy()}
    hdi_df = az.hdi(data, hdi_prob=0.89).to_dataframe()
    hdi_df = hdi_df.T.rename(columns={"lower": "hdi_11%", "higher": "hdi_89%"})

    # Calculate the summary statistics for the complete-pooling model.
    summary_df = az.summary(data, round_to=6).join(hdi_df)

    # Calculate empirical values.
    population_mean = (df["Current hits"] / df["Current at-bats"]).mean()
    population_std = (df["Current hits"] / df["Current at-bats"]).std()
    x = (df["Current hits"] / df["Current at-bats"]).values
    posterior_upper_hdi = np.array(summary_df["hdi_89%"].tolist() * df.shape[0])
    posterior_lower_hdi = np.array(summary_df["hdi_11%"].tolist() * df.shape[0])

    # Create the figure data source.
    source = ColumnDataSource(
        {
            "x": x,
            "y": summary_df["mean"].tolist() * df.shape[0],
            "upper_hdi": posterior_upper_hdi,
            "lower_hdi": posterior_lower_hdi,
            "lower_std": [population_mean - population_std] * df.shape[0],
            "upper_std": [population_mean + population_std] * df.shape[0],
            "name": df["Name"].values,
        }
    )

    # Create the figure.
    plot = figure(
        plot_width=500,
        plot_height=500,
        title="Complete-pooling",
        x_axis_label="Observed hits / at-bats",
        y_axis_label="Predicted chance of a hit",
        y_range=[0.05, 0.55],
        x_range=[0.14, 0.41],
    )

    # Create the mean chance for at-bat hits line.
    plot.line(
        x=[0, 1],
        y=[population_mean, population_mean],
        line_color="orange",
        line_width=3,
        level="underlay",
        legend_label="Population mean",
    )

    # Create a band that contains the standard deviation of the mean chance for
    # at-bat hits.
    std_band = Band(
        base="x",
        lower="lower_std",
        upper="upper_std",
        source=source,
        level="underlay",
        fill_alpha=0.2,
        fill_color="orange",
        line_width=0.2,
        line_color="orange",
    )
    plot.add_layout(std_band)

    # Create the HDI interval whiskers for each player.
    whiskers = Whisker(
        base="x",
        upper="upper_hdi",
        lower="lower_hdi",
        source=source,
        line_color="steelblue",
    )
    whiskers.upper_head.line_color = "steelblue"
    whiskers.lower_head.line_color = "steelblue"
    plot.add_layout(whiskers)

    # Create the player's at-bat hit chance for the complete-pooling model.
    glyph = plot.circle(
        x="x",
        y="y",
        source=source,
        size=10,
        line_color="white",
        fill_color="steelblue",
        legend_label="Players",
    )
    tooltips = HoverTool(
        renderers=[glyph],
        tooltips=[
            ("Name", "@name"),
            ("Posterior Upper HDI", "@upper_hdi{0.000}"),
            ("Posterior Mean", "@y{0.000}"),
            ("Posterior Lower HDI", "@lower_hdi{0.000}"),
        ],
    )
    plot.add_tools(tooltips)

    # Add a legend to the figure.
    plot.legend.location = "top_left"
    plot.legend.click_policy = "mute"

    # Style the figure.
    plots.style(plot)

    return plot


def plot_no_pooling_diagnostics(
    samples: MonteCarloSamples,
    query: RVIdentifier,
    df: pd.DataFrame,
) -> List[List[Figure]]:
    """
    Plot the no-pooling model diagnostics.

    :param samples: Bean Machine inference object.
    :type samples: MonteCarloSamples
    :param query: Bean Machine query object.
    :type query: RVIdentifier
    :param df: Dataframe of model data.
    :type df: pd.DataFrame
    :return: A nested list of Bokeh figures.
    :rtype: List[List[Figure]]
    """
    # Prepare data for the figures.
    names = df["Name"].values
    keys = [f"θ[{name}]" for name in names]
    values = np.dsplit(samples.get(query).numpy(), 18)
    values = [value.reshape(4, 5000) for value in values]
    diagnostics_data = dict(zip(keys, values))

    # Cycle through each query and create the diagnostics plots using arviz.
    diag_plots = []
    for key, value in diagnostics_data.items():
        ac_plot = az.plot_autocorr({key: value}, show=False)[0].tolist()
        tr_plot = az.plot_trace(
            {key: value},
            kind="rank_bars",
            show=False,
        )[0].tolist()
        ess = az.plot_ess({key: value}, kind="evolution", show=False)[0][0]
        post = az.plot_posterior({key: value}, show=False)[0][0]

        # Style the plots from arviz.
        for i, p in enumerate(tr_plot):
            if i == 0:
                p.plot_width = 300
                for j, renderer in enumerate(p.renderers):
                    renderer._property_values["glyph"].line_color = COLORS[j]
                    renderer._property_values["glyph"].line_dash = "solid"
                    renderer._property_values["glyph"].line_width = 2
                    renderer._property_values["glyph"].line_alpha = 0.6
                p.x_range.start = 0
                p.x_range.end = 1
            else:
                p.plot_width = 600
            p.plot_height = 300
            plots.style(p)
        for p in ac_plot:
            p.plot_width = 300
            p.plot_height = 300
            plots.style(p)
        ess.plot_width = 300
        ess.plot_height = 300
        plots.style(ess)
        post.plot_width = 300
        post.plot_height = 300
        plots.style(post)
        post.x_range.start = 0
        post.x_range.end = 1
        diag_plots.append([post, ess, tr_plot[0], tr_plot[1], *ac_plot])

    return diag_plots


def plot_pareto_prior() -> Figure:
    """
    Plot a family of Pareto distributions.

    :return: Bokeh figure with prior distributions.
    :rtype: Figure
    """
    plot = figure(
        plot_width=400,
        plot_height=400,
        title="Pareto distribution",
        x_axis_label="Support",
        x_range=[1, 3],
    )
    colors = ["steelblue", "orange", "brown", "magenta"]

    x = np.linspace(1, 3, 1000)
    for i, alpha in enumerate(np.linspace(start=1.5, stop=3, num=4)):
        pareto_samples = pareto.pdf(x, alpha)

        pareto_source = ColumnDataSource({"x": x, "y": pareto_samples})
        plot.line(
            x="x",
            y="y",
            source=pareto_source,
            line_color=colors[i],
            line_alpha=0.7,
            line_width=2,
            legend_label=f"α = {alpha}",
        )

    plots.style(plot)
    plot.yaxis.major_tick_line_color = None
    plot.yaxis.minor_tick_line_color = None
    plot.yaxis.major_label_text_font_size = "0pt"

    return plot


def plot_no_pooling_model(
    samples: MonteCarloSamples,
    query: RVIdentifier,
    df: pd.DataFrame,
) -> Figure:
    """
    Plot the no-pooling model.

    :param samples: Bean Machine inference object.
    :type samples: MonteCarloSamples
    :param query: Bean Machine query object.
    :type query: RVIdentifier
    :param df: Pandas dataframe with model data.
    :type df: pd.DataFrame
    :return: Bokeh figure of the model.
    :rtype: Figure
    """
    names = df["Name"].values
    keys = [f"θ[{name}]" for name in names]
    values = np.dsplit(samples.get(query).numpy(), 18)
    values = [value.reshape(4, 5000) for value in values]
    data = dict(zip(keys, values))

    hdi_df = az.hdi(data, hdi_prob=0.89).to_dataframe()
    hdi_df = hdi_df.T.rename(columns={"lower": "hdi_11%", "higher": "hdi_89%"})
    summary_df = az.summary(data, round_to=4).join(hdi_df)

    x = (df["Current hits"] / df["Current at-bats"]).values
    posterior_upper_hdi = summary_df["hdi_89%"]
    posterior_lower_hdi = summary_df["hdi_11%"]
    population_mean = (df["Current hits"] / df["Current at-bats"]).mean()

    # Create the source of data for the figure.
    source = ColumnDataSource(
        {
            "x": x,
            "y": summary_df["mean"].values,
            "upper_hdi": posterior_upper_hdi,
            "lower_hdi": posterior_lower_hdi,
            "name": df["Name"].values,
        }
    )

    # Create the figure.
    plot = figure(
        plot_width=500,
        plot_height=500,
        title="No-pooling",
        x_axis_label="Observed hits / at-bats",
        y_axis_label="Predicted chance of a hit",
        x_range=[0.14, 0.41],
        y_range=[0.05, 0.55],
    )

    # Add the mean at-bat hit chance to the figure.
    plot.line(
        x=[0, 1],
        y=[population_mean, population_mean],
        line_color="orange",
        line_width=3,
        level="underlay",
        legend_label="Population mean",
    )

    # Add the standard deviation of the current mean at-bat hit chance to the
    # figure.
    std_band = Band(
        base="x",
        lower="lower_std",
        upper="upper_std",
        source=source,
        level="underlay",
        fill_alpha=0.2,
        fill_color="orange",
        line_width=0.2,
        line_color="orange",
    )
    plot.add_layout(std_band)

    # Add the empirical current at-bat hits to the figure.
    plot.line(
        x=x,
        y=(df["Current hits"] / df["Current at-bats"]).values,
        line_color="grey",
        line_alpha=0.7,
        line_width=2.0,
        legend_label="Current hits / Current at-bats",
    )

    # Add HDI whiskers to each player in the figure.
    whiskers = Whisker(
        base="x",
        upper="upper_hdi",
        lower="lower_hdi",
        source=source,
        line_color="steelblue",
    )
    whiskers.upper_head.line_color = "steelblue"
    whiskers.lower_head.line_color = "steelblue"
    plot.add_layout(whiskers)

    # Add the modeled player at-bat hit chance to the figure.
    glyph = plot.circle(
        x="x",
        y="y",
        source=source,
        size=10,
        line_color="white",
        fill_color="steelblue",
        legend_label="Players",
    )
    tooltips = HoverTool(
        renderers=[glyph],
        tooltips=[
            ("Name", "@name"),
            ("Posterior Upper HDI", "@upper_hdi{0.000}"),
            ("Posterior Mean", "@y{0.000}"),
            ("Posterior Lower HDI", "@lower_hdi{0.000}"),
        ],
    )
    plot.add_tools(tooltips)

    # Add a legend to the figure.
    plot.legend.location = "top_left"
    plot.legend.click_policy = "mute"

    # Style the figure.
    plots.style(plot)

    return plot


def _sample_data_prep(
    samples: MonteCarloSamples,
    df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """
    Prepare sample data for plotting.

    :param samples: Bean Machine inference object.
    :type samples: MonteCarloSamples
    :param df: Dataframe of the model data.
    :type df: pd.DataFrame
    :return: Dictionary of data for plotting.
    :rtype: Dict[str, np.ndarray]
    """
    keys = []
    values = []
    samples_xr = samples.to_xarray()
    n_chains = samples_xr.coords.get("chain").values.shape[0]
    n_samples = samples_xr.coords.get("draw").values.shape[0]
    data_vars = samples_xr.data_vars
    for key in data_vars.keys():
        name = key.__dict__["wrapper"].__name__
        if "theta" in name:
            v = np.dsplit(samples.get(key).numpy(), df.shape[0])
            v = [value_.reshape(n_chains, n_samples) for value_ in v]
            values.extend(v)
            k = [f"θ[{player_name}]" for player_name in df["Name"].values]
            keys.extend(k)
        if "kappa" in name:
            keys.append("κ")
            v = samples.get(key).numpy()
            values.append(v)
        if "phi" in name:
            keys.append("φ")
            v = samples.get(key).numpy()
            values.append(v)
    return dict(zip(keys, values))


def plot_partial_pooling_diagnostics(
    samples: MonteCarloSamples,
    df: pd.DataFrame,
) -> List[List[Figure]]:
    """
    Plot the partial-pooling model diagnostics.

    :param samples: Bean Machine inference object.
    :type samples: MonteCarloSamples
    :param df: Dataframe of model data.
    :type df: pd.DataFrame
    :return: A nested list of Bokeh figures.
    :rtype: List[List[Figure]]
    """
    # Prepare data for the figures.
    diagnostics_data = _sample_data_prep(samples, df)

    # Cycle through each query and create the diagnostics plots using arviz.
    diagnostic_plots = []
    for key, value in diagnostics_data.items():
        ac_plot = az.plot_autocorr({key: value}, show=False)[0].tolist()
        tr_plot = az.plot_trace(
            {key: value},
            kind="rank_bars",
            show=False,
        )[0].tolist()
        ess = az.plot_ess({key: value}, kind="evolution", show=False)[0][0]
        post = az.plot_posterior({key: value}, show=False)[0][0]

        # Style the plots from arviz.
        for i, p in enumerate(tr_plot):
            if i == 0:
                p.plot_width = 300
                for j, renderer in enumerate(p.renderers):
                    renderer._property_values["glyph"].line_color = COLORS[j]
                    renderer._property_values["glyph"].line_dash = "solid"
                    renderer._property_values["glyph"].line_width = 2
                    renderer._property_values["glyph"].line_alpha = 0.6
            else:
                p.plot_width = 600
            p.plot_height = 300
            plots.style(p)
        for p in ac_plot:
            p.plot_width = 300
            p.plot_height = 300
            plots.style(p)
        ess.plot_width = 300
        ess.plot_height = 300
        plots.style(ess)
        post.plot_width = 300
        post.plot_height = 300
        plots.style(post)
        diagnostic_plots.append([post, ess, tr_plot[0], tr_plot[1], *ac_plot])

    return diagnostic_plots


def plot_partial_pooling_model(samples: MonteCarloSamples, df: pd.DataFrame) -> Figure:
    """
    Partial-pooling model plot.

    :param samples: Bean Machine inference object.
    :type samples: MonteCarloSamples
    :param df: Dataframe of the model data.
    :type df: pd.DataFrame
    :return: Bokeh figure of the partial-pooling model.
    :rtype: Figure
    """
    # Prepare data for the figure.
    diagnostics_data = _sample_data_prep(samples, df)
    hdi_df = az.hdi(diagnostics_data, hdi_prob=0.89).to_dataframe()
    hdi_df = hdi_df.T.rename(columns={"lower": "hdi_11%", "higher": "hdi_89%"})
    summary_df = az.summary(diagnostics_data, round_to=4).join(hdi_df)
    theta_index = summary_df[
        summary_df.index.astype(str).str.contains("θ")
    ].index.values
    x = (df["Current hits"] / df["Current at-bats"]).values
    y = summary_df.loc[theta_index, "mean"]
    upper_hdi = summary_df.loc[theta_index, "hdi_89%"]
    lower_hdi = summary_df.loc[theta_index, "hdi_11%"]
    population_mean = (df["Current hits"] / df["Current at-bats"]).mean()

    # Create the figure data source.
    source = ColumnDataSource(
        {
            "x": x,
            "y": y,
            "upper_hdi": upper_hdi,
            "lower_hdi": lower_hdi,
            "name": df["Name"].values,
        }
    )

    # Create the figure.
    plot = figure(
        plot_width=500,
        plot_height=500,
        title="Partial pooling",
        x_axis_label="Observed hits / at-bats",
        y_axis_label="Predicted chance of a hit",
        x_range=[0.14, 0.41],
        y_range=[0.05, 0.55],
    )

    # Add the empirical mean at-bat hit chance to the figure.
    plot.line(
        x=[0, 1],
        y=[population_mean, population_mean],
        line_color="orange",
        line_width=3,
        level="underlay",
        legend_label="Population mean",
    )

    # Add the standard deviation of the mean at-bat hit chance to the figure.
    std_band = Band(
        base="x",
        lower="lower_std",
        upper="upper_std",
        source=source,
        level="underlay",
        fill_alpha=0.2,
        fill_color="orange",
        line_width=0.2,
        line_color="orange",
    )
    plot.add_layout(std_band)

    # Add the empirical at-bat hit chance to the figure.
    plot.line(
        x=x,
        y=(df["Current hits"] / df["Current at-bats"]).values,
        line_color="grey",
        line_alpha=0.7,
        line_width=2.0,
        legend_label="Current hits / Current at-bats",
    )

    # Add the HDI whiskers to the figure.
    whiskers = Whisker(
        base="x",
        upper="upper_hdi",
        lower="lower_hdi",
        source=source,
        line_color="steelblue",
    )
    whiskers.upper_head.line_color = "steelblue"
    whiskers.lower_head.line_color = "steelblue"
    plot.add_layout(whiskers)

    # Add the partial-pooling model data to the figure.
    glyph = plot.circle(
        x="x",
        y="y",
        source=source,
        size=10,
        line_color="white",
        fill_color="steelblue",
        legend_label="Players",
    )
    tooltips = HoverTool(
        renderers=[glyph],
        tooltips=[
            ("Name", "@name"),
            ("Posterior Upper HDI", "@upper_hdi{0.000}"),
            ("Posterior Mode", "@mode{0.000}"),
            ("Posterior Lower HDI", "@lower_hdi{0.000}"),
        ],
    )
    plot.add_tools(tooltips)

    # Add a legend to the figure.
    plot.legend.location = "top_left"
    plot.legend.click_policy = "mute"

    # Style the figure.
    plots.style(plot)

    return plot


def plot_shrinkage(
    no_pooling_samples: MonteCarloSamples,
    partial_pooling_samples: MonteCarloSamples,
    df: pd.DataFrame,
) -> Figure:
    """
    Plot shrinkage due to partial-pooling model.

    :param no_pooling_samples: Bean Machine inference object for no-pooling.
    :type no_pooling_samples: MonteCarloSamples
    :param partial_pooling_samples: BM inference object for partial-pooling.
    :type partial_pooling_samples: MonteCarloSamples
    :param df: Dataframe with model data in it.
    :type df: pd.DataFrame
    :return: Bokeh plot showing shrinkage.
    :rtype: Figure
    """
    # Prepare data for the figure.
    population_mean = (df["Current hits"] / df["Current at-bats"]).mean()
    population_std = (df["Current hits"] / df["Current at-bats"]).std()
    lower_std = [population_mean - population_std] * df.shape[0]
    upper_std = [population_mean + population_std] * df.shape[0]
    x = (df["Current hits"] / df["Current at-bats"]).values
    names = df["Name"].values

    pp_data = _sample_data_prep(partial_pooling_samples, df)
    pp_summary_df = az.summary(pp_data, round_to=4)
    pp_theta_index = pp_summary_df.index.astype(str).str.contains("θ")
    pp_y = pp_summary_df.loc[pp_theta_index, "mean"].values
    pp_source = ColumnDataSource({"x": x, "y": pp_y, "name": names})

    no_pooling_data = _sample_data_prep(no_pooling_samples, df)
    np_summary_df = az.summary(no_pooling_data, round_to=4)
    np_theta_index = np_summary_df.index.astype(str).str.contains("θ")
    np_y = np_summary_df.loc[np_theta_index, "mean"].values
    np_source = ColumnDataSource({"x": x, "y": np_y, "name": names})

    # Create the figure.
    plot = figure(
        plot_width=500,
        plot_height=500,
        title="Partial pooling shift",
        x_axis_label="Observed hits / at-bats",
        y_axis_label="Predicted chance of a hit",
        x_range=[0.14, 0.41],
        y_range=[0.05, 0.55],
    )

    # Create the mean chance for at-bat hits line.
    plot.line(
        x=[0, 1],
        y=[population_mean, population_mean],
        line_color="orange",
        line_width=3,
        level="underlay",
        legend_label="Population mean",
    )

    # Create a band that contains the standard deviation of the mean chance for
    # at-bat hits.
    source = ColumnDataSource(
        {
            "x": x,
            "lower_std": lower_std,
            "upper_std": upper_std,
        }
    )
    std_band = Band(
        base="x",
        lower="lower_std",
        upper="upper_std",
        source=source,
        level="underlay",
        fill_alpha=0.2,
        fill_color="orange",
        line_width=0.2,
        line_color="orange",
    )
    plot.add_layout(std_band)

    # Add the empirical current chances to the figure.
    plot.line(
        x=x,
        y=x,
        line_color="grey",
        line_alpha=0.7,
        line_width=2.0,
        legend_label="Current hits / Current at-bats",
    )

    # Add the partial-pooling model chances.
    plot.circle(
        x="x",
        y="y",
        source=pp_source,
        size=10,
        line_color="white",
        fill_color="steelblue",
        legend_label="Partial-pooling",
    )

    # Add the no-pooling model chances.
    plot.circle(
        x="x",
        y="y",
        source=np_source,
        size=10,
        line_color="steelblue",
        fill_color="white",
        legend_label="No-pooling",
    )

    # Add arrows to show the shrinkage.
    for i in range(len(x)):
        plot.add_layout(
            Arrow(
                end=VeeHead(size=10),
                x_start=np_source.data["x"][i],
                y_start=np_source.data["y"][i],
                x_end=pp_source.data["x"][i],
                y_end=pp_source.data["y"][i],
            )
        )

    # Add a legend to the figure.
    plot.legend.location = "top_left"
    plot.legend.click_policy = "mute"

    # Style the figure.
    plots.style(plot)

    return plot
