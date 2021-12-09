# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Module for the ``Mixture model using count data`` tutorial."""
from numbers import Number
from typing import Any, Dict, List, Union

import arviz as az
import numpy as np
import pandas as pd
import torch
import torch.distributions as dist
from beanmachine.ppl import RVIdentifier
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from bokeh.models import Band, ColumnDataSource, HoverTool
from bokeh.plotting import figure
from bokeh.plotting.figure import Figure
from torch import Tensor


COLORS = ["#2a2eec", "#fa7c17", "#328c06", "#c10c90"]


def plot_value_counts(df: pd.DataFrame) -> List[List[Figure]]:
    """
    Plot the pre-drug and post-drug value counts as a bar plot.

    :param df: Pandas dataframe object of the model data.
    :type df: pd.DataFrame
    :return: A list of figures to display in the notebook.
    :rtype: List[List[Figure]]
    """
    predrug_data = df["predrug"].value_counts().sort_index()
    predrug_index = predrug_data.index.values
    postdrug_data = df["postdrug"].value_counts().sort_index()
    postdrug_index = postdrug_data.index.values
    PADDING = 2
    x = np.arange(0, max(predrug_data.index.max(), postdrug_data.index.max()) + PADDING)
    top_predrug = np.zeros(len(x))
    top_predrug[predrug_index] = predrug_data
    top_postdrug = np.zeros(len(x))
    top_postdrug[postdrug_index] = postdrug_data
    OFFSET = 0.5
    left = x - OFFSET
    right = x + OFFSET
    bottom = np.zeros(len(x))

    figs = []
    for i, column in enumerate(["predrug", "postdrug"]):
        # Create the figure.
        p = figure(
            plot_width=700,
            plot_height=300,
            y_axis_label="Counts",
            x_axis_label="PVC events",
            title=f'PVC events "{column}"',
            y_range=[0, 8],
            x_range=[-2, 52],
            outline_line_color="black",
        )

        # Prepare data for the figure.
        source_data = {
            "x": x,
            "left": left,
            "top": None,
            "right": right,
            "bottom": bottom,
        }
        if i == 0:
            source_data["top"] = top_predrug
        else:
            source_data["top"] = top_postdrug
        source = ColumnDataSource(source_data)

        # Add data to the figure.
        glyph = p.quad(
            left="left",
            top="top",
            right="right",
            bottom="bottom",
            source=source,
            fill_color="steelblue",
            line_color="white",
            fill_alpha=0.7,
            hover_fill_color="orange",
            hover_line_color="black",
            hover_alpha=1,
        )

        # Add tooltips to the figure.
        tips = HoverTool(
            renderers=[glyph],
            tooltips=[("Count", "@top"), ("PVC events", "@x")],
        )
        p.add_tools(tips)

        # Style the figure
        p.grid.grid_line_alpha = 0.2
        p.grid.grid_line_color = "gray"
        p.grid.grid_line_width = 0.3
        p.yaxis.minor_tick_line_color = None
        figs.append(p)

    return [[figs[0]], [figs[1]]]


class SimulatedPredictiveChecks:
    """Simulated predictive checks base class."""

    def __init__(
        self,
        data: pd.DataFrame,
        samples_with_observations: Union[None, MonteCarloSamples] = None,
        samples_without_observations: Union[None, MonteCarloSamples] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Plot prior/posterior predictive checks.

        :param data: Pandas dataframe object of the model data.
        :type data: pd.DataFrame
        :param samples_with_observations: Bean Machine inference object.
        :type samples_with_observations: MonteCarloSamples
        :param samples_without_observations: Bean Machine inference object.
        :type samples_without_observations: MonteCarloSamples
        """
        self.data = data
        self.n_records = self.data.shape[0]

        if samples_with_observations is None:
            if samples_without_observations is None:
                msg = (
                    'Either one of "samples_with_observations" or '
                    '"samples_without_observations" must be supplied. '
                )
                raise TypeError(msg)

        if samples_with_observations is not None:
            self.samples_with_observations_xr = samples_with_observations.to_xarray()
            coords = self.samples_with_observations_xr.coords
            self.n_chains_with_observations = coords.get("chain").values.shape[0]
            self.n_samples_with_observations = coords.get("draw").values.shape[0]
            self.queries_with_observations = self.samples_with_observations_xr.data_vars
        else:
            self.samples_with_observations_xr = None
            self.n_chains_with_observations = None
            self.n_samples_with_observations = None
            self.queries_with_observations = None

        if samples_without_observations is not None:
            self.samples_without_observations_xr = (
                samples_without_observations.to_xarray()
            )
            coords = self.samples_without_observations_xr.coords
            self.n_chains_without_observations = coords.get("chain").values.shape[0]
            self.n_samples_without_observations = coords.get("draw").values.shape[0]
            self.queries_without_observations = (
                self.samples_without_observations_xr.data_vars
            )
        else:
            self.samples_without_observations_xr = None
            self.n_chains_without_observations = None
            self.n_samples_without_observations = None
            self.queries_without_observations = None

    def _simulate_data(self, *args, **kwargs) -> Any:
        msg = "To be implemented by the inheriting class."
        raise NotImplementedError(msg)

    def simulate_data(self, *args, **kwargs) -> Any:
        """Simulate data for the predictive plots."""
        return self._simulate_data(*args, **kwargs)

    def _generate_plot_data(
        self,
        simulated_data: List[Tensor],
    ) -> Dict[str, List[Number]]:
        xs = []
        ys = []
        lower_hdis = []
        upper_hdis = []
        mins = []
        maxs = []
        histograms = []
        bins = []
        for i in range(self.n_records):
            n_bins = int(simulated_data[i].max()) + 1
            hist, bins_ = np.histogram(simulated_data[i], bins=n_bins)
            histograms.append(hist.tolist())
            bins.append(bins_.tolist())
            simulation_mean = simulated_data[i].mean().item()
            xs.append(simulation_mean)
            ys.append(i + 1)
            lower_hdi, upper_hdi = az.hdi(simulated_data[i].numpy(), hdi_prob=0.89)
            lower_hdis.append(int(lower_hdi))
            upper_hdis.append(int(upper_hdi))
            mins.append(int(simulated_data[i].min().item()))
            maxs.append(int(simulated_data[i].max().item()))

        return {
            "x": xs,
            "y": ys,
            "lower_hdi": lower_hdis,
            "upper_hdi": upper_hdis,
            "minimum": mins,
            "maximum": maxs,
            "histogram": histograms,
            "bins": bins,
        }

    def _model(self, *args, **kwargs) -> Any:
        msg = "To be implemented by the inheriting class."
        raise NotImplementedError(msg)

    def model(self, *args, **kwargs) -> Any:
        """Model definition for data simulation."""
        return self._model(*args, **kwargs)

    def _plot_prior_predictive_checks(self) -> Figure:
        msg = "To be implemented by the inheriting class."
        raise NotImplementedError(msg)

    def plot_prior_predictive_checks(self) -> Figure:
        """Plot the prior predictive checks."""
        return self._plot_prior_predictive_checks()

    def _plot_posterior_predictive_checks(self) -> Figure:
        msg = "To be implemented by the inheriting class."
        raise NotImplementedError(msg)

    def plot_posterior_predictive_checks(self) -> Figure:
        """Plot the posterior predictive checks."""
        return self._plot_posterior_predictive_checks()


class PlotMixin:
    """Mixin for plotting."""

    MODEL_NAME = None

    def plot_predictive_checks(
        self,
        simulated_data: Dict[str, List[Number]],
        title: str,
    ) -> Figure:
        """
        Plot of the predictive check.

        :param simulated_data: Object containing the simulated data to plot.
        :type simulated_data: Dict[str, List[Number]]
        :param title: String to differentiate between prior or predictive plots.
        :type title: str
        :returns: Bokeh plot of the predictive check.
        :rtype: Figure
        """
        p = figure(
            title=f"{title} distributions vs. observations",
            outline_line_color="black",
            y_axis_label="Patient ID",
            x_axis_label="PVC events",
            toolbar_location=None,
            x_range=[-2, 52],
            y_range=[0.5, 12.99],
        )
        for i in range(self.n_records):
            # Prepare data.
            SCALE = 0.8
            lower_hdis = np.array(simulated_data["lower_hdi"])
            upper_hdis = np.array(simulated_data["upper_hdi"])
            bins = np.array(simulated_data["bins"][i])
            histogram = np.array(simulated_data["histogram"][i])
            histogram = SCALE * (histogram / histogram.max())
            bin_pairs = list(zip(bins[:-1], bins[1:]))
            step_x = [bin_pairs[0][0]]
            step_y = [i + 1]
            for j, bin_pair in enumerate(bin_pairs):
                step_x.append(bin_pair[0])
                step_y.append(histogram[j] + i + 1)
                step_x.append(bin_pair[1])
                step_y.append(histogram[j] + i + 1)
            step_x.append(bin_pairs[-1][1])
            step_y.append(i + 1)

            # Histogram
            hist_source = ColumnDataSource({"x": step_x, "y": step_y})
            p.step(  # Top portion of the histogram.
                x="x",
                y="y",
                source=hist_source,
                line_color="steelblue",
                line_width=1,
                line_alpha=0.7,
            )
            p.line(  # Lower portion of the histogram.
                x=[step_x[0], step_x[-1]],
                y=[step_y[0], step_y[-1]],
                line_color="steelblue",
                line_width=0.5,
                line_alpha=0.7,
            )
            band_source = ColumnDataSource(  # Fill for the histogram.
                {
                    "base": step_x,
                    "lower": np.linspace(step_y[0], step_y[-1], num=len(step_y)),
                    "upper": step_y,
                }
            )
            band = Band(
                base="base",
                lower="lower",
                upper="upper",
                source=band_source,
                fill_color="steelblue",
                fill_alpha=0.1,
                level="underlay",
            )
            p.add_layout(band)
            p.varea(  # Legend label for the histogram.
                x="base",
                y1="lower",
                y2="upper",
                source=band_source,
                fill_color="steelblue",
                fill_alpha=0.1,
                legend_label="Simulated distribution",
            )
            # HDI
            p.line(
                x=[lower_hdis[i], upper_hdis[i]],
                y=[simulated_data["y"][i] - 0.025] * 2,
                line_color="steelblue",
                line_width=2,
                legend_label="Simulated 89% HDI",
            )
            # Simulated mean
            source = ColumnDataSource(
                {
                    "x": [simulated_data["x"][i]],
                    "y": [simulated_data["y"][i]],
                    "hdi": [
                        f"{simulated_data['lower_hdi'][i]}–"
                        f"{simulated_data['upper_hdi'][i]}"
                    ],
                    "minmax": [
                        f"{simulated_data['minimum'][i]}–"
                        f"{simulated_data['maximum'][i]}"
                    ],
                }
            )
            locals()[f"mean_{i}"] = p.circle(
                x="x",
                y="y",
                source=source,
                size=7,
                fill_color="white",
                line_color="steelblue",
                hover_fill_color="orange",
                hover_line_color="black",
                level="overlay",
                legend_label="Simulated mean",
            )
            locals()[f"mean_tips_{i}"] = HoverTool(
                renderers=[locals()[f"mean_{i}"]],
                tooltips=[
                    ("Simulated mean", "@x{0.00}"),
                    ("Simulated 89% HDI", "@hdi"),
                    ("Simulated min/max", "@minmax"),
                ],
            )
            p.add_tools(locals()[f"mean_tips_{i}"])
            # Observed data
            source = ColumnDataSource(
                {
                    "x": [self.observed[i]],
                    "y": [simulated_data["y"][i]],
                }
            )
            locals()[f"true_{i}"] = p.square(
                x="x",
                y="y",
                size=10,
                source=source,
                fill_color="magenta",
                line_color="white",
                hover_fill_color="green",
                legend_label="Postdrug PVC counts",
            )
            locals()[f"true_tips_{i}"] = HoverTool(
                renderers=[locals()[f"true_{i}"]],
                tooltips=[
                    ("Patient ID", "@y"),
                    ("Postdrug PVC counts", "@x"),
                ],
            )
            p.add_tools(locals()[f"true_tips_{i}"])
            # Total data
            source = ColumnDataSource(
                {
                    "x": [self.t[i]],
                    "y": [simulated_data["y"][i]],
                }
            )
            locals()[f"true_{i}"] = p.diamond(
                x="x",
                y="y",
                size=15,
                source=source,
                fill_color="brown",
                line_color="white",
                hover_fill_color="black",
                legend_label="Predrug + Postdrug PVC counts",
            )
            locals()[f"true_tips_{i}"] = HoverTool(
                renderers=[locals()[f"true_{i}"]],
                tooltips=[
                    ("Patient ID", "@y"),
                    ("Predrug + Postdrug PVC counts", "@x"),
                ],
            )
            p.add_tools(locals()[f"true_tips_{i}"])

        p.grid.grid_line_alpha = 0.2
        p.grid.grid_line_color = "gray"
        p.grid.grid_line_width = 0.3
        p.yaxis.minor_tick_line_color = None
        p.yaxis[0].ticker.desired_num_ticks = len(self.observed)
        p.legend.location = "bottom_right"

        return p

    def prior_predictive_plot(self) -> Figure:
        """Prior predictive plot mixin manager."""
        simulated_data = self._simulate_data(using="prior")
        plot_data = self._generate_plot_data(simulated_data)

        return self.plot_predictive_checks(plot_data, f"{self.MODEL_NAME} prior")

    def posterior_predictive_plot(self) -> Figure:
        """Posterior predictive plot mixin manager."""
        simulated_data = self._simulate_data(using="posterior")
        plot_data = self._generate_plot_data(simulated_data)

        return self.plot_predictive_checks(plot_data, f"{self.MODEL_NAME} posterior")


class Model1PredictiveChecks(SimulatedPredictiveChecks, PlotMixin):
    """Model 1 predictive checks."""

    MODEL_NAME = "Model 1"

    def __init__(self, p_query: RVIdentifier, *args, **kwargs) -> None:
        """
        Model 1 predictive check initialization.

        :param p_query: Bean Machine query object.
        :type p_query: RVIdentifier
        :returns: None
        """
        super().__init__(*args, **kwargs)
        self.t = self.data["total"].astype(int).values
        self.observed = self.data["postdrug"].astype(int).values
        if self.samples_with_observations_xr is not None:
            data_vars = self.samples_with_observations_xr.data_vars
            p_with_observations = data_vars.get(p_query())
            p_with_observations = p_with_observations.values.flatten()
            self.p_with_observations = torch.tensor(p_with_observations)
        if self.samples_without_observations_xr is not None:
            data_vars = self.samples_without_observations_xr.data_vars
            p_without_observations = data_vars.get(p_query())
            p_without_observations = p_without_observations.values.flatten()
            self.p_without_observations = torch.tensor(p_without_observations)

    def _model(self, i: int, p: Tensor) -> dist.Binomial:
        return dist.Binomial(torch.tensor(self.t[i]), p)

    def _simulate_data(self, using: str = "prior", N: int = 1) -> List[Tensor]:
        p = torch.zeros((0,))
        if using == "prior":
            p = self.p_without_observations
        elif using == "posterior":
            p = self.p_with_observations
        simulated_data = []
        for i in range(self.n_records):
            simulation = self._model(i, p).sample((N,)).flatten()
            simulated_data.append(simulation)

        return simulated_data

    def _plot_prior_predictive_checks(self) -> Figure:
        return self.prior_predictive_plot()

    def _plot_posterior_predictive_checks(self) -> Figure:
        return self.posterior_predictive_plot()


class Model2PredictiveChecks(SimulatedPredictiveChecks, PlotMixin):
    """Model 2 predictive checks."""

    MODEL_NAME = "Model 2"

    def __init__(
        self, p_query: RVIdentifier, s_query_str: str, *args, **kwargs
    ) -> None:
        """
        Model 2 predictive checks initialization.

        :param p_query: Bean Machine query object.
        :type p_query: RVIdentifier
        :param s_query_str: Start of the name of the Bean Machine query object.
        :type s_query_str: str
        :returns: None
        """
        super().__init__(*args, **kwargs)
        self.t = self.data["total"].astype(int).values
        self.observed = self.data["postdrug"].astype(int).values
        if self.samples_with_observations_xr is not None:
            data_vars = self.samples_with_observations_xr.data_vars
            p_with_observations = data_vars.get(p_query())
            p_with_observations = p_with_observations.values.flatten()
            self.p_with_observations = torch.tensor(p_with_observations)
            s_with_observations = {
                key: data_vars.get(key)
                for key, _ in self.samples_with_observations_xr.items()
                if s_query_str in str(key)
            }
            s_with_observations = dict(
                sorted(s_with_observations.items(), key=lambda item: item[0].arguments)
            )
            self.s_with_observations = list(s_with_observations.values())
        if self.samples_without_observations_xr is not None:
            data_vars = self.samples_without_observations_xr.data_vars
            p_without_observations = data_vars.get(p_query())
            p_without_observations = p_without_observations.values.flatten()
            self.p_without_observations = torch.tensor(p_without_observations)
            s_without_observations = {
                key: data_vars.get(key)
                for key, _ in self.samples_without_observations_xr.items()
                if s_query_str in str(key)
            }
            s_without_observations = dict(
                sorted(
                    s_without_observations.items(), key=lambda item: item[0].arguments
                )
            )
            self.s_without_observations = list(s_without_observations.values())

    def _model(self, i: int, p: Tensor, s: Tensor) -> dist.Binomial:
        return dist.Binomial(torch.tensor(self.t[i]), p * s)

    def _simulate_data(self, using: str = "prior", N: int = 1) -> List[Tensor]:
        p = torch.zeros((0,))
        s = torch.zeros((self.n_records, 0))
        if using == "prior":
            p = self.p_without_observations
            s = self.s_without_observations
        elif using == "posterior":
            p = self.p_with_observations
            s = self.s_with_observations
        simulated_data = []
        for i in range(self.n_records):
            simulation = self._model(i, p, s[i].values.flatten())
            simulation = simulation.sample((N,)).flatten()
            simulated_data.append(simulation)

        return simulated_data

    def _plot_prior_predictive_checks(self) -> Figure:
        return self.prior_predictive_plot()

    def _plot_posterior_predictive_checks(self) -> Figure:
        return self.posterior_predictive_plot()


class Model3PredictiveChecks(SimulatedPredictiveChecks, PlotMixin):
    """Model 3 predictive checks."""

    MODEL_NAME = "Model 3"

    def __init__(
        self,
        p_query: RVIdentifier,
        s_query_str: str,
        *args,
        **kwargs,
    ) -> None:
        """
        Model 3 predictive checks initialization.

        :param p_query: Bean Machine query object.
        :type p_query: RVIdentifier
        :param s_query_str: Start of the name of the Bean Machine query object.
        :type s_query_str: str
        """
        super().__init__(*args, **kwargs)
        self.t = self.data["total"].astype(int).values
        self.observed = self.data["postdrug"].astype(int).values
        if self.samples_with_observations_xr is not None:
            data_vars = self.samples_with_observations_xr.data_vars
            p_with_observations = data_vars.get(p_query())
            p_with_observations = p_with_observations.values.flatten()
            self.p_with_observations = torch.tensor(p_with_observations)
            s_with_observations = {
                key: torch.tensor(data_vars.get(key).values).reshape(
                    self.n_chains_with_observations * self.n_samples_with_observations,
                    2,
                )
                for key, _ in self.samples_with_observations_xr.items()
                if s_query_str in str(key)
            }
            s_with_observations = dict(
                sorted(s_with_observations.items(), key=lambda item: item[0].arguments)
            )
            self.s_with_observations = list(s_with_observations.values())
        if self.samples_without_observations_xr is not None:
            data_vars = self.samples_without_observations_xr.data_vars
            p_without_observations = data_vars.get(p_query())
            p_without_observations = p_without_observations.values.flatten()
            self.p_without_observations = torch.tensor(p_without_observations)
            s_without_observations = {
                key: torch.tensor(data_vars.get(key).values).reshape(
                    self.n_chains_without_observations
                    * self.n_samples_without_observations,
                    2,
                )
                for key, _ in self.samples_without_observations_xr.items()
                if s_query_str in str(key)
            }
            s_without_observations = dict(
                sorted(
                    s_without_observations.items(), key=lambda item: item[0].arguments
                )
            )
            self.s_without_observations = list(s_without_observations.values())

    def _model(self, i: int, p: Tensor, s: List[Tensor]) -> dist.Binomial:
        # Handle the case when both weights are zero.
        switch = s[i].clone()
        switch[(switch.sum(dim=1) == 0.0).nonzero()] = torch.tensor(
            [1.0, 1.0],
            dtype=torch.float64,
        )
        s = dist.Categorical(probs=switch).sample((1,)).flatten()
        return dist.Binomial(torch.tensor(self.t[i]), p * s)

    def _simulate_data(self, using: str = "prior", N: int = 1) -> List[Tensor]:
        p = torch.zeros((0,))
        s = [torch.zeros((self.n_records, 0))]
        if using == "prior":
            p = self.p_without_observations
            s = self.s_without_observations
        elif using == "posterior":
            p = self.p_with_observations
            s = self.s_with_observations
        simulated_data = []
        for i in range(self.n_records):
            simulation = self._model(i, p, s)
            simulation = simulation.sample((N,)).flatten()
            simulated_data.append(simulation)

        return simulated_data

    def _plot_prior_predictive_checks(self) -> Figure:
        return self.prior_predictive_plot()

    def _plot_posterior_predictive_checks(self) -> Figure:
        return self.posterior_predictive_plot()


def plot_diagnostics(
    samples: MonteCarloSamples,
    ordering: Union[None, List[str]] = None,
) -> List[Figure]:
    """
    Plot model diagnostics.

    :param samples: Bean Machine inference object.
    :type samples: MonteCarloSamples
    :param ordering: Define an ordering for how the plots are displayed.
    :type ordering: List[str]
    :return: Bokeh figures with visual diagnostics.
    :rtype: List[Figure]
    """
    # Prepare the data for the figure.
    samples_xr = samples.to_xarray()
    data = {str(key): value.values for key, value in samples_xr.data_vars.items()}

    if ordering is not None:
        diagnostics_data = {}
        for key in ordering:
            key = str(key)
            diagnostics_data[key] = data[key]
    else:
        diagnostics_data = data

    # Cycle through each query and create the diagnostics plots using arviz.
    diagnostics_plots = []
    for key, value in diagnostics_data.items():
        ac_plot = az.plot_autocorr({key: value}, show=False)[0].tolist()
        tr_plot = az.plot_trace(
            {key: value},
            kind="rank_bars",
            show=False,
        )[0].tolist()
        posterior_plot = az.plot_posterior({key: value}, show=False)[0][0]
        posterior_plot.plot_width = 300
        posterior_plot.plot_height = 300
        posterior_plot.grid.grid_line_alpha = 0.2
        posterior_plot.grid.grid_line_color = "gray"
        posterior_plot.grid.grid_line_width = 0.3
        posterior_plot.yaxis.minor_tick_line_color = None
        posterior_plot.outline_line_color = "black"
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
            p.grid.grid_line_alpha = 0.2
            p.grid.grid_line_color = "gray"
            p.grid.grid_line_width = 0.3
            p.yaxis.minor_tick_line_color = None
            p.outline_line_color = "black"
        for p in ac_plot:
            p.plot_width = 300
            p.plot_height = 300
            p.grid.grid_line_alpha = 0.2
            p.grid.grid_line_color = "gray"
            p.grid.grid_line_width = 0.3
            p.yaxis.minor_tick_line_color = None
            p.outline_line_color = "black"
        ps = [posterior_plot, tr_plot[0], tr_plot[1], *ac_plot]
        diagnostics_plots.append(ps)

    return diagnostics_plots
