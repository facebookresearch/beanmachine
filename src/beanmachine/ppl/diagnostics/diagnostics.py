# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import math
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from plotly.subplots import make_subplots
from torch import Tensor

from . import common_plots, common_statistics as common_stats


class BaseDiagnostics:
    def __init__(self, samples: MonteCarloSamples):
        self.samples = samples
        self.statistics_dict = {}
        self.plots_dict = {}

    def _prepare_query_list(
        self, query_list: Optional[List[RVIdentifier]] = None
    ) -> List[RVIdentifier]:
        if query_list is None:
            return list(self.samples.keys())
        for query in query_list:
            if not (query in self.samples):
                raise ValueError(f"query {self._stringify_query(query)} does not exist")
        return query_list

    def summaryfn(self, func: Callable, display_names: List[str]) -> Callable:
        """
        this function keeps a directory of all summary-related functions,
        so it could handle the overridden functions and new ones that user defines

        :param func: method which is going to be executed when summary() is called.
        :param display_name: the name appears in the summary() output dataframe
        :returns: user-visible function that can be called over a list of queries
        """
        statistics_name = func.__name__
        self.statistics_dict[statistics_name] = (func, display_names)
        return self._standalone_summary_stat_function(statistics_name, func)

    def _prepare_summary_stat_input(
        self, query: RVIdentifier, chain: Optional[int] = None
    ):
        query_samples = self.samples[query]
        if query_samples.shape[0] != 1:
            # squeeze out non-chain singleton dims
            query_samples = query_samples.squeeze()
        if chain is not None:
            query_samples = query_samples[chain].unsqueeze(0)
        return query_samples

    def _create_table(
        self, query: RVIdentifier, results: List[Tensor], func_list: List[str]
    ) -> pd.DataFrame:
        """
        this function turns output of each summary stat function to a dataframe
        """
        out_pd = pd.DataFrame()
        if len(results) > 0:
            single_result_set = results[0]
            if single_result_set is not None and len(single_result_set) > 0:
                for flattened_index in range(single_result_set[0].numel()):
                    index = np.unravel_index(
                        flattened_index, tuple(single_result_set[0].size())
                    )
                    row_data = []
                    rowname = f"{self._stringify_query(query)}{list(index)}"

                    for result in results:
                        num_of_sets = result.size()[0]
                        for set_num in range(num_of_sets):
                            row_data.append(result[set_num][index].item())
                    cur = pd.DataFrame([row_data], columns=func_list, index=[rowname])
                    if out_pd.empty:
                        out_pd = cur
                    else:
                        out_pd = pd.concat([out_pd, cur])
        return out_pd

    def _stringify_query(self, query: RVIdentifier) -> str:
        return f"{query.function.__name__}{query.arguments}"

    def _execute_summary_stat_funcs(
        self,
        query: RVIdentifier,
        func_dict: Dict[str, Tuple[Callable, str]],
        chain: Optional[int] = None,
        raise_warning: bool = False,
    ):
        frames = pd.DataFrame()
        query_results = []
        func_list = []
        queried_samples = self._prepare_summary_stat_input(query, chain)
        for _k, (func, display_names) in func_dict.items():
            result = func(queried_samples)
            if result is None:
                # in the case of r hat and other algorithms, they may return None
                # if the samples do not have enough chains or have the wrong shape
                if raise_warning:
                    warnings.warn(
                        f"{display_names} cannot be calculated for the provided samples"
                    )
                continue
            # the first dimension is equivalant to the size of the display_names
            if len(display_names) <= 1:
                result = result.unsqueeze(0)
            query_results.append(result)
            func_list.extend(display_names)
        out_df = self._create_table(query, query_results, func_list)
        if frames.empty:
            frames = out_df
        else:
            frames = pd.concat([frames, out_df])
        return frames

    def summary(
        self,
        query_list: Optional[List[RVIdentifier]] = None,
        chain: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        this function outputs a table summarizing results of registered functions
        in self.statistics_dict for requested queries in query_list,
        if chain is None, results correspond to the aggreagated chains
        """
        frames = pd.DataFrame()
        query_list = self._prepare_query_list(query_list)
        for query in query_list:
            out_df = self._execute_summary_stat_funcs(
                query, self.statistics_dict, chain
            )
            frames = pd.concat([frames, out_df])
        frames.sort_index(inplace=True)
        return frames

    def _prepare_plots_input(
        self, query: RVIdentifier, chain: Optional[int] = None
    ) -> Tensor:
        """
        :param query: the query for which registered plot functions are called
        :param chain: the chain that query samples are extracted from
        :returns: tensor of query samples
        """
        query_samples = self.samples[query]
        if chain is not None:
            return query_samples[chain].unsqueeze(0)
        return query_samples

    def plotfn(self, func: Callable, display_name: str) -> Callable:
        """
        this function keeps a directory of all plot-related functions
        so it could handle the overridden functions and new ones that user defines

        :param func: method which is going to be executed when plot() is called.
        :param display_name: appears as part of the plot title for func
        :returns: user-visible function that can be called over a list of queries
        """
        self.plots_dict[func.__name__] = (func, display_name)
        return self._standalone_plot_function(func.__name__, func)

    def _execute_plot_funcs(
        self,
        query: RVIdentifier,
        func_dict: Dict[str, Tuple[Callable, str]],
        chain: Optional[int] = None,
        display: Optional[bool] = False,
    ):  # task T57168727 to add type
        figs = []
        queried_samples = self._prepare_plots_input(query, chain)
        for _k, (func, display_name) in func_dict.items():
            trace, labels = common_plots.plot_helper(queried_samples, func)
            title = f"{self._stringify_query(query)} {display_name}"
            fig = self._display_results(
                trace,
                [title + label for label in labels],
                # pyre-fixme[6]: Expected `bool` for 3rd param but got `Optional[bool]`.
                display,
            )
            figs.append(fig)
        return figs

    def plot(
        self,
        query_list: Optional[List[RVIdentifier]] = None,
        display: Optional[bool] = False,
        chain: Optional[int] = None,
    ):  # task T57168727 to add type
        """
        this function outputs plots related to registered functions in
        self.plots_dict for requested queries in query_list
        :param query_list: list of queries for which plot functions will be called
        :param chain: the chain that query samples are extracted from
        :returns: plotly object holding the results from registered plot functions
        """
        figs = []
        query_list = self._prepare_query_list(query_list)
        for query in query_list:
            fig = self._execute_plot_funcs(query, self.plots_dict, chain, display)
            figs.extend(fig)
        return figs

    def _display_results(
        self, traces, labels: List[str], display: bool
    ):  # task T57168727 to add type
        """
        :param traces: a list of plotly objects
        :param labels: plot labels
        :returns: a plotly subplot object
        """
        fig = make_subplots(
            rows=math.ceil(len(traces) / 2), cols=2, subplot_titles=tuple(labels)
        )

        r = 1
        for trace in traces:
            for data in trace:
                fig.add_trace(data, row=math.ceil(r / 2), col=((r - 1) % 2) + 1)
            r += 1
        if display:
            plotly.offline.iplot(fig)
        return fig

    def _standalone_plot_function(self, func_name: str, func: Callable) -> Callable:
        """
        this function makes each registered plot function directly callable by the user
        """

        @functools.wraps(func)
        def _wrapper(
            query_list: List[RVIdentifier],
            chain: Optional[int] = None,
            display: Optional[bool] = False,
        ):
            figs = []
            query_list = self._prepare_query_list(query_list)
            for query in query_list:
                fig = self._execute_plot_funcs(
                    query, {func_name: self.plots_dict[func_name]}, chain, display
                )
                figs.extend(fig)
            return figs

        return _wrapper

    def _standalone_summary_stat_function(
        self, func_name: str, func: Callable
    ) -> Callable:
        """
        this function makes each registered summary-stat related function directly callable by the user
        """

        @functools.wraps(func)
        def _wrapper(query_list: List[RVIdentifier], chain: Optional[int] = None):
            frames = pd.DataFrame()
            query_list = self._prepare_query_list(query_list)
            for query in query_list:
                out_df = self._execute_summary_stat_funcs(
                    query, {func_name: self.statistics_dict[func_name]}, chain, True
                )
                frames = pd.concat([frames, out_df])
            return frames

        return _wrapper


class Diagnostics(BaseDiagnostics):
    def __init__(self, samples: MonteCarloSamples):
        super().__init__(samples)
        """
        every function related to summary stat should be registered in the constructor
        """
        self.mean = self.summaryfn(common_stats.mean, display_names=["avg"])
        self.std = self.summaryfn(common_stats.std, display_names=["std"])
        self.confidence_interval = self.summaryfn(
            common_stats.confidence_interval, display_names=["2.5%", "50%", "97.5%"]
        )
        self.split_r_hat = self.summaryfn(
            common_stats.split_r_hat, display_names=["r_hat"]
        )
        self.effective_sample_size = self.summaryfn(
            common_stats.effective_sample_size, display_names=["n_eff"]
        )
        self.trace = self.plotfn(common_plots.trace_plot, display_name="trace")
        self.autocorr = self.plotfn(common_plots.autocorr, display_name="autocorr")
