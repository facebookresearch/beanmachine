# Copyright (c) Facebook, Inc. and its affiliates.

from typing import List, Optional

import numpy as np
import pandas as pd
from beanmachine.ppl.diagnostics.common_statistics import confidence_interval, mean, std
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from beanmachine.ppl.model.utils import RVIdentifier
from torch import Tensor


class BaseDiagnostics:
    def __init__(self, samples: MonteCarloSamples):
        self.samples = samples
        self.statistics_dict = {}

    def summaryfn(
        self, func, display_names: List[str], statistics_name: str = None
    ):
        """
        this function keeps a directory of all summary-related functions,
        so it could handle the overridden functions and new ones that user defines
        """
        if not statistics_name:
            statistics_name = func.__name__
        self.statistics_dict[statistics_name] = (func, display_names)

    def _prepare_input(self, query: RVIdentifier, chain: Optional[int] = None):
        query_samples = self.samples[query]
        if chain is not None:
            query_samples = query_samples[chain].unsqueeze(0)
        new_dim = (-1, )
        new_dim += tuple(query_samples.shape[2:])
        return query_samples.view(new_dim)

    def _create_table(
        self, query: RVIdentifier, results: List[Tensor], func_list: List[str]
    ) -> pd.DataFrame:
        out_pd = pd.DataFrame()
        single_result_set = results[0]
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
        if query_list is None:
            query_list = list(self.samples.get_rv_names())
        for query in query_list:
            if not (query in self.samples.get_rv_names()):
                raise ValueError(
                    f"query {self._stringify_query(query)} does not exist"
                )
            query_results = []
            func_list = []
            queried_samples = self._prepare_input(query, chain)
            for _k, (func, display_names) in self.statistics_dict.items():
                result = func(queried_samples)
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
        frames.sort_index(inplace=True)
        return frames


class Diagnostics(BaseDiagnostics):
    def __init__(self, samples: MonteCarloSamples):
        super().__init__(samples)
        """
        every function related to summary stat should be registered in the constructor
        """
        self.summaryfn(mean, display_names=["avg"])
        self.summaryfn(std, display_names=["std"])
        self.summaryfn(
            confidence_interval, display_names=["2.5%", "50%", "97.5%"]
        )
