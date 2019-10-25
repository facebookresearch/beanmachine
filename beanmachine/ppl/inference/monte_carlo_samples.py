# Copyright (c) Facebook, Inc. and its affiliates.abs

import copy
from typing import Dict, List

from beanmachine.ppl.inference.monte_carlo_samples_data import MonteCarloSamplesData
from beanmachine.ppl.model.utils import RVIdentifier
from torch import Tensor


class MonteCarloSamples(object):
    """
    Represents a view of the data representing the results of infer

    If no chain is specified, the data across all chains is accessible
    If a chain is specified, only the data from the chain will be accessible
    """

    def __init__(self, chain_results: List[Dict[RVIdentifier, Tensor]]):
        self.data = MonteCarloSamplesData(chain_results)
        self.chain = None

    def __getitem__(self, rv: RVIdentifier) -> Tensor:
        """
        Let C be the number of chains,
        S be the number of samples

        if no chain specified:
            samples[var] returns a Tensor of (C, S, (shape of Var))
        if a chain is specified:
            samples[var] returns a Tensor of (S, (shape of Var))

        :param rv: random variable to see samples
        :returns: samples drawn during inference for the specified variable
        """
        if self.chain is None:
            return self.data.rv_dict[rv]
        else:
            return self.data.rv_dict[rv][self.chain]

    def __str__(self):
        return str(self.data.rv_dict)

    def _specific_chain_copy(self, chain: int):
        new_mcs = copy.copy(self)
        new_mcs.chain = chain
        return new_mcs

    def get_chain(self, chain: int = 0):
        """
        View the samples drawn by during inference for a specific chain

        :param chain: specific chain to view
        :returns: view of the data restricted to specified chain
        """
        if self.chain is not None:
            raise ValueError(
                f"The current MonteCarloSamples object has already been"
                f" restricted to chain {self.chain}"
            )
        elif chain < 0 or chain >= self.get_num_chains():
            raise IndexError("Please specify a valid chain")
        return self._specific_chain_copy(chain)

    def get_variable(self, rv: RVIdentifier) -> Tensor:
        """
        :param rv: random variable to view values of
        :results: samples drawn during inference for the specified variable
        """
        return self[rv]

    def get_rv_names(self) -> List[RVIdentifier]:
        """
        :returns: a list of the queried random variables
        """
        return list(self.data.rv_dict.keys())

    def get_num_chains(self) -> int:
        """
        :returns: the number of chains run during inference
        """
        return self.data.num_chains
