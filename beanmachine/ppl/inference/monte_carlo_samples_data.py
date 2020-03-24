# Copyright (c) Facebook, Inc. and its affiliates.

from collections import defaultdict
from typing import Dict, List

import torch
from beanmachine.ppl.model.utils import RVIdentifier
from torch import Tensor


class MonteCarloSamplesData:
    """
    Samples data object to store samples of queries
    with multiple chains and additional metadata

    self.rv_dict : Dict[RVIdentifier, Tensor]
    self.acceptance_results_dict : Dict[RVIdentifier, Tensor]
    """

    def __init__(self, chain_results: List[Dict[RVIdentifier, Tensor]]):
        self.rv_dict = defaultdict()
        self.acceptance_results_dict = defaultdict()

        self.num_chains = len(chain_results)

        if self.num_chains > 0:
            one_chain = chain_results[0]
            for random_variable in one_chain:
                rv_chains = [chain[random_variable] for chain in chain_results]
                self.rv_dict[random_variable] = torch.stack(rv_chains)
