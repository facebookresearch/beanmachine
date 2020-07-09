# Copyright (c) Facebook, Inc. and its affiliates.

from collections import defaultdict
from typing import Dict, List, Union

import torch
from beanmachine.ppl.model.utils import RVIdentifier
from torch import Tensor


class MonteCarloSamplesData:
    """
    Samples data object to store samples of queries
    with multiple chains and additional metadata

    Chain results can be a list of Dict[RVIdentifier, Tensor] or a
    Dict[RVIdentifer, Tensor] with the leading dimension representing
    the number of chains.

    self.rv_dict : Dict[RVIdentifier, Tensor]
    self.acceptance_results_dict : Dict[RVIdentifier, Tensor]
    """

    def __init__(
        self,
        chain_results: Union[
            List[Dict[RVIdentifier, Tensor]], Dict[RVIdentifier, Tensor]
        ],
    ):
        if isinstance(chain_results, list):
            self.rv_dict = defaultdict()
            self.num_chains = len(chain_results)
            if self.num_chains > 0:
                one_chain = chain_results[0]
                for random_variable in one_chain:
                    rv_chains = [chain[random_variable] for chain in chain_results]
                    self.rv_dict[random_variable] = torch.stack(rv_chains)
        else:
            self.rv_dict = chain_results
            self.num_chains = len(next(iter(chain_results.values())))
        self.acceptance_results_dict = defaultdict()
