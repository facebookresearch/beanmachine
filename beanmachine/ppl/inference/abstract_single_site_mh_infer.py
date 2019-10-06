# Copyright (c) Facebook, Inc. and its affiliates.
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Dict, Tuple

import torch
import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.inference.abstract_infer import AbstractInference
from beanmachine.ppl.model.statistical_model import StatisticalModel
from beanmachine.ppl.model.utils import Mode, RandomVariable
from torch import Tensor


class AbstractSingleSiteMHInference(AbstractInference, metaclass=ABCMeta):
    """
    Abstract inference object that all single-site MH inference algorithms
    inherit from.
    """

    def __init__(self):
        super().__init__()

    def initialize_world(self):
        """
        Initializes the world variables with queries and observation calls.

        :param queries: random variables to query
        :param observations: observed random variables with their values
        """
        StatisticalModel.set_observations(self.observations_)
        StatisticalModel.set_mode(Mode.INFERENCE)
        for node in self.observations_:
            # makes the call for the observation node, which will run sample(node())
            # that results in adding its corresponding Variable and its dependent
            # Variable to the world
            node_func, node_args = node
            node_func._wrapper(*node_args)
        for node in self.queries_:
            # makes the call for the query node, which will run sample(node())
            # that results in adding its corresponding Variable and its dependent
            # Variable to the world.
            node_func, node_args = node
            node_func._wrapper(*node_args)

        self.world_.accept_diff()

    @abstractmethod
    def propose(self, node: RandomVariable) -> Tuple[Tensor, Tensor]:
        """
        Abstract method to be implemented by classes that inherit from
        AbstractSingleSiteMHInference. This implementation will include how the
        inference algorithm proposes a new value for a given node.

        :param node: the node for which we'll need to propose a new value for.
        :returns: a new proposed value for the node and the difference in log_prob
        of the old and newly proposed value.
        """
        raise NotImplementedError("Inference algorithm must implement propose.")

    def accept_or_reject_update(
        self,
        node_log_update: Tensor,
        children_log_updates: Tensor,
        proposal_log_update: Tensor,
    ):
        """
        Accepts or rejects the change in the diff by setting a stochastic
        threshold by drawing a sample from a Uniform distribution. It accepts
        the change if sum of all log_prob updates are larger than this threshold
        and rejects otherwise.

        :param node_log_update: log_prob update to the node that was resampled
        from.
        :param children_log_updates: log_prob updates of the immediate children
        of the node that was resampled from.
        :param proposal_log_update: log_prob update of the proposal
        """
        log_update = children_log_updates + node_log_update + proposal_log_update

        if log_update >= tensor(0.0):
            self.world_.accept_diff()
        else:
            alpha = dist.Uniform(tensor(0.0), tensor(1.0)).sample().log()
            if log_update > alpha:
                self.world_.accept_diff()
            else:
                self.world_.reject_diff()

    def post_process(self, node: RandomVariable) -> Tensor:
        """
        To be implemented by inference algorithms that need post-processing
        after diff is created to compute the final proposal log update.

        :param node: the node for which we have already proposed a new value for.
        :param proposal_log_update: proposal log update computed up until now.
        :returns: final proposal log update
        """
        return tensor(0.0)

    def _infer(self, num_samples: int) -> Dict[RandomVariable, Tensor]:
        """
        Run inference algorithms.

        :param num_samples: number of samples to collect for the query.
        :returns: samples for the query
        """
        self.initialize_world()
        queries_sample = defaultdict()
        for _ in range(num_samples):
            for node in self.world_.get_all_world_vars().copy():
                if node in self.observations_:
                    continue
                if not self.world_.contains_in_world(node):
                    continue
                proposed_value, negative_proposal_log_update = self.propose(node)

                children_log_updates, world_log_updates, node_log_update = self.world_.propose_change(
                    node, proposed_value, self.stack_
                )
                positive_proposal_log_update = self.post_process(node)
                proposal_log_update = (
                    positive_proposal_log_update + negative_proposal_log_update
                )
                self.accept_or_reject_update(
                    node_log_update, children_log_updates, proposal_log_update
                )
            for query in self.queries_:
                query_func, query_args = query
                # unsqueeze the sampled value tensor, which adds an extra dimension
                # along which we'll be adding samples generated at each iteration
                if query not in queries_sample:
                    queries_sample[query] = query_func._wrapper(*query_args).unsqueeze(
                        0
                    )
                else:
                    queries_sample[query] = torch.cat(
                        [
                            queries_sample[query],
                            query_func._wrapper(*query_args).unsqueeze(0),
                        ],
                        dim=0,
                    )
            self.world_.accept_diff()

        return queries_sample
