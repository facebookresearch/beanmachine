# Copyright (c) Facebook, Inc. and its affiliates.
from abc import ABCMeta, abstractmethod
from collections import defaultdict

import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.inference.abstract_infer import AbstractInference
from beanmachine.ppl.model.statistical_model import StatisticalModel
from beanmachine.ppl.model.utils import Mode


class AbstractSingleSiteMHInference(AbstractInference, metaclass=ABCMeta):
    """
    Abstract inference object that all single-site inference algorithms inherit
    from.
    """

    def __init__(self, queries, observations):
        super().__init__(queries, observations)

    def initialize_world(self):
        """
        Initializes the world variables with queries and observation calls.
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
    def propose(self, node):
        """
        Abstract method to be implemented by classes that inherit from
        AbstractSingleSiteMHInference.

        this implementation will include how the inference algorithm proposes a
        new value for a given node.
        """
        raise NotImplementedError("Inference algorithm must implement propose.")

    def accept_or_reject_update(
        self, node_log_update, children_log_updates, proposal_log_update
    ):
        """
        Accepts or rejects the change in the diff by setting a stochastic
        threshold by drawing a sample from a Uniform distribution.
            accepts the change if children's log_prob update is larger than this
            threshold and rejects otherwise.
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

    def _infer(self, num_samples):
        """
        Run inference algorithms.

        parameter is:
            num_samples: number of samples to collect for the query.

        returns:
            samples for the query
        """
        self.initialize_world()
        queries_sample = defaultdict(list)
        for _ in range(num_samples):
            for node in self.world_.get_all_world_vars():
                if node in self.observations_:
                    continue
                proposed_value, proposal_log_update = self.propose(node)

                children_log_updates, world_log_updates, node_log_update = self.world_.propose_change(
                    node, proposed_value, self.stack_
                )
                self.accept_or_reject_update(
                    node_log_update, children_log_updates, proposal_log_update
                )
            for query in self.queries_:
                query_func, query_args = query
                queries_sample[query].append(query_func._wrapper(*query_args))
            self.world_.accept_diff()
        return queries_sample
