# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import ABCMeta
from typing import Callable, Dict, Union

import torch
from beanmachine.ppl.legacy.inference.rejection_sampling_infer import RejectionSampling
from beanmachine.ppl.legacy.world import World


LOGGER = logging.getLogger("beanmachine")


class ApproximateBayesianComputation(RejectionSampling, metaclass=ABCMeta):
    """
    Inference object for vanilla ABC inference. Other ABC inference types will inherit from this class.
    For details refer to this paper: https://doi.org/10.1371/journal.pcbi.1002803
    """

    def __init__(
        self,
        distance_function: Union[Dict, Callable] = torch.dist,
        tolerance: Union[Dict, float] = 0.0,
        max_attempts_per_sample: int = 10000,
        simulate: bool = False,
    ):
        """
        :param distance_function: This can be a single Callable method which will be applied to all
        summay statistics, or a dict which would have the summary statistics as keys and the specific
        distance functions as values
        :param tolerance: This can be a single float value which will be applied to all
        summay statistics, or a dict which would have the summary statistics as keys and the specific
        tolerances as values
        :param max_attempts_per_sample: number of attempts to make per sample before inference stops
        :param simulate: if True, operate in simulation mode else perform inference
        """
        super().__init__(max_attempts_per_sample)
        self.distance_function = distance_function
        self.tolerance = tolerance
        self.simulate = simulate

    def _single_inference_step(self) -> int:
        """
        Single inference step of the vanilla ABC algorithm which attempts to obtain a sample.
        Samples are generated from the prior of the node to be observed, and their summary statistic is
        compared with summary statistic of provided observations. If distance is within provided
        tolerence values, the sample is accepted.

        :returns: 1 if sample is accepted and 0 if sample is rejected (used to update the tqdm iterator)
        """
        self.world_ = World()
        self.world_.set_initialize_from_prior(True)
        self.world_.set_maintain_graph(False)
        self.world_.set_cache_functionals(True)

        if self.simulate:
            # in simulate mode, user passes obtained samples as observations and shall query nodes to be
            # simulated. This required observations to set instead of being sampled from prior
            self.world_.set_observations(self.observations_)

        for summary_statistic, observed_summary in self.observations_.items():
            # makes the call for the summary statistic node, which will run sample(node())
            # that results in adding its corresponding Variable and its dependent
            # Variable to the world, as well as computing it's value
            computed_summary = self.world_.call(summary_statistic)
            if self.simulate:
                # if we are simulating, simply accept sample
                self._accept_sample()
                return 1
            # check if passed observation is a tensor, if not, cast it
            if not torch.is_tensor(observed_summary):
                observed_summary = torch.tensor(observed_summary)
            # check if the shapes of computed and provided summary matches
            if computed_summary.shape != observed_summary.shape:
                raise ValueError(
                    f"Shape mismatch in random variable {summary_statistic}"
                    + "\nshape does not match with observation\n"
                    + f"Expected observation shape: {computed_summary.shape};"
                    + f"Provided observation shape{observed_summary.shape}"
                )

            # if user passed a dict for distance functions, load from it, else load default
            if isinstance(self.distance_function, dict):
                distance_function = self.distance_function[summary_statistic]
            else:
                distance_function = self.distance_function
            # we allow users to pass either a dict or a single value for tolerance
            if isinstance(self.tolerance, dict):
                tolerance = self.tolerance[summary_statistic]
            else:
                tolerance = self.tolerance

            # perform rejection
            reject = torch.gt(
                distance_function(computed_summary.float(), observed_summary.float()),
                tolerance,
            )
            if reject:
                self._reject_sample(node_key=summary_statistic)
                return 0
        self._accept_sample()
        return 1
