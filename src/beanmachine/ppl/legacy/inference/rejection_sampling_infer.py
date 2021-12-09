# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import ABCMeta
from collections import defaultdict
from typing import Dict

import torch
from beanmachine.ppl.inference.utils import VerboseLevel
from beanmachine.ppl.legacy.inference.abstract_infer import AbstractMCInference
from beanmachine.ppl.legacy.world import World
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.model.utils import LogLevel
from torch import Tensor
from tqdm.auto import tqdm


LOGGER_INFERENCE = logging.getLogger("beanmachine.inference")


class RejectionSampling(AbstractMCInference, metaclass=ABCMeta):
    """
    Inference object for rejection sampling inference. ABC inference
    algorithms will inherit from this class, and override the single_inference_step method
    """

    _observations_must_be_rv = False

    def __init__(self, max_attempts_per_sample=1e4, tolerance=0.0):
        super().__init__()
        self.num_accepted_samples = 0
        self.queries_sample = defaultdict()
        self.attempts_per_sample = 0
        self.max_attempts_per_sample = int(max_attempts_per_sample)
        self.tolerance = torch.tensor(tolerance)

    def _accept_sample(self):
        """
        This method performs the procedure of accepting a sample. It adds the requested qureies into the
        accepted samples dict, updated number of accepted samples and resets the counter for attempts per
        sample.
        """
        for query in self.queries_:
            # unsqueeze the sampled value tensor, which adds an extra dimension
            # along which we'll be adding samples generated at each iteration
            query_val = self.world_.call(query).unsqueeze(0).clone()
            if query not in self.queries_sample:
                self.queries_sample[query] = query_val
            else:
                self.queries_sample[query] = torch.cat(
                    [
                        self.queries_sample[query],
                        query_val,
                    ],
                    dim=0,
                )
        self.num_accepted_samples += 1
        self.attempts_per_sample = 0

    def _reject_sample(self, node_key: RVIdentifier):
        """
        This method performs the procedure of rejecting a sample. This includes logging the rejection,
        incrementing the number of attempts for current sample, and raising an error if they excced the
        max amount.
        :param node_key: the node which triggered the sample rejection. used for debug logging
        """
        self.attempts_per_sample += 1
        LOGGER_INFERENCE.log(
            LogLevel.DEBUG_UPDATES.value,
            f"sample {self.num_accepted_samples}, attempt {self.attempts_per_sample}"
            + f" failed\n rejected node: {node_key}",
        )
        # check if number of attempts per sample exceeds the max allowed, report error and exit
        if self.attempts_per_sample >= self.max_attempts_per_sample:
            raise RuntimeError("max_attempts_per_sample exceeded")

    def _single_inference_step(self) -> int:
        """
        Single inference step of the rejection sampling algorithm which attempts to obtain a sample.
        Samples are generated from prior of the node to be observed, and are compared with provided
        observations.If all observations are equal or within provided tolerence values, the sample
        is accepted.

        :returns: 1 if sample is accepted and 0 if sample is rejected (used to update the tqdm iterator)
        """
        self.world_ = World()
        self.world_.set_initialize_from_prior(True)
        self.world_.set_maintain_graph(False)
        self.world_.set_cache_functionals(True)

        for node_key, node_observation in self.observations_.items():
            temp_sample = self.world_.call(node_key)
            node_var = self.world_.get_node_in_world(node_key)
            # a functional will not be in the world, so we access its sample differently
            node_var_sample = node_var.value if node_var else temp_sample
            # check if node_observation is a tensor, if not, cast it
            if not torch.is_tensor(node_observation):
                node_observation = torch.tensor(node_observation)
            # check if shapes match else throw error
            if node_var_sample.shape != node_observation.shape:
                raise ValueError(
                    f"Shape mismatch in random variable {node_key}"
                    + "\nshape does not match with observation\n"
                    + f"Expected observation shape: {node_var_sample.shape};"
                    + f"Provided observation shape{node_observation.shape}"
                )
            # perform rejection
            samples_dont_match = torch.gt(
                torch.abs(node_var_sample.float() - node_observation.float()),
                self.tolerance,
            )
            # pyre-fixme
            if samples_dont_match.any():
                self._reject_sample(node_key)
                return 0

        self._accept_sample()
        return 1

    def _infer(
        self,
        num_samples: int,
        num_adaptive_samples: int = 0,
        verbose: VerboseLevel = VerboseLevel.LOAD_BAR,
        initialize_from_prior: bool = False,
    ) -> Dict[RVIdentifier, Tensor]:
        """
        Run rejection sampling inference algorithm

        :param num_samples: number of samples excluding adaptation.
        :param num_adapt_steps: not used in rejection sampling
        :param verbose: Integer indicating how much output to print to stdio
        :param initialize_from_prior: boolean to initialize samples from prior
        :returns: samples for the query
        """
        self.num_accepted_samples = 0
        self.queries_sample = defaultdict()
        total_attempted_samples = 0
        pbar = tqdm(
            total=num_samples, disable=not bool(verbose == VerboseLevel.LOAD_BAR)
        )
        while self.num_accepted_samples < num_samples:
            pbar.update(self._single_inference_step())
            total_attempted_samples += 1
        pbar.close()
        LOGGER_INFERENCE.log(
            LogLevel.DEBUG_UPDATES.value,
            f"Inference completed; accepted {num_samples} from \
             {total_attempted_samples} attempted samples. \
             \nAcceptance rate: {float(num_samples/total_attempted_samples)}",
        )
        return self.queries_sample
