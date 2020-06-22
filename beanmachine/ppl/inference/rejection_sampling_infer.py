# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from abc import ABCMeta
from collections import defaultdict
from typing import Dict

import torch
from beanmachine.ppl.inference.abstract_infer import AbstractInference, VerboseLevel
from beanmachine.ppl.model.statistical_model import StatisticalModel
from beanmachine.ppl.model.utils import LogLevel, Mode, RVIdentifier
from torch import Tensor
from tqdm.auto import tqdm


LOGGER_UPDATES = logging.getLogger("beanmachine.debug.updates")


class RejectionSampling(AbstractInference, metaclass=ABCMeta):
    """
    Inference object for rejection sampling inference. ABC inference
    algorithms will inherit from this class, and override the single_inference_step method
    """

    def __init__(self):
        super().__init__()
        self.num_accepted_samples = 0
        self.queries_sample = defaultdict()

    def single_inference_step(self, initialize_from_prior: bool = True):
        """
        Single inference step of the rejection sampling algorithm.
        Samples from prior of the node to be observed and compares
        with provided observation. If all observations are accepted,
        the queries are appended to samples dict.

        :param initialize_from_prior: boolean to initialize samples from prior
        """
        self.world_ = StatisticalModel.reset()
        self.world_.set_initialize_from_prior(initialize_from_prior)
        StatisticalModel.set_mode(Mode.INFERENCE)
        for node_key, node_observation in self.observations_.items():
            temp_sample = node_key.function._wrapper(*node_key.arguments)
            node_var = self.world_.get_node_in_world(node_key)
            # a functional will not be in the world, so we access its sample differently
            node_var_sample = node_var.value if node_var else temp_sample
            # perform rejection
            samples_dont_match = node_var_sample != node_observation
            reject = (
                samples_dont_match.any()
                if torch.is_tensor(samples_dont_match)
                else bool(samples_dont_match)
            )
            if reject:
                return
        for query in self.queries_:
            # unsqueeze the sampled value tensor, which adds an extra dimension
            # along which we'll be adding samples generated at each iteration
            if query not in self.queries_sample:
                self.queries_sample[query] = (
                    query.function._wrapper(*query.arguments).unsqueeze(0).clone()
                )
            else:
                self.queries_sample[query] = torch.cat(
                    [
                        self.queries_sample[query],
                        query.function._wrapper(*query.arguments).unsqueeze(0).clone(),
                    ],
                    dim=0,
                )
        self.num_accepted_samples += 1

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
        # pyre-fixme
        pbar = tqdm(
            total=num_samples, disable=not bool(verbose == VerboseLevel.LOAD_BAR)
        )
        while self.num_accepted_samples < num_samples:
            prev_accepted_samples = self.num_accepted_samples
            self.single_inference_step()
            pbar.update(self.num_accepted_samples - prev_accepted_samples)
            total_attempted_samples += 1
            # give a warning if number of attempts are 100x, 1000x and 10000x num_samples
            if total_attempted_samples in (
                num_samples * factor for factor in [100, 1000, 10000]
            ):
                LOGGER_UPDATES.log(
                    LogLevel.DEBUG_UPDATES.value,
                    f"Very low acceptance rate; consider respecifing the model? \
                    \nAccepted {self.num_accepted_samples} from {total_attempted_samples} attempted samples. \
                    \nAcceptance rate: {float(self.num_accepted_samples/total_attempted_samples)}",
                )
        pbar.close()
        LOGGER_UPDATES.log(
            LogLevel.DEBUG_UPDATES.value,
            f"Inference completed; accepted {num_samples} from {total_attempted_samples} attempted samples. \
             \nAcceptance rate: {float(num_samples/total_attempted_samples)}",
        )
        return self.queries_sample
