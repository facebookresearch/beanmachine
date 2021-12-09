# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import ABCMeta
from collections import defaultdict
from typing import Callable, Dict, List, Union

import torch
import torch.distributions as dist
from beanmachine.ppl.experimental.abc.abc_infer import ApproximateBayesianComputation
from beanmachine.ppl.inference.utils import VerboseLevel
from beanmachine.ppl.legacy.inference.proposer.single_site_random_walk_proposer import (
    SingleSiteRandomWalkProposer,
)
from beanmachine.ppl.legacy.world import World
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.model.utils import LogLevel
from torch import Tensor
from tqdm.auto import tqdm


LOGGER_PROPOSER = logging.getLogger("beanmachine.proposer")
LOGGER = logging.getLogger("beanmachine")


class ApproximateBayesianComputationSequentialMonteCarlo(
    ApproximateBayesianComputation, metaclass=ABCMeta
):
    """
    Inference object for ABC-SMC inference.
    This implemenation follows from the paper: https://pubmed.ncbi.nlm.nih.gov/19880371/
    """

    def __init__(
        self,
        tolerance_schedule: Union[Dict, List],
        perturb_kernel: Union[Callable, None] = None,
        distance_function: Union[Dict, Callable] = torch.dist,
        max_attempts_per_sample: int = 10000,
    ):
        """
        :param perturb_kernel: The pertrubation kernel which takes previously accepted sample
        and performs perturbation to it
        :param distance_function: This can be a single Callable method which will be applied to all
        summay statistics, or a dict which would have the summary statistics as keys and the specific
        distance functions as values
        :param tolerance_schedule: This can be a single list of tolerances which will be applied to all
        summay statistics, or a dict which would have the summary statistics as keys and the specific
        list tolerances as values
        :param max_attempts_per_sample: number of attempts to make per sample before inference stops
        """
        super().__init__()
        self.max_attempts_per_sample = max_attempts_per_sample
        if perturb_kernel is None:
            self.perturb_kernel = self.gaussian_random_walk
        else:
            self.perturb_kernel = perturb_kernel
        self.distance_function = distance_function
        self.tolerance_schedule = tolerance_schedule
        self.queries_sample_weights = []
        self.previous_stage_queries_sample_weights = []
        self.previous_stage_queries_sample = defaultdict()

    def weighted_sample_draw(self) -> Dict:
        """
        performs a weighted draw from samples of previous stage
        """
        weighted_sample_draw = {}
        draw = dist.Categorical(self.previous_stage_queries_sample_weights).sample()

        for sample, value in self.previous_stage_queries_sample.items():
            weighted_sample_draw[sample] = value[draw]
        return weighted_sample_draw

    def gaussian_random_walk(self, sample: Dict, stage: int) -> Dict:
        """
        Default perturb kernel. Performs a sigle step gaussian random walk. Variance decays with stage
        :returns: perturbed sample
        """
        proposer = SingleSiteRandomWalkProposer(
            step_size=(
                0.01 * (torch.exp(-(torch.tensor(stage, dtype=torch.float32)))).item()
            )
        )
        perturbed_sample = {}
        for key, value in sample.items():
            self.world_.call(key)
            node_var = self.world_.get_node_in_world(key)
            # pyre-fixme
            node_var.update_value(value)
            perturbation, _, _ = proposer.propose(key, self.world_)
            perturbed_sample[key] = perturbation

        return perturbed_sample

    def _single_inference_step(self, stage: int) -> int:
        """
        Single inference step of the ABC-SMC algorithm which attempts to obtain a sample. In the first
        stage, samples are generated from the prior of the node to be observed, and their summary
        statistic is compared with summary statistic of provided observations. If distance is within
        provided tolerence values, the sample is accepted. In concequent stages, the samples are drawn
        from the pool of accepted samples of the last stage, perturbed using a perturbation kernel and
        then the summray statistic is computed and compared for accept/reject. Each stage has a different
        tolerance value.
        :param stage: the stage of ABC-SMC inference used to choose from the tolerance schedule
        :returns: 1 if sample is accepted and 0 if sample is rejected (used to update the tqdm iterator)
        """
        self.world_ = World()
        self.world_.set_initialize_from_prior(True)
        self.world_.set_maintain_graph(False)
        self.world_.set_cache_functionals(True)

        if not stage == 0:
            weighted_sample = self.weighted_sample_draw()
            perturbed_sample_draw = self.perturb_kernel(weighted_sample, stage)
            # although this method is used to set observations, we use it here to set values RVs
            # to the generated perturbations
            self.world_.set_observations(perturbed_sample_draw)
        weights = []
        for summary_statistic, observed_summary in self.observations_.items():
            # makes the call for the summary statistic node, which will run sample(node())
            # that results in adding its corresponding Variable and its dependent
            # Variable to the world, as well as computing it's value
            computed_summary = self.world_.call(summary_statistic)
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
            if isinstance(self.tolerance_schedule, dict):
                tolerance = self.tolerance_schedule[summary_statistic][stage]
            else:
                tolerance = self.tolerance_schedule[stage]

            # perform rejection
            distance = distance_function(
                computed_summary.float(), observed_summary.float()
            )
            reject = torch.gt(distance, tolerance)
            if reject:
                self._reject_sample(node_key=summary_statistic)
                return 0
            weights.append((tolerance - distance) / tolerance)
        self.queries_sample_weights.append(torch.mean(torch.stack(weights)))
        self._accept_sample()
        return 1

    def prepare_next_stage(self):
        """
        sets several dicts to start next stage
        """
        self.num_accepted_samples = 0
        self.previous_stage_queries_sample = self.queries_sample
        self.previous_stage_queries_sample_weights = torch.stack(
            self.queries_sample_weights
        )
        self.queries_sample = defaultdict()
        self.queries_sample_weights = []

    def _infer(
        self,
        num_samples: int,
        num_adaptive_samples: int = 0,
        verbose: VerboseLevel = VerboseLevel.LOAD_BAR,
        initialize_from_prior: bool = True,
    ) -> Dict[RVIdentifier, Tensor]:
        """
        Run ABC-SMC inference algorithm

        :param num_samples: number of samples excluding adaptation.
        :param num_adapt_steps: not used in rejection sampling
        :param verbose: Integer indicating how much output to print to stdio
        :param initialize_from_prior: boolean to initialize samples from prior
        :returns: samples for the query
        """
        if isinstance(self.tolerance_schedule, dict):
            num_stages = len(
                self.tolerance_schedule[list(self.tolerance_schedule.keys())[0]]
            )
        else:
            num_stages = len(self.tolerance_schedule)

        for stage in range(num_stages):
            total_attempted_samples = 0
            pbar = tqdm(
                desc=f"Stage {stage + 1} of {num_stages}",
                total=num_samples,
                disable=not bool(verbose == VerboseLevel.LOAD_BAR),
            )
            while self.num_accepted_samples < num_samples:
                pbar.update(self._single_inference_step(stage))
                total_attempted_samples += 1
            pbar.close()
            LOGGER_PROPOSER.log(
                LogLevel.DEBUG_UPDATES.value,
                f"Inference stage {stage} completed; accepted {num_samples} from \
                    {total_attempted_samples} attempted samples. \
                    \nAcceptance rate: {float(num_samples/total_attempted_samples)}",
            )
            if not stage == (num_stages - 1):
                self.prepare_next_stage()
        return self.queries_sample
