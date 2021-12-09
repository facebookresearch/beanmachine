# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import ABCMeta
from collections import defaultdict
from typing import Callable, Dict, Union

import torch
import torch.distributions as dist
from beanmachine.ppl.experimental.abc.abc_infer import ApproximateBayesianComputation
from beanmachine.ppl.inference.utils import safe_log_prob_sum, VerboseLevel
from beanmachine.ppl.legacy.inference.proposer.single_site_random_walk_proposer import (
    SingleSiteRandomWalkProposer,
)
from beanmachine.ppl.legacy.world import World
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.model.utils import LogLevel
from torch import Tensor, tensor
from tqdm.auto import tqdm


LOGGER_UPDATES = logging.getLogger("beanmachine.debug.updates")
LOGGER_ERROR = logging.getLogger("beanmachine.error")


class AdaptiveApproximateBayesianComputationSequentialMonteCarlo(
    ApproximateBayesianComputation, metaclass=ABCMeta
):
    """
    Inference object for adaptive ABC-SMC inference.
    This implemenation follows from the paper: https://pubmed.ncbi.nlm.nih.gov/19880371/
    """

    def __init__(
        self,
        initial_tolerance: float,
        target_tolerance: float,
        perturb_kernel: Union[Callable, None] = None,
        distance_function: Union[Dict, Callable] = torch.dist,
        max_attempts_per_sample: int = 10000,
        survival_rate: float = 0.5,
    ):
        """
        :initial tolerance: The tolerance to set for the first stage of inference
        :param target tolreance: The inference process stops when this tolrance value is attained
        :param perturb_kernel: The pertrubation kernel which takes previously accepted sample
        and performs perturbation to it
        :param distance_function: This can be a single Callable method which will be applied to all
        summay statistics, or a dict which would have the summary statistics as keys and the specific
        distance functions as values
        :param max_attempts_per_sample: number of attempts to make per sample before inference stops
        :param survival_rate: the fraction of samples to pass to the next stage
        """
        super().__init__()
        self.max_attempts_per_sample = max_attempts_per_sample
        if perturb_kernel is None:
            self.perturb_kernel = self.gaussian_random_walk_with_mh_steps
        else:
            self.perturb_kernel = perturb_kernel
        self.distance_function = distance_function
        self.tolerance = initial_tolerance
        self.target_tolerance = target_tolerance
        self.survival_rate = survival_rate
        self.queries_sample_weights = []
        self.previous_stage_queries_sample_distances = []
        self.previous_stage_queries_sample = defaultdict()
        self.step_size = 0.01
        self.num_adaptive_samples = 0

    def weighted_sample_draw(self) -> Dict:
        """
        performs a weighted draw from samples of previous stage.
        self.previous_stage_queries_sample_distances stores the distance values of previous stage
        samples. The distance is stored per joint sample.
        This method makes a weighted sample draw from them by weighing lower distance higher.
        """
        weighted_sample_draw = {}
        # We draw by giving more weight to samples with lower distance, we know max distance = tolerance
        weights = self.tolerance - self.previous_stage_queries_sample_distances
        draw = dist.Categorical(weights).sample()
        for sample, value in self.previous_stage_queries_sample.items():
            weighted_sample_draw[sample] = value[draw]
        return weighted_sample_draw

    def gaussian_random_walk_with_mh_steps(self, sample: Dict, stage: int) -> Dict:
        """
        Default perturb kernel. Performs a sigle step gaussian random walk. Variance decays with stage
        :returns: perturbed sample
        """
        proposer = SingleSiteRandomWalkProposer(step_size=self.step_size)
        for key, value in sample.items():
            self.world_.call(key)
            node_var = self.world_.get_node_in_world(key)
            # pyre-fixme
            node_var.update_value(value)
            is_accepted = False
            (
                proposed_value,
                negative_proposal_log_update,
                auxiliary_variables,
            ) = proposer.propose(key, self.world_)
            node_var.update_value(proposed_value)
            proposal_distribution, _ = proposer.get_proposal_distribution(
                key,
                # pyre-fixme
                node_var,
                self.world_,
                auxiliary_variables,
            )
            positive_proposal_log_update = safe_log_prob_sum(
                proposal_distribution.proposal_distribution, value
            )
            proposal_log_update = (
                positive_proposal_log_update + negative_proposal_log_update
            )
            node_var.update_value(value)
            if proposal_log_update >= tensor(0.0):
                is_accepted = True
            else:
                alpha = dist.Uniform(tensor(0.0), tensor(1.0)).sample().log()
                if proposal_log_update > alpha:
                    is_accepted = True
                else:
                    is_accepted = False
            acceptance_prob = torch.min(
                tensor(1.0, dtype=proposal_log_update.dtype),
                torch.exp(proposal_log_update),
            )
            if self.num_accepted_samples <= self.num_adaptive_samples:
                # do adaptation in each stage
                # pyre-fixme
                if node_var.value.reshape([-1]).shape[0] == 1:
                    target_acc_rate = tensor(0.44)
                    c = torch.reciprocal(target_acc_rate)
                else:
                    target_acc_rate = tensor(0.234)
                    c = torch.reciprocal(1.0 - target_acc_rate)
                new_step_size = self.step_size * torch.exp(
                    (acceptance_prob - target_acc_rate)
                    * c
                    / (self.num_accepted_samples + 1.0)
                )
                self.step_size = new_step_size.item()
            # pyre-fixme
            node_var.update_value(proposed_value if is_accepted else value)

    def _single_inference_step(self, stage: int) -> int:
        """
        Single inference step of the adaptive ABC-SMC algorithm which attempts to obtain a sample.
        In the first stage, samples are generated from the prior of the node to be observed, and their
        summary statistic is compared with summary statistic of provided observations. If distance is
        within provided tolerence values, the sample is accepted. In concequent stages, the samples are
        drawn from accepted samples of the last stage, perturbed using a perturbation kernel and
        then the summray statistic is computed and compared for accept/reject. Each stage has a different
        tolerance value which is computed from the max distance of accepted sample from last stage.
        :param stage: the stage of Adaptive ABC-SMC inference used to choose from the tolerance schedule
        :returns: 1 if sample is accepted and 0 if sample is rejected (used to update the tqdm iterator)
        """
        self.world_ = World()
        self.world_.set_initialize_from_prior(True)
        self.world_.set_maintain_graph(False)
        self.world_.set_cache_functionals(True)

        if not stage == 0:
            # do a weighted sample draw and perturb step
            weighted_sample = self.weighted_sample_draw()
            self.perturb_kernel(weighted_sample, stage)

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

            # perform rejection
            distance = distance_function(
                computed_summary.float(), observed_summary.float()
            )
            reject = torch.gt(distance, self.tolerance)
            if reject:
                self._reject_sample(node_key=summary_statistic)
                return 0
            weights.append(distance)
        self.queries_sample_weights.append(torch.mean(torch.stack(weights)))
        self._accept_sample()
        return 1

    def prepare_next_stage(self, num_samples: int, stage: int):
        """
        select the top (survival_rate * num_samples) samples with min distance for next stage
        """
        self.previous_stage_queries_sample_distances, indices = torch.topk(
            torch.stack(self.queries_sample_weights),
            int(num_samples * self.survival_rate),
            largest=False,
        )
        self.tolerance = torch.max(self.previous_stage_queries_sample_distances).item()
        if self.tolerance <= self.target_tolerance:
            return
        self.previous_stage_queries_sample = defaultdict()
        for key in self.queries_sample:
            self.previous_stage_queries_sample[key] = self.queries_sample[key][indices]
        self.queries_sample = defaultdict()
        self.num_accepted_samples = 0
        self.queries_sample_weights = []

    def _infer(
        self,
        num_samples: int,
        num_adaptive_samples: int = 0,
        verbose: VerboseLevel = VerboseLevel.LOAD_BAR,
        initialize_from_prior: bool = True,
    ) -> Dict[RVIdentifier, Tensor]:
        """
        Run adaptive ABC-SMC inference algorithm

        :param num_samples: number of samples excluding adaptation.
        :param num_adapt_steps: not used in rejection sampling
        :param verbose: Integer indicating how much output to print to stdio
        :param initialize_from_prior: boolean to initialize samples from prior
        :returns: samples for the query
        """
        stage = 0
        self.num_adaptive_samples = num_adaptive_samples
        while self.tolerance >= self.target_tolerance:
            total_attempted_samples = 0
            pbar = tqdm(
                desc=f"Stage {stage + 1} with tolerance {self.tolerance}",
                total=num_samples,
                disable=not bool(verbose == VerboseLevel.LOAD_BAR),
            )
            while self.num_accepted_samples < num_samples:
                pbar.update(self._single_inference_step(stage))
                total_attempted_samples += 1
            pbar.close()
            LOGGER_UPDATES.log(
                LogLevel.DEBUG_UPDATES.value,
                f"Inference stage {stage} completed; accepted {num_samples} from \
                    {total_attempted_samples} attempted samples. \
                    \nAcceptance rate: {float(num_samples/total_attempted_samples)}",
            )
            self.prepare_next_stage(num_samples, stage)
            stage += 1
        return self.queries_sample
