import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributions as dist
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from beanmachine.ppl.inference.single_site_compositional_infer import (
    SingleSiteCompositionalInference,
)
from beanmachine.ppl.model.statistical_model import sample
from torch import tensor


class CrowdSourcedAnnotationModel(object):
    def __init__(
        self,
        expected_correctness: float,
        concentration: float,
        num_categories: int,
        num_labelers: int,
        num_items: int,
        num_labels_list: List,
        vector_y: List,
        vector_J_i: List,
        iterations: int,
        inference_type: str,
    ):
        alphas = (1 - expected_correctness) / (num_categories - 1)
        alpha = torch.ones((num_categories, num_categories)) * alphas * concentration
        alpha[torch.arange(num_categories), torch.arange(num_categories)] = (
            expected_correctness * concentration
        )
        alpha = alpha.unsqueeze(0).repeat(num_labelers, 1, 1)
        beta = torch.ones(num_categories) * (1 / num_categories)
        self.num_categories = num_categories
        self.num_labelers = num_labelers
        self.num_items = num_items
        self.iterations = iterations
        self.inference_type = inference_type
        self.alpha = alpha
        self.beta = beta
        self.flattened_labelers = tensor(vector_J_i)
        self.flattened_labels = tensor(vector_y)
        self.flattened_items = tensor(
            [
                i
                for i, num_label in enumerate(num_labels_list)
                for _ in range(num_label)
            ],
            dtype=torch.long,
        )

    # prevalence of category classes
    @sample
    def pi(self):
        return dist.Dirichlet(self.beta)

    # Labelers' confusion matrix, conc.shape = (num_labelers,
    # num_categories, num_categories)
    @sample
    def theta(self):
        return dist.Dirichlet(self.alpha)

    # this is a dummy observation of true for each item
    @sample
    def true(self):
        # we will compute the likelihood of items items labels
        likelihood = (
            self.pi()
            * torch.zeros((self.num_items, self.num_categories))
            .scatter_add_(
                0,
                self.flattened_items.unsqueeze(1).expand(-1, self.num_categories),
                self.theta()[self.flattened_labelers, :, self.flattened_labels].log_(),
            )
            .exp_()
        ).sum(dim=1)
        return dist.Bernoulli(likelihood)

    def infer(self) -> Tuple[MonteCarloSamples, float]:
        observed_dict = {self.true(): torch.ones(self.num_items)}
        start_time = 0
        samples = {}
        if self.inference_type == "mcmc":
            mh = SingleSiteCompositionalInference()
            start_time = time.time()
            samples = mh.infer(
                [self.pi(), self.theta()], observed_dict, self.iterations, 1
            ).get_chain()

        elif self.inference_type == "vi":
            print("ImplementationError; exiting...")
            exit(1)
        elapsed_time_sample_beanmachine = time.time() - start_time
        return (samples, elapsed_time_sample_beanmachine)


def obtain_posterior(
    data_train: Tuple[List, List, List],
    args_dict: Dict,
    model: CrowdSourcedAnnotationModel,
) -> Tuple[List, Dict]:
    """
    Beanmachine impmementation of CLARA model.

    :param data_train: tuple of np.ndarray (y, J_i, num_labels)
    :param args_dict: a dict of model arguments
    :returns: samples_beanmachine(dict): posterior samples of all parameters
    :returns: timing_info(dict): compile_time, inference_time
    """
    vector_y, vector_J_i, num_labels = data_train
    num_labelers = int(args_dict["k"])
    num_items = len(num_labels)
    num_labels_list = num_labels
    num_categories, labeler_rate, expected_correctness, concentration = args_dict[
        "model_args"
    ]
    iterations = args_dict["num_samples_beanmachine-vectorized"]
    inference_type = args_dict["inference_type"]

    crowdSourcedAnnotationModel = CrowdSourcedAnnotationModel(
        expected_correctness,
        concentration,
        num_categories,
        num_labelers,
        num_items,
        num_labels_list,
        vector_y,
        vector_J_i,
        iterations,
        inference_type,
    )
    start_time = time.time()
    samples, timing_info = crowdSourcedAnnotationModel.infer()
    elapsed_time_beanmachine = time.time() - start_time

    # repackage samples into shape required by PPLBench
    samples_formatted = []
    for i in range(args_dict["num_samples_beanmachine-vectorized"]):
        sample_dict = {}
        sample_dict["pi"] = (
            samples.get_variable(crowdSourcedAnnotationModel.pi())[i].detach().numpy()
        )
        theta = np.zeros((num_labelers, num_categories, num_categories))
        theta = (
            samples.get_variable(crowdSourcedAnnotationModel.theta())[i]
            .detach()
            .numpy()
        )
        sample_dict["theta"] = theta
        samples_formatted.append(sample_dict)
    timing_info = {"compile_time": 0, "inference_time": elapsed_time_beanmachine}

    return (samples_formatted, timing_info)
