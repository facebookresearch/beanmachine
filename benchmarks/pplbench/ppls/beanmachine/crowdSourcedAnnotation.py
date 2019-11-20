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


def format_data(Y_i: List, J_i: List, num_labels: List) -> Tuple[List, List]:
    """
    param: Y_i: List of all the labels produced by different labelers
    param: J_i: List of the labelers that produced the respective label
    param: num_labels: List representing number of labels for each item

    Example:
    Y_i = [1, 1, 1, 0, 2, 2, 1, 2, 1, 1, 1, 0, 0,
        1, 1, 1, 0, 1, 1, 2, 0, 1, 1, 2, 0, 0, 2, 1]
    J_i = [0, 7, 5, 1, 4, 9, 3, 8, 5, 3, 5, 9, 2,
        4, 2, 6, 0, 9, 5, 8, 1, 9, 4, 8, 9, 0, 7, 5]
    num_labels = [1, 4, 4, 2, 2, 3, 5, 1, 5, 1]

    Returns:
    [[1], [1, 1, 0, 2], [2, 1, 2, 1], [1, 1], [0, 0],
    [1, 1, 1], [0, 1, 1, 2, 0], [1], [1, 2, 0, 0, 2], [1]]
    [[0], [7, 5, 1, 4], [9, 3, 8, 5], [3, 5], [9, 2],
    [4, 2, 6], [0, 9, 5, 8, 1], [9], [4, 8, 9, 0, 7], [5]]
    """
    Y_ij = []
    J_ij = []
    counter = 0
    for i in range(len(num_labels)):
        y = []
        j = []
        num_labelers = num_labels[i]
        for _j in range(num_labelers):
            y.append(Y_i[counter])
            j.append(J_i[counter])
            counter += 1

        Y_ij.append(y)
        J_ij.append(j)
    return (Y_ij, J_ij)


class CrowdSourcedAnnotationModel(object):
    def __init__(
        self,
        expected_correctness: float,
        concentration: float,
        num_categories: int,
        num_labelers: int,
        num_items: int,
        Y_ij: List,
        J_ij: List,
        iterations: int,
        inference_type: str,
    ):
        alphas = (1 - expected_correctness) / (num_categories - 1)
        alpha = torch.ones((num_categories, num_categories)) * alphas * concentration
        alpha[torch.arange(num_categories), torch.arange(num_categories)] = (
            expected_correctness * concentration
        )
        beta = torch.ones(num_categories) * (1 / num_categories)
        self.num_categories = num_categories
        self.num_labelers = num_labelers
        self.num_items = num_items
        self.Y_ij = Y_ij
        self.J_ij = J_ij
        self.iterations = iterations
        self.inference_type = inference_type
        self.alpha = alpha
        self.beta = beta

    # probability labeler j will label item with true label k as items 1..K
    # shape (num_labelers, num_categories)
    @sample
    def theta(self, j: int, k: int):
        return dist.Dirichlet(self.alpha[k])

    # prevalence of category classes
    @sample
    def pi(self):
        return dist.Dirichlet(self.beta)

    # true label for item i
    @sample
    def z(self, i: int):
        return dist.Categorical(self.pi())

    # this is observed data. Label given to item i by labeler j
    @sample
    def y(self, i: int, j: int):
        return dist.Categorical(self.theta(j, self.z(i).item()))

    def infer(self) -> Tuple[MonteCarloSamples, float]:
        observed_dict = {}
        start_time = 0
        samples = {}
        for i in range(self.num_items):
            for j in range(len(self.J_ij[i])):
                observed_dict[self.y(i, self.J_ij[i][j])] = tensor(self.Y_ij[i][j])

        if self.inference_type == "mcmc":
            mh = SingleSiteCompositionalInference()
            start_time = time.time()
            samples = mh.infer(
                [self.pi()]
                + [
                    self.theta(j, k)
                    for j in range(self.num_labelers)
                    for k in range(self.num_categories)
                ],
                observed_dict,
                self.iterations,
                1,
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
    Y_ij, J_ij = format_data(vector_y, vector_J_i, num_labels)
    num_labelers = int(args_dict["k"])
    num_items = len(num_labels)
    num_categories, labeler_rate, expected_correctness, concentration = args_dict[
        "model_args"
    ]
    iterations = args_dict["num_samples_beanmachine"]
    inference_type = args_dict["inference_type"]

    crowdSourcedAnnotationModel = CrowdSourcedAnnotationModel(
        expected_correctness,
        concentration,
        num_categories,
        num_labelers,
        num_items,
        Y_ij,
        J_ij,
        iterations,
        inference_type,
    )
    start_time = time.time()
    samples, timing_info = crowdSourcedAnnotationModel.infer()
    elapsed_time_beanmachine = time.time() - start_time

    # repackage samples into shape required by PPLBench
    samples_formatted = []
    for i in range(args_dict["num_samples_beanmachine"]):
        sample_dict = {}
        sample_dict["pi"] = (
            samples.get_variable(crowdSourcedAnnotationModel.pi())[i].detach().numpy()
        )
        theta = np.zeros((num_labelers, num_categories, num_categories))
        for j in range(num_labelers):
            for k in range(num_categories):
                theta[j, k, :] = (
                    samples.get_variable(crowdSourcedAnnotationModel.theta(j, k))[i]
                    .detach()
                    .numpy()
                )
        sample_dict["theta"] = theta
        samples_formatted.append(sample_dict)
    timing_info = {"compile_time": 0, "inference_time": elapsed_time_beanmachine}

    return (samples_formatted, timing_info)
