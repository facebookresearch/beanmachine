# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import List

import numpy as np
import pandas as pd
import torch
from beanmachine.applications.clara.nmc_df import LabelingErrorBMModel


torch.manual_seed(123)


# this function copied from pplbench/models/crowdSourcedAnnotationModel.py
def generate_data(pi: List[float]):
    np.random.seed(1234)
    J = 10
    n_items = 500

    K = 2
    concentration, expected_correctness = 10, 0.75

    # choose a true class z for each item
    z = np.random.choice(range(K), p=pi, size=n_items)  # true label , shape [I]
    # set prior that each labeler on average has 50% chance of getting true label
    alpha = ((1 - expected_correctness) / (K - 1)) * np.ones([K, K]) + (
        expected_correctness - (1 - expected_correctness) / (K - 1)
    ) * np.eye(K)
    alpha *= concentration
    # sample confusion matrices theta for labelers from this dirichlet prior
    theta = np.zeros([J, K, K])
    for j in range(J):
        for k in range(K):
            # theta_jk ~ Dirichlet(alpha_k)
            theta[j, k] = np.random.dirichlet(alpha[k])

    # select labelers for each item, get their labels for that item
    J_i = []
    y = []
    i = 0
    num_labels = []
    while i < n_items:
        num_j = np.random.poisson(1)
        # check if we sampled 0, if yes redo this loop
        if num_j == 0:
            continue
        cur_labelers = []
        cur_labelers = np.random.choice(J, size=num_j, replace=False)
        y_i = []
        for j in cur_labelers:
            y_i.append(np.random.choice(range(K), p=theta[j, z[i]]))
        i += 1
        J_i.extend(cur_labelers)
        y.extend(y_i)
        num_labels.append(num_j)

    items = []
    row = {"labelers": J_i, "ratings": y, "num_labels": num_labels}
    items.append(row)
    return pd.DataFrame(items)


class TestNMC(unittest.TestCase):
    def test_prevalence_CI(self):
        pi = [0.16, 1 - 0.16]
        item_df = generate_data(pi)
        model = LabelingErrorBMModel(burn_in=1000, num_samples=1000)
        output = model.fit(item_df)

        for g in range(2):
            self.assertAlmostEqual(
                output.theta_cis[0][g][0], pi[g], msg=f"lb for category {g}", delta=0.1
            )
            self.assertAlmostEqual(
                output.theta_cis[0][g][1], pi[g], msg=f"ub for category {g}", delta=0.1
            )
