# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd

from .nmc import obtain_posterior


logger = logging.getLogger(__name__)


class ModelOutput(NamedTuple):
    theta_samples: np.array
    theta_means: np.array
    theta_cis: np.array
    psi_samples: np.array
    psi_means: np.array
    psi_cis: np.array
    item_samples: np.array
    item_means: np.array
    item_cis: np.array


class LabelingErrorBMModel(object):
    """
    Inference for CLARA's uniform model using an specialized and scalable
    algorithm called nmc
    """

    def __init__(
        self, burn_in: int = 1000, num_samples: int = 1000, ci_coverage: float = 0.95
    ):
        self.burn_in = burn_in
        self.num_samples = num_samples
        self.ub = 100 * (ci_coverage / 2.0 + 0.5)
        self.lb = 100 - self.ub

    # Compute the mean and 95% CI of theta given the samples collected
    def _process_theta_output(self, thetas, n_item_groups, n_unique_ratings):
        theta_means = np.empty((n_item_groups, n_unique_ratings))
        theta_cis = np.empty((n_item_groups, n_unique_ratings, 2))

        for ig in range(n_item_groups):
            mean = np.mean(thetas[:, ig, :], axis=0)
            lb, ub = np.percentile(thetas[:, ig, :], [self.lb, self.ub], axis=0)
            theta_means[ig, :] = mean
            theta_cis[ig, :, 0] = lb
            theta_cis[ig, :, 1] = ub
        return theta_means, theta_cis

    def _process_clara_input_df_for_nmc(self, df: pd.DataFrame):
        labels = df.ratings.values[0]
        labelers = df.labelers.values[0]
        num_labels = df.num_labels.values[0]
        labeler_idx = category_idx = 0
        rating_vocab, labelers_names = {}, {}

        for e in labelers:
            if e not in labelers_names.keys():
                labelers_names[e] = labeler_idx
                labeler_idx += 1
        for e in labels:
            if e not in rating_vocab.keys():
                rating_vocab[e] = category_idx
                category_idx += 1

        concentration, expected_correctness = 10, 0.75
        args_dict = {}
        args_dict["k"] = labeler_idx
        args_dict["num_samples_nmc"] = self.num_samples
        args_dict["burn_in_nmc"] = self.burn_in
        args_dict["model_args"] = (category_idx, 1, expected_correctness, concentration)

        data_train = (labels, labelers, num_labels)
        return (data_train, args_dict, num_labels, category_idx)

    def _process_item_output(
        self, prevalence_samples, labeler_confusion, num_categories, input_df
    ):

        labels = input_df[0].loc[0]
        labelers = input_df[1].loc[0]
        num_labels = input_df[2].loc[0]

        labelers_list, labels_list = [], []

        pos = 0
        for i in range(len(num_labels)):
            item_labelers, item_labels = [], []
            for _j in range(num_labels[i]):
                item_labelers.append(labelers[pos])
                item_labels.append(labels[pos])
                pos += 1
            labelers_list.append(item_labelers)
            labels_list.append(item_labels)

        n_items = len(num_labels)
        n_unique_ratings = num_categories

        item_means = np.empty((n_items, n_unique_ratings))
        item_cis = np.empty((n_items, n_unique_ratings, 2))

        n_samples = len(prevalence_samples)
        # collect sampled probabilities for each item
        item_samples = np.empty((n_samples, n_items, n_unique_ratings))

        for i in range(n_items):
            for s in range(n_samples):
                item_samples[s, i] = prevalence_samples[s]
                for k in range(n_unique_ratings):
                    for j in range(num_labels[i]):
                        item_rating = labels_list[i][j]
                        item_labeler = labelers_list[i][j]
                        item_samples[s, i, k] *= labeler_confusion[s][item_labeler][k][
                            item_rating
                        ]
                item_samples[s, i, :] /= sum(item_samples[s, i, :])

            item_means[i] = np.mean(item_samples[:, i], axis=0)
            lb, ub = np.percentile(item_samples[:, i], [self.lb, self.ub], axis=0)
            item_cis[i, :, 0] = lb
            item_cis[i, :, 1] = ub

        return item_samples, item_means, item_cis

    # Compute the mean and 95% CI of psi given the samples collected
    def _process_psi_output(self, psis, n_labeler_groups, n_unique_ratings):
        psi_means = np.empty((n_labeler_groups, n_unique_ratings, n_unique_ratings))
        psi_cis = np.empty((n_labeler_groups, n_unique_ratings, n_unique_ratings, 2))

        for lg in range(n_labeler_groups):
            s = psis[:, lg]
            mean = np.mean(s, axis=0)
            lb, ub = np.percentile(s, [self.lb, self.ub], axis=0)
            psi_means[lg] = mean
            psi_cis[lg, :, :, 0] = lb
            psi_cis[lg, :, :, 1] = ub
        return psi_means, psi_cis

    # Fit the model using exact inference
    def fit(
        self, df: pd.DataFrame, n_item_groups: int = None, n_labeler_groups: int = None
    ):
        out = self._process_clara_input_df_for_nmc(df)
        data_train, args_dict, num_labels, num_categories = out
        items = []
        items.append(data_train)

        if n_item_groups is None:
            n_item_groups = 1
        if n_labeler_groups is None:
            n_labeler_groups = args_dict["k"]

        logger.info("Fitting using NMC ...")
        # TO DO: incorporate n_item_groups and n_labeler_groups into nmc
        # TO DO: extend nmc to support n_item_groups > 1 and
        # n_labeler_groups != args_dict["k"]
        samples, timing = obtain_posterior(data_train, args_dict)
        logger.info(f"Fitting took {timing['inference_time']} sec")

        samples_df = pd.DataFrame(samples)

        # process output
        logger.info("Processing outputs ...")
        thetas = np.empty((self.num_samples, n_item_groups, num_categories))
        psis = np.empty(
            (self.num_samples, args_dict["k"], num_categories, num_categories)
        )
        for i in range(self.num_samples):
            thetas[i] = samples_df["pi"].values[i]
            psis[i] = samples_df["theta"].values[i]

        theta_means, theta_cis = self._process_theta_output(
            thetas, n_item_groups, num_categories
        )
        psi_means, psi_cis = self._process_psi_output(
            psis, args_dict["k"], num_categories
        )
        item_samples, item_means, item_cis = self._process_item_output(
            thetas, psis, num_categories, pd.DataFrame(items)
        )
        return ModelOutput(
            theta_samples=thetas,
            theta_means=theta_means,
            theta_cis=theta_cis,
            psi_samples=psis,
            psi_means=psi_means,
            psi_cis=psi_cis,
            item_samples=item_samples,
            item_means=item_means,
            item_cis=item_cis,
        )
