# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# For now supports only ordered numeric variables
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from beanmachine.ppl.experimental.causal_inference.models.bart.exceptions import (
    NotInitializedError,
)
from beanmachine.ppl.experimental.causal_inference.models.bart.grow_prune_tree_proposer import (
    GrowPruneTreeProposer,
)
from beanmachine.ppl.experimental.causal_inference.models.bart.node import LeafNode
from beanmachine.ppl.experimental.causal_inference.models.bart.split_rule import (
    CompositeRules,
)
from beanmachine.ppl.experimental.causal_inference.models.bart.tree import Tree
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal


class BART:
    """Bayesian Additive Regression Trees (BART) are Bayesian sum of trees models [1] Default parameters are taken from [1].

    Reference:
        [1] Hugh A. Chipman, Edward I. George, Robert E. McCulloch (2010). "BART: Bayesian additive regression trees"
        https://projecteuclid.org/journals/annals-of-applied-statistics/volume-4/issue-1/BART-Bayesian-additive-regression-trees/10.1214/09-AOAS285.full

    Args:
        num_trees: Number of trees.
        alpha: Parameter used in the tree depth prior, Eq. 7 of [1].
        beta: Parameter used in the tree depth prior, Eq. 7 of [1].
        k: Parameter used in the u_i_j prior, Eq. 8 of [1].
        sigma_concentration: Concentration parameter (alpha) for the inverse gamma distribution prior of p(sigma).
        sigma_rate: Rate parameter (beta) for the inverse gamma distribution prior of p(sigma).
        num_burn: Number of samples burned-in.
        tree_sampler: The tree sampling method used.
        num_sample: Number of samples to collect.
        p_grow: Probability of tree growth. Used by the tree sampler.
        random_state: Random state used to seed.


    """

    def __init__(
        self,
        num_trees: int = 200,
        alpha: float = 0.95,
        beta: float = 2.0,
        k: float = 2.0,
        noise_sd_concentration: float = 3.0,
        noise_sd_rate: float = 1.0,
        tree_sampler: Optional[GrowPruneTreeProposer] = None,
        random_state: Optional[int] = None,
    ):

        self.num_trees = num_trees
        self.num_samples = None
        self.all_trees = []
        self.all_tree_predictions = None
        self.k = k
        self.leaf_mean = LeafMean(
            prior_loc=0.0, prior_scale=0.5 / (self.k * math.sqrt(self.num_trees))
        )
        self.alpha = alpha
        self.beta = beta

        if noise_sd_concentration <= 0 or noise_sd_rate <= 0:
            raise ValueError("Invalid specification of noise_sd distribution priors")
        self.noise_sd_concentration = noise_sd_concentration
        self.noise_sd_rate = noise_sd_rate
        self.sigma = NoiseStandardDeviation(
            prior_concentration=self.noise_sd_concentration,
            prior_rate=self.noise_sd_rate,
        )
        self.samples = None
        self.X = None
        self.y = None
        self.y_min = None
        self.y_max = None
        if random_state is not None:
            torch.manual_seed(random_state)
        if tree_sampler is None:
            self.tree_sampler = GrowPruneTreeProposer(grow_probability=0.5)
        elif isinstance(tree_sampler, GrowPruneTreeProposer):
            self.tree_sampler = tree_sampler
        else:
            NotImplementedError("tree_sampler not implemented")

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        num_samples: int = 1000,
        num_burn: int = 250,
    ) -> BART:
        """Fit the training data and learn the parameters of the model.

        Args:
            X: Training data / covariate matrix of shape (num_samples, input_dimensions).
            y: Response vector of shape (num_samples, 1).
        """
        self.num_samples = num_samples
        self._load_data(X, y)

        self.samples = {"trees": [], "sigmas": []}
        self._init_trees(X)

        for iter_id in range(num_burn + num_samples):
            print(f"Sampling iteration {iter_id}")
            trees, sigma = self._step()
            if iter_id >= num_burn:
                self.samples["trees"].append(trees)
                self.samples["sigmas"].append(sigma)
        return self

    def _load_data(self, X: torch.Tensor, y: torch.Tensor):
        """
        Load the training data. The response is scaled to [-1, 1] as per [1].

        Reference:
            [1] Hugh A. Chipman, Edward I. George, Robert E. McCulloch (2010). "BART: Bayesian additive regression trees"
        https://projecteuclid.org/journals/annals-of-applied-statistics/volume-4/issue-1/BART-Bayesian-additive-regression-trees/10.1214/09-AOAS285.full


        Args:
            X: Training data / covariate matrix of shape (num_samples, input_dimensions).
            y: Response vector of shape (num_samples, 1).

        """
        if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise ValueError("Expected type torch.Tensor")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Number of samples in X {X.shape[0]} not the same as in y {y.shape[0]}"
            )
        self.X = X
        self.y_min = y.min()
        self.y_max = y.max()
        self.y = self._scale(y).reshape(-1, 1)

    def _scale(self, y: torch.Tensor) -> torch.Tensor:
        """
        Scale tensor to [-1. ,1.]

        Args:
            y: Input tensor.
        """
        max_ = torch.ones_like(y)
        min_ = -torch.ones_like(y)
        y_std = (y - self.y_min) / (self.y_max - self.y_min)
        return y_std * (max_ - min_) + min_

    def _inverse_scale(self, y: torch.Tensor) -> torch.Tensor:
        """
        Rescale tensor back from [-1. ,1.].

        Args:
            y: Input tensor.
        """
        max_ = torch.ones_like(y)
        min_ = -torch.ones_like(y)
        y_std = (y - min_) / (max_ - min_)
        return y_std * (self.y_max - self.y_min) + self.y_min

    def _init_trees(self, X: torch.Tensor):
        """
        Initialize the trees of the model.

        Args:
            X: Training data / covariate matrix of shape (num_samples, input_dimensions).
        """
        num_dims = X.shape[-1]
        num_points = X.shape[0]
        for _ in range(self.num_trees):
            self.all_trees.append(
                Tree(
                    nodes=[
                        LeafNode(
                            val=self.leaf_mean.sample_prior(),
                            composite_rules=CompositeRules(
                                all_dims=list(range(num_dims))
                            ),
                            depth=0,
                        )
                    ]
                )
            )
        self.all_tree_predictions = torch.zeros(
            (num_points, self.num_trees, 1), dtype=torch.float
        )

    def _step(self) -> Tuple[List, float]:
        """Take a single MCMC step"""
        if self.X is None or self.y is None:
            raise NotInitializedError("No training data")

        for tree_id in range(len(self.all_trees)):
            partial_residual = (
                self.y - self._predict_step() + self.all_tree_predictions[:, tree_id]
            )
            self.all_trees[tree_id] = self.tree_sampler.propose(
                tree=self.all_trees[tree_id],
                X=self.X,
                partial_residual=partial_residual,
                alpha=self.alpha,
                beta=self.beta,
                sigma_val=self.sigma.val,
                leaf_mean_prior_scale=self.leaf_mean_prior_scale,
            )
            self._update_leaf_mean(self.all_trees[tree_id], partial_residual)
            self.all_tree_predictions[:, tree_id] = self.all_trees[tree_id].predict(
                self.X
            )
        self._update_sigma(self.y - self._predict_step())
        return self.all_trees, self.sigma.val

    def _update_leaf_mean(self, tree: Tree, partial_residual: torch.Tensor):
        """
        Use Eq. 2.10 of [1] to update leaf node values by sampling from posterior distribution.

        Reference:
            [1] Andrew Gelman et al. "Bayesian Data Analysis", 3rd ed.

        Args:
            tree: Tree whos leaf is being updated.
            partial_residual: Current residual of the model excluding this tree of shape (num_samples, 1).

        """
        if self.X is None:
            raise NotInitializedError("No training data")
        for leaf_node in tree.leaf_nodes():
            new_leaf_val = self.leaf_mean.sample_posterior(
                node=leaf_node,
                X=self.X,
                y=partial_residual,
                current_sigma_val=self.sigma.val,
            )
            if new_leaf_val is not None:
                leaf_node.val = new_leaf_val

    def _update_sigma(self, full_residual: torch.Tensor):
        """
        Use Eq. from section 2.6 of [1] to update sigma by sampling from posterior distribution.

        Reference:
            [1] Andrew Gelman et al. "Bayesian Data Analysis", 3rd ed.

        Args:
            partial_residual: Current residual of the model excluding this tree of shape (num_samples, 1).

        """
        self.sigma.sample(self.X, full_residual)

    def _predict_step(
        self,
        X: Optional[torch.Tensor] = None,
        trees: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Get a prediction from a list of trees.

        Args:
            X: Covariate matrix to predict on. If None provided, predictions are made on the  training set of shape (num_samples, input_dimensions).
            trees: Trees to perform prediction. The prediction is the sum of predictions from these trees.
                If None provided, the last drawn sample of trees is used for prediction.

        Returns:
            prediction: Prediction of shape (num_samples, 1).
        """

        if self.X is None or self.all_trees is None:
            raise NotInitializedError("Model not trained")
        if X is None:
            X = self.X
        if trees is None:
            trees = self.all_trees

        prediction = torch.zeros((len(X), 1), dtype=torch.float)
        for single_tree in trees:
            prediction += single_tree.predict(self.X)
        return prediction

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Perform a prediction using all the samples collected in the model.

        Args:
            X: Covariate matrix to predict on of shape (num_samples, input_dimensions).

        Returns:
            prediction: Prediction corresponding to averaf of all samples of shape (num_samples, 1).
        """

        prediction = torch.zeros((len(X), 1), dtype=torch.float)
        for sample_id in range(self.num_samples):
            unscaled_prediction = self._predict_step(
                X=X, trees=self.samples["trees"][sample_id]
            )
            prediction += self._inverse_scale(unscaled_prediction)
        return prediction / self.num_samples

    @property
    def leaf_mean_prior_scale(self):
        if self.leaf_mean is None:
            raise NotInitializedError("LeafMean prior not set.")
        return self.leaf_mean.prior_scale


class NoiseStandardDeviation:
    """The NoiseStandardDeviation class encapsulates the noise standard deviation.
    The variance is parametrized by an inverse-gamma prior which is conjugate to a normal likelihood.

    Args:
        prior_concentration (float): Also called alpha. Must be greater than zero.
        prior_rate (float): Also called beta. Must be greater than 0.
        val (float): Current value of noise standard deviation.
    """

    def __init__(
        self, prior_concentration: float, prior_rate: float, val: Optional[float] = None
    ):
        if prior_concentration <= 0 or prior_rate <= 0:
            raise ValueError("Invalid prior hyperparameters")
        self.prior_concentration = prior_concentration
        self.prior_rate = prior_rate
        if val is None:
            self.sample(X=torch.Tensor([]), residual=torch.Tensor([]))  # prior init
        else:
            self._val = val

    @property
    def val(self) -> float:
        return self._val

    @val.setter
    def val(self, val: float):
        self._val = val

    def sample(self, X: torch.Tensor, residual: torch.Tensor) -> float:
        """Sample from the posterior distribution of sigma.
        If empty tensors are passed for X and residual, there will be no update so the sampling will be from the prior.

        Note:
            This sets the value of the `val` attribute to the drawn sample.

        Args:
            X: Covariate matrix / training data shape (num_samples, input_dimensions).
            residual: The current residual of the model shape (num_samples, 1).
        """
        self.val = self._get_sample(X, residual)
        return self.val

    def _get_sample(self, X: torch.Tensor, residual: torch.Tensor) -> float:
        """
        Draw a sample from the posterior.

        Args:
            X: Covariate matrix / training data of shape (num_samples, input_dimensions).
            residual: The current residual of the model of shape (num_samples, 1).

        """
        posterior_concentration = self.prior_concentration + (len(X) / 2.0)
        posterior_rate = self.prior_rate + (0.5 * (torch.sum(torch.square(residual))))
        draw = torch.pow(Gamma(posterior_concentration, posterior_rate).sample(), -0.5)
        return draw.item()


class LeafMean:
    """
    Class to sample form the prior and posterior distributions of the leaf nodes in BART.

    Reference:
        [1] Hugh A. Chipman, Edward I. George, Robert E. McCulloch (2010). "BART: Bayesian additive regression trees"
        https://projecteuclid.org/journals/annals-of-applied-statistics/volume-4/issue-1/BART-Bayesian-additive-regression-trees/10.1214/09-AOAS285.full

    Args:
        prior_loc: Prior location parameter.
        prior_scale: Prior scale parameter.
    """

    def __init__(self, prior_loc: float, prior_scale: float):
        if prior_scale < 0:
            raise ValueError("Invalid prior hyperparameters")
        self._prior_loc = prior_loc
        self._prior_scale = prior_scale

    @property
    def prior_scale(self):
        return self._prior_scale

    def sample_prior(self):
        return Normal(loc=self._prior_loc, scale=self._prior_scale).sample().item()

    def sample_posterior(
        self,
        node: LeafNode,
        X: torch.Tensor,
        y: torch.Tensor,
        current_sigma_val: float,
    ):
        X_in_node, y_in_node = node.data_in_node(X, y)
        if len(X_in_node) == 0:
            return None  # no new data
        num_points_in_node = len(X_in_node)
        prior_variance = (self._prior_scale) ** 2
        likelihood_variance = (current_sigma_val**2) / num_points_in_node
        likelihood_mean = torch.sum(y_in_node) / num_points_in_node
        posterior_variance = 1.0 / (1.0 / prior_variance + 1.0 / likelihood_variance)
        posterior_mean = (
            likelihood_mean * prior_variance + self._prior_loc * likelihood_variance
        ) / (likelihood_variance + prior_variance)
        return (
            Normal(loc=posterior_mean, scale=math.sqrt(posterior_variance))
            .sample()
            .item()
        )
