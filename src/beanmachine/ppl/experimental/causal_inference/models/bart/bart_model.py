# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# For now supports only ordered numeric variables
from __future__ import annotations

import math

from copy import deepcopy
from typing import cast, List, Optional, Tuple

import torch

from beanmachine.ppl.experimental.causal_inference.models.bart.exceptions import (
    NotInitializedError,
)
from beanmachine.ppl.experimental.causal_inference.models.bart.grow_from_root_tree_proposer import (
    GrowFromRootTreeProposer,
)
from beanmachine.ppl.experimental.causal_inference.models.bart.grow_prune_tree_proposer import (
    GrowPruneTreeProposer,
)
from beanmachine.ppl.experimental.causal_inference.models.bart.node import LeafNode
from beanmachine.ppl.experimental.causal_inference.models.bart.scalar_samplers import (
    LeafMean,
    NoiseStandardDeviation,
)
from beanmachine.ppl.experimental.causal_inference.models.bart.split_rule import (
    CompositeRules,
)
from beanmachine.ppl.experimental.causal_inference.models.bart.tree import Tree
from torch.distributions.dirichlet import Dirichlet
from tqdm.auto import trange


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
        self.all_tree_predictions = None
        self._all_trees = None
        self.leaf_mean = None
        self.k = k
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

        if isinstance(self.tree_sampler, GrowPruneTreeProposer):
            self._step = self._grow_prune_step
        else:
            NotImplementedError(
                "step function not defined"
            )  # this should never be raised

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        num_samples: int = 1000,
        num_burn: int = 250,
    ) -> BART:
        """Fit the training data and learn the parameters of the model.

        Args:
            X: Training data / covariate matrix of shape (num_observations, input_dimensions).
            y: Response vector of shape (num_observations, 1).
        """
        self.num_samples = num_samples
        self._load_data(X, y)

        self.samples = {"trees": [], "sigmas": []}
        self.leaf_mean = LeafMean(
            prior_loc=0.0, prior_scale=0.5 / (self.k * math.sqrt(self.num_trees))
        )
        self._init_trees(X)

        for iter_id in trange(num_burn + num_samples):
            trees, sigma = self._step()
            self._all_trees = trees
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
            X: Training data / covariate matrix of shape (num_observations, input_dimensions).
            y: Response vector of shape (num_observations, 1).

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
            X: Training data / covariate matrix of shape (num_observations, input_dimensions).
        """
        self._all_trees = []
        num_dims = X.shape[-1]
        num_points = X.shape[0]
        for _ in range(self.num_trees):
            self._all_trees.append(
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

    def _grow_prune_step(self) -> Tuple[List, float]:
        """Take a single MCMC step using the GrowPrune approach of the original BART [1].

        Reference:
            [1] Hugh A. Chipman, Edward I. George, Robert E. McCulloch (2010). "BART: Bayesian additive regression trees"
        https://projecteuclid.org/journals/annals-of-applied-statistics/volume-4/issue-1/BART-Bayesian-additive-regression-trees/10.1214/09-AOAS285.full
        """
        if self.X is None or self.y is None:
            raise NotInitializedError("No training data")

        new_trees = [deepcopy(tree) for tree in self._all_trees]
        all_tree_predictions = deepcopy(self.all_tree_predictions)
        for tree_id in range(len(new_trees)):
            # all_tree_predictions.shape -> (num_observations, num_trees, 1)
            current_predictions = torch.sum(all_tree_predictions, dim=1)
            last_iter_tree_prediction = all_tree_predictions[:, tree_id]
            partial_residual = self.y - current_predictions + last_iter_tree_prediction
            new_trees[tree_id] = self.tree_sampler.propose(
                tree=new_trees[tree_id],
                X=self.X,
                partial_residual=partial_residual,
                alpha=self.alpha,
                beta=self.beta,
                sigma_val=self.sigma.val,
                leaf_mean_prior_scale=self.leaf_mean_prior_scale,
            )
            self._update_leaf_mean(new_trees[tree_id], partial_residual)
            all_tree_predictions[:, tree_id] = new_trees[tree_id].predict(self.X)
        self.all_tree_predictions = all_tree_predictions
        self._update_sigma(self.y - torch.sum(all_tree_predictions, dim=1))
        return new_trees, self.sigma.val

    def _update_leaf_mean(self, tree: Tree, partial_residual: torch.Tensor):
        """
        Use Eq. 2.10 of [1] to update leaf node values by sampling from posterior distribution.

        Reference:
            [1] Andrew Gelman et al. "Bayesian Data Analysis", 3rd ed.

        Args:
            tree: Tree whos leaf is being updated.
            partial_residual: Current residual of the model excluding this tree of shape (num_observations, 1).

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
            partial_residual: Current residual of the model excluding this tree of shape (num_observations, 1).

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

        if self.X is None or self._all_trees is None:
            raise NotInitializedError("Model not trained")
        if X is None:
            X = self.X
        if trees is None:
            trees = self._all_trees

        prediction = torch.zeros((len(X), 1), dtype=torch.float)
        for single_tree in trees:
            prediction += single_tree.predict(X)
        return prediction

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Perform a prediction using all the samples collected in the model.

        Args:
            X: Covariate matrix to predict on of shape (num_observations, input_dimensions).

        Returns:
            prediction: Prediction corresponding to average of all samples of shape (num_observations, 1).
        """

        prediction = torch.mean(
            self.get_posterior_predictive_samples(X), dim=-1, dtype=torch.float
        )
        return prediction.reshape(-1, 1)

    def predict_with_quantiles(
        self, X: torch.Tensor, quantiles: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the quantiles of prediction.

        Args:
            X: Covariate matrix to predict on of shape (num_samples, input_dimensions).
            quantiles: The quantiles required. If nothing supplied, the default quantiles are [0.025, 0.5, 0.975]

        Returns:
            prediction, qvals: Prediction corresponding to average of all samples of shape (num_observations, 1),
                qvals tensor of shape (num_obs, len(quantiles)) and qvals[:, i]
                is quantile value corresponding to quantiles[i].

        """
        if quantiles is None:
            quantiles = torch.Tensor([0.025, 0.5, 0.975])
        for q in quantiles:
            if not 0.0 < q < 1.0:
                raise ValueError("Quantiles must be in (0, 1)")
        prediction_samples = self.get_posterior_predictive_samples(X)
        prediction = torch.mean(prediction_samples, dim=-1, dtype=torch.float).reshape(
            -1, 1
        )

        qvals = (
            torch.quantile(prediction_samples, dim=1, q=quantiles)
            .transpose(0, 1)
            .reshape(-1, len(quantiles))
        )
        return prediction, qvals

    def get_posterior_predictive_samples(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns samples from the posterior predictive distribution P(y|X).

        Args:
            X: Covariate matrix to predict on of shape (num_observations, input_dimensions).

        Returns:
            posterior_predictive_samples: Samples from the predictive distribution P(y|X) of shape (num_observations, num_samples).
        """
        posterior_predictive_samples = []
        for sample_id in range(self.num_samples):
            single_prediction_sample = self._inverse_scale(
                self._predict_step(X=X, trees=self.samples["trees"][sample_id])
            )  # ( torch.Size(num_observations, 1) )
            posterior_predictive_samples.append(single_prediction_sample)
        return torch.concat(posterior_predictive_samples, dim=-1)

    @property
    def leaf_mean_prior_scale(self):
        if self.leaf_mean is None:
            raise NotInitializedError("LeafMean prior not set.")
        return self.leaf_mean.prior_scale


class XBART(BART):
    """Implementes XBART [1] which is a faster implementation of Bayesian Additive Regression Trees (BART) are Bayesian sum of trees models [2].
    Default parameters are taken from [1].

    Reference:
        [1] He J., Yalov S., Hahn P.R. (2018). "XBART: Accelerated Bayesian Additive Regression Trees"
        https://arxiv.org/abs/1810.02215

        [2] Hugh A. Chipman, Edward I. George, Robert E. McCulloch (2010). "BART: Bayesian additive regression trees"
        https://projecteuclid.org/journals/annals-of-applied-statistics/volume-4/issue-1/BART-Bayesian-additive-regression-trees/10.1214/09-AOAS285.full

    Args:
        num_trees: Number of trees. If this is not set in the constructor explicitly,
            it defaults to 0 and is adaptively set as a function of the trianing data in the ```fit()``` method.
        alpha: Parameter used in the tree depth prior, Eq. 7 of [2].
        beta: Parameter used in the tree depth prior, Eq. 7 of [2].
        tau: Prior variance of the leaf-specific mean paramete used in the u_i_j prior, section 2.2 of [1].
        noise_sd_concentration: Concentration parameter (alpha) for the inverse gamma distribution prior of p(sigma).
        noise_sd_rate: Rate parameter (beta) for the inverse gamma distribution prior of p(sigma).
        tree_sampler: The tree sampling method used.
        random_state: Random state used to seed.
        num_cuts: The maximum number of cuts per dimension.
        num_null_cuts: Number of "no split" null cuts to consider along each dimension.
            This affects the tree depth as discussed in [1].
        m: Size of the subset of variables that are sampled for cutting points in the post
            burnin period as discussed in section 3.4 of [1].

    """

    def __init__(
        self,
        num_trees: int = 0,
        alpha: float = 0.95,
        beta: float = 2.0,
        tau: Optional[float] = None,
        noise_sd_concentration: float = 3.0,
        noise_sd_rate: float = 1.0,
        tree_sampler: Optional[GrowFromRootTreeProposer] = None,
        random_state: Optional[int] = None,
        num_cuts: Optional[int] = None,
        num_null_cuts: int = 1,
        m: Optional[int] = None,
    ):
        self.num_cuts = num_cuts
        self.num_null_cuts = num_null_cuts
        self.tau = tau
        self.m = m

        super().__init__(
            num_trees=num_trees,
            alpha=0.95,
            beta=1.25,
            noise_sd_concentration=3.0,
            noise_sd_rate=1.0,
            tree_sampler=None,
            random_state=None,
        )

        if tree_sampler is None:
            self.tree_sampler = GrowFromRootTreeProposer()
        elif isinstance(tree_sampler, GrowFromRootTreeProposer):
            self.tree_sampler = tree_sampler
        else:
            raise NotImplementedError("tree_sampler not implemented")
        self._step = self._grow_from_root_step
        self.var_counts = None
        self.all_tree_var_counts = None

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        num_samples: int = 25,
        num_burn: int = 15,
    ) -> XBART:
        """Fit the training data and learn the parameters of the model.

        Reference:
        [1] He J., Yalov S., Hahn P.R. (2018). "XBART: Accelerated Bayesian Additive Regression Trees"
        https://arxiv.org/abs/1810.02215

        Args:
            X: Training data / covariate matrix of shape (num_observations, input_dimensions).
            y: Response vector of shape (num_observations, 1).
            num_samples: Number of post burnin samples to draw.
            num_burn: Number of burnin samples to draw (for adaptation).
        """

        self.num_samples = num_samples
        self._load_data(X, y)

        if not self.num_trees > 0:
            self._adaptively_init_num_trees()

        if self.tau is None:
            self._adaptively_init_tau()
        self.tau = cast(float, self.tau)
        self.leaf_mean = LeafMean(prior_loc=0.0, prior_scale=math.sqrt(self.tau))
        if self.num_cuts is None:
            self._adaptively_init_num_cuts()
        if self.m is None:
            self.m = self.X.shape[-1]

        self.samples = {"trees": []}
        self._init_trees(X)

        self.all_tree_predictions = (
            (torch.clone(self.y) / self.num_trees)
            .unsqueeze(1)
            .tile((1, self.num_trees, 1))
        )
        self.var_counts = torch.ones((X.shape[-1],))
        self.all_tree_var_counts = torch.ones((self.num_trees, X.shape[-1]))

        is_burnin_period = True
        num_dims_to_sample = self.X.shape[-1]
        for iter_id in trange(num_burn + num_samples):
            if iter_id >= num_burn:
                is_burnin_period = False
                num_dims_to_sample = self.m
            trees = self._step(num_dims_to_sample=num_dims_to_sample)
            self._all_trees = trees
            if not is_burnin_period:
                self.samples["trees"].append(trees)
        return self

    def _adaptively_init_num_trees(self):
        """Implements the default for number of trees from section 3.1 of [1].

        Reference:
            [1] He J., Yalov S., Hahn P.R. (2018). "XBART: Accelerated Bayesian Additive Regression Trees"
        https://arxiv.org/abs/1810.02215
        """
        n = len(self.X)
        self.num_trees = int(math.pow(math.log(n), math.log(math.log(n))) / 4)

    def _adaptively_init_tau(self):
        """Implements the default for tau from section 3.1 of [1].

        Reference:
            [1] He J., Yalov S., Hahn P.R. (2018). "XBART: Accelerated Bayesian Additive Regression Trees"
        https://arxiv.org/abs/1810.02215
        """
        if not self.num_trees > 0:
            raise NotInitializedError("num_trees not set")
        self.tau = (3 / 10) * (torch.var(self.y).item() / self.num_trees)

    def _adaptively_init_num_cuts(self):
        """Implements the default for number of cuts, C from section 3.3 of [1].

        Reference:
            [1] He J., Yalov S., Hahn P.R. (2018). "XBART: Accelerated Bayesian Additive Regression Trees"
        https://arxiv.org/abs/1810.02215
        """
        n = len(self.X)
        self.num_cuts = max(math.sqrt(n), 100)

    def _grow_from_root_step(self, num_dims_to_sample: int) -> List[Tree]:
        """Take a single MCMC step using the Grow-from-root approach of xBART [1].

        Reference:
            [1] He J., Yalov S., Hahn P.R. (2018). "XBART: Accelerated Bayesian Additive Regression Trees"
        https://arxiv.org/abs/1810.02215

        Args:
            num_dims_to_sample: Size of the subset of variables that are sampled for cutting points
                as discussed in section 3.4 of [1].
        """
        if self.X is None or self.y is None:
            raise NotInitializedError("No training data")

        all_tree_predictions = deepcopy(self.all_tree_predictions)
        new_trees = []
        for tree_id in range(self.num_trees):
            # all_tree_predictions.shape -> (num_observations, num_trees, 1)
            current_predictions = torch.sum(all_tree_predictions, dim=1)
            last_iter_tree_prediction = all_tree_predictions[:, tree_id]
            partial_residual = self.y - current_predictions + last_iter_tree_prediction

            w = self._draw_var_weights()
            new_tree, new_var_counts = self.tree_sampler.propose(
                X=self.X,
                partial_residual=partial_residual,
                m=num_dims_to_sample,
                w=w,
                alpha=self.alpha,
                beta=self.beta,
                sigma_val=self.sigma.val,
                leaf_sampler=self.leaf_mean,
                root_node=self._get_root_node(),
                num_cuts=self.num_cuts,
                num_null_cuts=self.num_null_cuts,
            )
            new_trees.append(new_tree)
            self.var_counts += new_var_counts - self.all_tree_var_counts[tree_id]
            self.all_tree_var_counts[tree_id] = new_var_counts
            all_tree_predictions[:, tree_id] = new_tree.predict(self.X)
            self._update_sigma(self.y - torch.sum(all_tree_predictions, dim=1))
        self.all_tree_predictions = all_tree_predictions
        return new_trees

    def _draw_var_weights(self) -> torch.Tensor:
        return Dirichlet(self.var_counts).sample()

    def _get_root_node(self):
        return LeafNode(
            depth=0,
            composite_rules=CompositeRules(all_dims=list(range(self.X.shape[-1]))),
        )
