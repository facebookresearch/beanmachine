# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import Counter
from math import log
from typing import cast, List, NamedTuple, Optional, Tuple

import torch

from beanmachine.ppl.experimental.causal_inference.models.bart.node import LeafNode

from beanmachine.ppl.experimental.causal_inference.models.bart.scalar_samplers import (
    LeafMean,
)
from beanmachine.ppl.experimental.causal_inference.models.bart.split_rule import (
    Operator,
    SplitRule,
)
from beanmachine.ppl.experimental.causal_inference.models.bart.tree import Tree

from torch import multinomial


class CutPoint(NamedTuple):
    dim: int
    cut_val: float


class SortedInvariants(NamedTuple):
    O_: torch.Tensor
    uniq_vals: List[List[float]]
    val_counts: List[Counter]


class GrowFromRootTreeProposer:
    """
    Implements the "Grow-from-root" backfitting algorithm as described in [1].

    Reference:
        [1] He J., Yalov S., Hahn P.R. (2018). "XBART: Accelerated Bayesian Additive Regression Trees"
        https://arxiv.org/abs/1810.02215

    """

    def __init__(self):
        self.num_cuts = None
        self.num_null_cuts = None

    def propose(
        self,
        X: torch.Tensor,
        partial_residual: torch.Tensor,
        m: int,
        w: torch.Tensor,
        sigma_val: float,
        leaf_sampler: LeafMean,
        alpha: float,
        beta: float,
        root_node: LeafNode,
        num_cuts: int,
        num_null_cuts: int,
    ) -> Tuple[Tree, torch.Tensor]:
        """
        Propose a new tree and modified Dirichlet weights based on the grow-from-root algorithm [1].

        Reference:
            [1] He J., Yalov S., Hahn P.R. (2018). "XBART: Accelerated Bayesian Additive Regression Trees"
        https://arxiv.org/abs/1810.02215

        Args:
            X: Training data / covariate matrix of shape (num_observations, input_dimensions).
            partial_residual: Residual vector of shape (num_observations, 1).
            m: Number of input dimensions / variables to sample. This is usually a subset of the total number of input dimensions in the input data.
            w: Vector of weights or probabilities of picking an input dimension.
            sigma_val: Current value of noise staqndard deviation.
            leaf_sampler: A sampler to sample the posterior distribution of leaf means.
            alpha: Hyperparameter controlling the prior probability of a node being terminal as discussed in [1].
            beta: Hyperparameter controlling the prior probability of a node being terminal as discussed in [1].
            root_node: Root of the tree to grow.
            num_cuts: Number of cuts to make along each dimensions.
            num_null_cuts: Weighting given to the no-split cut along each dimension as discussed in [1].

        """
        if num_cuts <= 0:
            raise ValueError("num_cuts has to be nonnegative")
        self.num_cuts = num_cuts
        if num_null_cuts <= 0 or num_null_cuts >= num_cuts:
            raise ValueError(
                "num_null_cuts has to be greater than or equal to 1 and lesser than total number of cuts"
            )
        self.num_null_cuts = num_null_cuts

        O_ = self._presort(X)
        uniq_vals, val_counts = self._get_uniq_elems(X, O_)
        root_invariants = SortedInvariants(
            O_=O_, uniq_vals=uniq_vals, val_counts=val_counts
        )
        all_leaf_nodes = []
        variable_counts = [0 for _ in range(X.shape[-1])]

        self._grow_from_root(
            current_node=root_node,
            X=X,
            partial_residual=partial_residual,
            invariants=root_invariants,
            m=m,
            w=w,
            sigma_val=sigma_val,
            leaf_sampler=leaf_sampler,
            alpha=alpha,
            beta=beta,
            all_leaf_nodes=all_leaf_nodes,
            variable_counts=variable_counts,
        )

        out_tree = Tree(nodes=all_leaf_nodes)

        return out_tree, torch.Tensor(variable_counts)

    def _presort(self, X: torch.Tensor) -> torch.Tensor:
        """
        Presort the input data to generate the O matrix as discussed in section 3.2 [1].

        Reference:
            [1] He J., Yalov S., Hahn P.R. (2018). "XBART: Accelerated Bayesian Additive Regression Trees"
        https://arxiv.org/abs/1810.02215

        Args:
            X: Training data / covariate matrix of shape (num_observations, input_dimensions).

        """
        num_observations, num_dims = X.shape
        O_ = torch.sort(X, 0)[-1]
        return torch.transpose(O_, dim0=0, dim1=1)

    def _get_uniq_elems(self, X: torch.Tensor, O_: torch.Tensor) -> Tuple[list, list]:
        """
        Get the unique values along every input dimension and the counts for each unique value.

        Args:
            X: Training data / covariate matrix of shape (num_observations, input_dimensions).
            O_: Index matrix of shape (input_dimensions, num_observations) contained the indexes
                of input data sorted along each dimension.

        """
        num_dims, num_observations = O_.shape
        uniq_vals = []
        val_counts = []
        for inp_dim in range(num_dims):
            dim_uniq_vals = []
            value_counter = Counter()
            for obs in range(num_observations):
                current_val = X[O_[inp_dim, obs], inp_dim].item()
                if obs == 0 or (current_val > X[O_[inp_dim, obs - 1], inp_dim]):
                    dim_uniq_vals.append(current_val)
                value_counter[current_val] += 1
            uniq_vals.append(dim_uniq_vals)
            val_counts.append(value_counter)
        return uniq_vals, val_counts

    def _grow_from_root(
        self,
        current_node: LeafNode,
        X: torch.Tensor,
        partial_residual: torch.Tensor,
        invariants: SortedInvariants,
        m: int,
        w: torch.Tensor,
        sigma_val: float,
        leaf_sampler: LeafMean,
        alpha: float,
        beta: float,
        all_leaf_nodes: List[LeafNode],
        variable_counts: List[int],
    ):
        """
        Implement the recursive grow-from-root strategy proposed in [1].

        Reference:
            [1] He J., Yalov S., Hahn P.R. (2018). "XBART: Accelerated Bayesian Additive Regression Trees"
        https://arxiv.org/abs/1810.02215

        Args:
            current_node: The node being mutated.
            X: Training data / covariate matrix of shape (num_observations, input_dimensions).
            partial_residual: Residual vector of shape (num_observations, 1).
            invariants: The sorted index matrix and unique values and unique counts used to maintain sorted order.
            m: Number of input dimensions / variables to sample. This is usually a subset of the total number of input dimensions in the input data.
            w: Vector of weights or probabilities of picking an input dimension.
            sigma_val: Current value of noise staqndard deviation.
            leaf_sampler: A sampler to sample the posterior distribution of leaf means.
            alpha: Hyperparameter controlling the prior probability of a node being terminal as discussed in [1].
            beta: Hyperparameter controlling the prior probability of a node being terminal as discussed in [1].
            all_leaf_nodes: All the laf nodes of the grown tree.
            variable_counts: The number of time each input dimensions / variable has been split while growing this tree.

        """
        dims_to_sample = self._sample_variables(m=m, w=w)
        cut_points = self._select_cutpoints(
            candidate_dims=dims_to_sample, uniq_vals=invariants.uniq_vals
        )
        sampled_cut_point = self._sample_cut_point(
            candidate_cut_points=cut_points,
            invariants=invariants,
            partial_residual=partial_residual,
            sigma_val=sigma_val,
            leaf_sampler=leaf_sampler,
            current_node=current_node,
            alpha=alpha,
            beta=beta,
        )
        if sampled_cut_point is None:
            current_node.val = leaf_sampler.sample_posterior(
                X=X, y=partial_residual, current_sigma_val=sigma_val, node=current_node
            )
            all_leaf_nodes.append(current_node)
            return

        variable_counts[sampled_cut_point.dim] += 1
        left_rule, right_rule = SplitRule(
            grow_dim=sampled_cut_point.dim,
            grow_val=sampled_cut_point.cut_val,
            operator=Operator.le,
        ), SplitRule(
            grow_dim=sampled_cut_point.dim,
            grow_val=sampled_cut_point.cut_val,
            operator=Operator.gt,
        )
        new_node = LeafNode.grow_node(
            current_node, left_rule=left_rule, right_rule=right_rule
        )
        left_invariants, right_invariants = self._sift(
            X=X, cut_point=sampled_cut_point, invariants=invariants
        )

        self._grow_from_root(
            current_node=cast(LeafNode, new_node.left_child),
            X=X,
            partial_residual=partial_residual,
            invariants=left_invariants,
            m=m,
            w=w,
            sigma_val=sigma_val,
            leaf_sampler=leaf_sampler,
            alpha=alpha,
            beta=beta,
            all_leaf_nodes=all_leaf_nodes,
            variable_counts=variable_counts,
        )

        self._grow_from_root(
            current_node=cast(LeafNode, new_node.right_child),
            X=X,
            partial_residual=partial_residual,
            invariants=right_invariants,
            m=m,
            w=w,
            sigma_val=sigma_val,
            leaf_sampler=leaf_sampler,
            alpha=alpha,
            beta=beta,
            all_leaf_nodes=all_leaf_nodes,
            variable_counts=variable_counts,
        )

    def _sample_variables(self, m: int, w: torch.Tensor) -> List[int]:
        """
        Sample a subset of input dimensions to split on as discussed in section 3.4 of [1].

        Reference:
            [1] He J., Yalov S., Hahn P.R. (2018). "XBART: Accelerated Bayesian Additive Regression Trees"
        https://arxiv.org/abs/1810.02215.

        Note:
        The number of sampled variables are set to min(m, count_nonzero(w)).

        Args:
            m: number of dimensions to sample, corresponding to 'm' in [1].
            w: Vector of weights of picking an input dimension.

        """
        m = cast(int, min(m, torch.count_nonzero(w).item()))
        return [
            _.item() for _ in multinomial(input=w, num_samples=m, replacement=False)
        ]

    def _select_cutpoints(
        self,
        candidate_dims: List[int],
        uniq_vals: List[List[float]],
    ) -> List[CutPoint]:
        """
        Select cutpoints along every dimension.

        Args:
            candidate_dims: Dimensions that are being split along.
            uniq_vals: Unique values along every dimension.

        """
        candidate_cuts = []
        for inp_dim in candidate_dims:
            # check for degeneracy
            if len(uniq_vals[inp_dim]) < 2:
                continue
            if len(uniq_vals[inp_dim]) <= self.num_cuts:
                skip_val_freq = 1
            elif self.num_cuts == 1:
                skip_val_freq = len(
                    uniq_vals[inp_dim]
                )  # just select the first val if only 1 cut required
            else:
                skip_val_freq = math.floor(
                    (len(uniq_vals[inp_dim]) - 2) / (self.num_cuts - 1)
                )
            curr_id = 0
            # all uniq vals except last get added to the bag
            while curr_id < (len(uniq_vals[inp_dim]) - 1):
                candidate_cuts.append(
                    CutPoint(dim=inp_dim, cut_val=uniq_vals[inp_dim][curr_id])
                )
                curr_id += skip_val_freq
        return candidate_cuts

    def _sample_cut_point(
        self,
        candidate_cut_points: List[CutPoint],
        partial_residual: torch.Tensor,
        invariants: SortedInvariants,
        sigma_val: float,
        leaf_sampler: LeafMean,
        current_node: LeafNode,
        alpha: float,
        beta: float,
    ) -> Optional[CutPoint]:
        """
        Select a sample cut point by using sampling probabilities calculated in eq. (4) of [1].

        Args:
            candidate_cut_points: DCut points to sample from.
            partial_residual: Residual vector of shape (num_observations, 1).
            invariants: The sorted index matrix and unique values and unique counts used to maintain sorted order.
            sigma_val: Current value of noise standard deviation.
            leaf_sampler: A sampler to sample the posterior distribution of leaf means.
            current_node: The node being mutated.
            alpha: Hyperparameter controlling the prior probability of a node being terminal as discussed in [1].
            beta: Hyperparameter controlling the prior probability of a node being terminal as discussed in [1].

        """
        if len(candidate_cut_points) == 0:
            return None
        selection_log_likelihoods = []
        selection_probabs = []
        total_num_observations = invariants.O_.shape[-1]
        total_residual = torch.sum(partial_residual[invariants.O_[0]]).item()
        tau = leaf_sampler.prior_scale**2
        sigma2 = sigma_val**2
        MAX_LOG_LIKELIHOOD = -float("inf")

        def _integrated_log_likelihood(
            num_observations: int,
            residual: float,
        ) -> float:

            log_likelihood = +0.5 * log(
                (sigma2) / (sigma2 + tau * num_observations)
            ) + 0.5 * (tau * (residual**2)) / (
                (sigma2) * (sigma2 + tau * num_observations)
            )
            return log_likelihood

        kappa = self.num_null_cuts * (
            (math.pow((1 + current_node.depth), beta) / alpha) - 1
        )

        null_log_likelihood = _integrated_log_likelihood(
            num_observations=total_num_observations, residual=total_residual
        ) + log(kappa)
        null_log_likelihood += log(len(candidate_cut_points))
        if null_log_likelihood > MAX_LOG_LIKELIHOOD:
            MAX_LOG_LIKELIHOOD = null_log_likelihood

        selection_log_likelihoods.append(null_log_likelihood)

        current_O_id_, current_uniq_val_id_ = 0, 0
        residuals_le_cutpoint, num_obs_le_cutpoint = [], []
        for cut_id, cut_point in enumerate(candidate_cut_points):
            current_residual = 0.0
            current_num_obs = 0

            if cut_id == 0 or cut_point.dim != candidate_cut_points[cut_id - 1].dim:
                residuals_le_cutpoint = []
                num_obs_le_cutpoint = []
                current_O_id_ = 0
                current_uniq_val_id_ = 0
            else:
                current_residual += residuals_le_cutpoint[-1]
                current_num_obs += num_obs_le_cutpoint[-1]

            while (
                invariants.uniq_vals[cut_point.dim][current_uniq_val_id_]
                <= cut_point.cut_val
            ):
                num_ties = invariants.val_counts[cut_point.dim][
                    invariants.uniq_vals[cut_point.dim][current_uniq_val_id_]
                ]
                current_num_obs += num_ties
                for _ in range(num_ties):
                    current_residual += partial_residual[
                        invariants.O_[cut_point.dim, current_O_id_]
                    ].item()
                    current_O_id_ += 1
                current_uniq_val_id_ += 1
            residuals_le_cutpoint.append(current_residual)
            num_obs_le_cutpoint.append(current_num_obs)
            cut_point_log_likelihood = _integrated_log_likelihood(
                num_observations=current_num_obs,
                residual=current_residual,
            ) + _integrated_log_likelihood(
                num_observations=(total_num_observations - current_num_obs),
                residual=(total_residual - current_residual),
            )
            if cut_point_log_likelihood > MAX_LOG_LIKELIHOOD:
                MAX_LOG_LIKELIHOOD = cut_point_log_likelihood
            selection_log_likelihoods.append(cut_point_log_likelihood)

        # turn it into likelihoods
        sum_ = 0.0
        for log_likelihood in selection_log_likelihoods:
            likelihood = math.exp(log_likelihood - MAX_LOG_LIKELIHOOD)
            sum_ += likelihood
            selection_probabs.append(likelihood)
        selection_probabs = torch.tensor([_ / sum_ for _ in selection_probabs])

        sampled_cut_id = cast(
            int, multinomial(input=selection_probabs, num_samples=1).item()
        )
        if sampled_cut_id == 0:
            # no split
            return None
        return candidate_cut_points[sampled_cut_id - 1]

    def _sift(
        self, X: torch.Tensor, invariants: SortedInvariants, cut_point: CutPoint
    ) -> Tuple[SortedInvariants, SortedInvariants]:
        """
        Sift all data into left and right partitions to maintain sorted order during recursion.

        Args:
            X: Training data / covariate matrix of shape (num_observations, input_dimensions).
            invariants: The sorted index matrix and unique values and unique counts used to maintain sorted order.
            cut_point: The cut point to split along.
        """
        num_dims, num_observations = invariants.O_.shape
        O_left, O_right = [], []
        uniq_vals_left, uniq_vals_right = [], []
        val_counts_left, val_counts_right = [], []

        for dim in range(num_dims):
            dim_O_left, dim_O_right = [], []
            dim_uniq_vals_left, dim_uniq_vals_right = [], []
            dim_val_counts_left, dim_val_counts_right = Counter(), Counter()

            for col in range(num_observations):
                obs_id = invariants.O_[dim, col].item()
                curr_observation_dim_val = X[obs_id, dim].item()

                if X[obs_id, cut_point.dim] <= cut_point.cut_val:
                    dim_O_left.append(obs_id)
                    if (
                        len(dim_uniq_vals_left) == 0
                        or dim_uniq_vals_left[-1] != curr_observation_dim_val
                    ):
                        dim_uniq_vals_left.append(curr_observation_dim_val)
                    dim_val_counts_left[curr_observation_dim_val] += 1

                else:
                    dim_O_right.append(obs_id)
                    if (
                        len(dim_uniq_vals_right) == 0
                        or dim_uniq_vals_right[-1] != curr_observation_dim_val
                    ):
                        dim_uniq_vals_right.append(curr_observation_dim_val)
                    dim_val_counts_right[curr_observation_dim_val] += 1
            O_left.append(dim_O_left)
            O_right.append(dim_O_right)
            uniq_vals_left.append(dim_uniq_vals_left)
            uniq_vals_right.append(dim_uniq_vals_right)
            val_counts_left.append(dim_val_counts_left)
            val_counts_right.append(dim_val_counts_right)
        left_invariants = SortedInvariants(
            O_=torch.tensor(O_left),
            uniq_vals=uniq_vals_left,
            val_counts=val_counts_left,
        )
        right_invariants = SortedInvariants(
            O_=torch.tensor(O_right),
            uniq_vals=uniq_vals_right,
            val_counts=val_counts_right,
        )
        return left_invariants, right_invariants
