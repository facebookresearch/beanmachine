# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import enum
import math
from typing import Union

import torch

from beanmachine.ppl.experimental.causal_inference.models.bart.exceptions import (
    GrowError,
    PruneError,
    TreeStructureError,
)
from beanmachine.ppl.experimental.causal_inference.models.bart.mutation import (
    GrowMutation,
    Mutation,
    PruneMutation,
)
from beanmachine.ppl.experimental.causal_inference.models.bart.node import (
    LeafNode,
    SplitNode,
)
from beanmachine.ppl.experimental.causal_inference.models.bart.split_rule import (
    Operator,
    SplitRule,
)
from beanmachine.ppl.experimental.causal_inference.models.bart.tree import Tree
from beanmachine.ppl.experimental.causal_inference.models.bart.tree_proposer import (
    TreeProposer,
)
from numpy.random import choice
from torch.distributions.uniform import Uniform


class MutationKind(enum.Enum):
    grow = "grow operation"
    prune = "prune operation"


class GrowPruneTreeProposer(TreeProposer):
    """This implements the Grow Prune tree sampling approach of Pratola [1] where the additional steps of
    BART (Change and Swap) are eliminated for computational efficiency.

    Reference:
        [1] Pratola MT, Chipman H, Higdon D, McCulloch R, Rust W (2013). “Parallel Bayesian Additive Regression Trees.”
        Technical report, University of Chicago.
        https://arxiv.org/pdf/1309.1906.pdf

    Args:
        grow_probability: Probability of growing a node.

    """

    def __init__(self, grow_probability: float = 0.5):
        if grow_probability > 1.0 or grow_probability < 0.0:
            raise ValueError(
                f"Grow probability {grow_probability} not a valid probabiity"
            )
        self.grow_probability = grow_probability
        self.prune_probability = 1 - self.grow_probability
        self._uniform = Uniform(0.0, 1.0)

    def propose(
        self,
        tree: Tree,
        X: torch.Tensor,
        partial_residual: torch.Tensor,
        alpha: float,
        beta: float,
        sigma_val: float,
        leaf_mean_prior_scale: float,
    ) -> Tree:
        """Propose a tree  based on a Metropolis-Hastings step. Refer to [1] for details.

        Reference:
            [1] Adam Kapelner & Justin Bleich (2014). "bartMachine: Machine Learning with Bayesian
                Additive Regression Trees".
                https://arxiv.org/pdf/1312.2171.pdf

        Args:
            tree: Previous tree.
            X: Covariate matrix / training data.
            partial_residual: Partial residual of the current tree model with respect to the training data.
            alpha: Hyperparameter used in tree prior.
            beta: Hyperparameter used in tree prior.
            sigma_val: Current estimate of noise standard deviation in the data.
            leaf_mean_prior_scale: Prior of the scale hyperparameter in the normal distribution of the leaf mean.

        """
        new_mutation = self._get_new_mutation(tree, X)

        # carry out move
        if new_mutation == MutationKind.grow:
            try:
                leaf_to_grow = self._select_leaf_to_grow(tree=tree, X=X)
            except GrowError:
                return self.propose(
                    tree,
                    X,
                    partial_residual,
                    alpha,
                    beta,
                    sigma_val,
                    leaf_mean_prior_scale,
                )
            grow_dim = self._select_grow_dim(leaf_to_grow, X)
            grow_val = self._get_grow_val(
                leaf_to_grow=leaf_to_grow, grow_dim=grow_dim, X=X
            )
            left_rule, right_rule = SplitRule(
                grow_dim=grow_dim, grow_val=grow_val, operator=Operator.le
            ), SplitRule(grow_dim=grow_dim, grow_val=grow_val, operator=Operator.gt)
            mutation = GrowMutation(
                old_node=leaf_to_grow,
                new_node=LeafNode.grow_node(
                    leaf_to_grow, left_rule=left_rule, right_rule=right_rule
                ),
            )
        elif new_mutation == MutationKind.prune:
            try:
                split_node_to_prune = self._select_split_node_to_prune(tree)
            except PruneError:
                return self.propose(
                    tree,
                    X,
                    partial_residual,
                    alpha,
                    beta,
                    sigma_val,
                    leaf_mean_prior_scale,
                )
            mutation = PruneMutation(
                old_node=split_node_to_prune,
                new_node=SplitNode.prune_node(split_node_to_prune),
            )
        else:
            raise TreeStructureError("Can only grow or prune")

        # Metropolis-Hasting step
        log_draw_probability = (
            self._get_log_transition_ratio(
                tree=tree,
                mutation=mutation,
                X=X,
            )
            + self._get_log_likelihood_ratio(
                mutation=mutation,
                X=X,
                partial_residual=partial_residual,
                sigma_val=sigma_val,
                leaf_mean_prior_scale=leaf_mean_prior_scale,
            )
            + self._get_log_structure_ratio(
                mutation=mutation,
                alpha=alpha,
                beta=beta,
                X=X,
            )
        )

        if self._uniform.sample().item() < math.exp(log_draw_probability):
            tree.mutate(mutation)
            return tree
        return tree

    def _get_new_mutation(self, tree: Tree, X: torch.Tensor) -> MutationKind:
        """Get a new mutation.

        Args:
            tree: Previous tree.
            X: Covariate matrix / training data.
        """
        if tree.num_nodes() == 1 or tree.num_prunable_split_nodes() == 0:
            return MutationKind.grow
        if tree.num_growable_leaf_nodes(X) == 0:
            return MutationKind.prune
        if bool(torch.bernoulli(torch.Tensor([self.grow_probability])).item()):
            return MutationKind.grow
        return MutationKind.prune

    def _select_leaf_to_grow(self, tree: Tree, X: torch.Tensor) -> LeafNode:
        """
        Select which leaf to grow.

        Args:
            tree: Previous tree.
            X: Covariate matrix / training data.
        """
        growable_leaf_nodes = tree.growable_leaf_nodes(X)
        if len(growable_leaf_nodes) < 1:
            raise GrowError("Leaf cannot be grown")
        return choice(growable_leaf_nodes)

    def _select_grow_dim(self, leaf_to_grow: LeafNode, X: torch.Tensor) -> int:
        """
        Select an input dimension to grow along.

        Args:
            tree: Previous tree.
            leaf_to_grow: Leaf currently being grown.
            X: Covariate matrix / training data.
        """
        if not leaf_to_grow.is_growable(X):
            raise GrowError("Leaf cannot be grown")
        return choice(leaf_to_grow.get_growable_dims(X))

    def _get_grow_val(
        self, leaf_to_grow: LeafNode, grow_dim: int, X: torch.Tensor
    ) -> float:
        """
        Select a value in the chosen input dimension to grow along.

        Args:
            tree: Previous tree.
            leaf_to_grow: Leaf currently being grown.
            grow_dim: Input dimension to grow along.
            X: Covariate matrix / training data.
        """
        if not leaf_to_grow.is_growable(X):
            raise GrowError("Leaf cannot be grown")
        growable_vals = leaf_to_grow.get_growable_vals(X, grow_dim)
        max_growable_val = torch.max(growable_vals)
        candidate_val = choice(growable_vals)
        degenerate_grow_condition = candidate_val == max_growable_val
        while degenerate_grow_condition:
            return choice(growable_vals)
        return candidate_val

    def _select_split_node_to_prune(self, tree: Tree) -> SplitNode:
        """
        Select and internal node to prune.

        Args:
            tree: Previous tree.
        """
        prunable_split_nodes = tree.prunable_split_nodes()
        if len(prunable_split_nodes) < 1:
            raise PruneError
        return choice(prunable_split_nodes)

    def _get_log_transition_ratio(
        self,
        tree: Tree,
        mutation: Mutation,
        X: torch.Tensor,
    ) -> float:
        """
        Get the log transition ratio as discussed in [1].

        [1] Adam Kapelner & Justin Bleich (2014). "bartMachine: Machine Learning with Bayesian
            Additive Regression Trees".
            https://arxiv.org/pdf/1312.2171.pdf

        Args:
            tree: Previous tree.
            mutation: Proposed mutation,
            X: Covariate matrix / training data.
        """
        if isinstance(mutation, GrowMutation):
            return self._grow_log_transition_ratio(tree=tree, mutation=mutation, X=X)
        elif isinstance(mutation, PruneMutation):
            return self._prune_log_transition_ratio(tree=tree, mutation=mutation, X=X)
        else:
            raise TreeStructureError("Can only grow or prune")

    def _grow_log_transition_ratio(
        self,
        tree: Tree,
        mutation: GrowMutation,
        X: torch.Tensor,
    ) -> float:
        """
        Implement expression for log( P(T -> T*) / P(T* -> T) ) in a GROW move as discussed in eq. 8 of [1]

        Reference:
            [1] Adam Kapelner & Justin Bleich. "bartMachine: Machine Learning with Bayesian Additive Regression Trees" (2013).
            https://arxiv.org/abs/1312.2171

        Args:
            tree: Previous tree.
            mutation: Proposed mutation,
            X: Covariate matrix / training data.

        """
        log_p_new_to_old_tree = math.log(self.prune_probability) - math.log(
            tree.num_prunable_split_nodes() + 1
        )
        log_p_old_to_new_tree = math.log(
            self.grow_probability
        ) + _log_probability_of_growing_a_tree(
            tree=tree,
            mutation=mutation,
            X=X,
        )
        return log_p_new_to_old_tree - log_p_old_to_new_tree

    def _prune_log_transition_ratio(
        self,
        tree: Tree,
        mutation: PruneMutation,
        X: torch.Tensor,
    ) -> float:
        """
        Implement expression for log( P(T -> T*) / P(T* -> T) ) in a PRUNE move as discussed in section A.2 of [1]

        Reference:
            [1] Adam Kapelner & Justin Bleich. "bartMachine: Machine Learning with Bayesian Additive Regression Trees" (2013).
            https://arxiv.org/abs/1312.2171

        Args:
            tree: Previous tree.
            mutation: Proposed mutation,
            X: Covariate matrix / training data.

        """
        num_growable_leaves_in_pruned_tree = tree.num_growable_leaf_nodes(X) - 1

        if num_growable_leaves_in_pruned_tree == 0:
            return -float("inf")  # impossible prune

        log_p_old_to_new_tree = math.log(self.prune_probability) - math.log(
            tree.num_prunable_split_nodes()
        )

        log_probability_selecting_leaf_to_grow = -math.log(
            num_growable_leaves_in_pruned_tree
        )
        log_probability_growing_leaf = _log_probability_of_growing_node(
            mutation=GrowMutation(
                old_node=mutation.new_node, new_node=mutation.old_node
            ),
            X=X,
        )
        log_p_new_to_old_tree = (
            math.log(self.grow_probability)
            + log_probability_selecting_leaf_to_grow
            + log_probability_growing_leaf
        )
        return log_p_new_to_old_tree - log_p_old_to_new_tree

    def _get_log_likelihood_ratio(
        self,
        mutation: Mutation,
        X: torch.Tensor,
        partial_residual: torch.Tensor,
        sigma_val: float,
        leaf_mean_prior_scale: float,
    ) -> float:
        """
        Implement expression for log( P(R | T*, sigma) / P(R | T, sigma) ) in a GROW move as discussed in  [1]

        Reference:
            [1] Adam Kapelner & Justin Bleich. "bartMachine: Machine Learning with Bayesian Additive Regression Trees" (2013).
            https://arxiv.org/abs/1312.2171

        Args:
            tree: Previous tree.
            mutation: Proposed mutation,
            sigma_val:urrent estimate of noise standard deviation in the data.
            leaf_mean_prior_scale: Prior of the scale hyperparameter in the normal distribution of the leaf mean.
            X: Covariate matrix / training data.
            partial_residual: Partial residual of the current tree model with respect to the training data.


        """
        if isinstance(mutation, GrowMutation):
            return self._grow_log_likelihood_ratio(
                mutation=mutation,
                sigma_val=sigma_val,
                leaf_mean_prior_scale=leaf_mean_prior_scale,
                X=X,
                partial_residual=partial_residual,
            )
        elif isinstance(mutation, PruneMutation):
            return -self._grow_log_likelihood_ratio(
                mutation=GrowMutation(
                    old_node=mutation.new_node, new_node=mutation.old_node
                ),
                sigma_val=sigma_val,
                leaf_mean_prior_scale=leaf_mean_prior_scale,
                X=X,
                partial_residual=partial_residual,
            )
        else:
            raise TreeStructureError(" Can only grow or prune")

    def _grow_log_likelihood_ratio(
        self,
        mutation: GrowMutation,
        sigma_val: float,
        leaf_mean_prior_scale: float,
        X: torch.Tensor,
        partial_residual: torch.Tensor,
    ) -> float:
        """
        Implement expression for log( P(R | T*, sigma) / P(R | T, sigma) ) in a GROW move as discussed in eq. 10 of [1]

        Reference:
            [1] Adam Kapelner & Justin Bleich. "bartMachine: Machine Learning with Bayesian Additive Regression Trees" (2013).
            https://arxiv.org/abs/1312.2171

        Args:
            tree: Previous tree.
            mutation: Proposed mutation,
            sigma_val:urrent estimate of noise standard deviation in the data.
            leaf_mean_prior_scale: Prior of the scale hyperparameter in the normal distribution of the leaf mean.
            X: Covariate matrix / training data.
            partial_residual: Partial residual of the current tree model with respect to the training data.

        """

        var = sigma_val**2
        var_mu = leaf_mean_prior_scale**2
        nodes = {
            "parent": mutation.old_node,
            "left": mutation.new_node.left_child,
            "right": mutation.new_node.right_child,
        }
        y_sum, num_points = {}, {}
        for node_label, node in nodes.items():
            X_conditioned, y_conditioned = node.data_in_node(X, partial_residual)
            y_sum[node_label] = torch.sum(y_conditioned)
            num_points[node_label] = len(X_conditioned)

        first_term = (var * (var + num_points["parent"] * leaf_mean_prior_scale)) / (
            (var + num_points["left"] * var_mu) * (var + num_points["right"] * var_mu)
        )
        first_term = math.log(math.sqrt(first_term))

        left_contribution = torch.square(y_sum["left"]) / (
            var + num_points["left"] * leaf_mean_prior_scale
        )
        right_contribution = torch.square(y_sum["right"]) / (
            var + num_points["right"] * leaf_mean_prior_scale
        )
        parent_contribution = torch.square(y_sum["parent"]) / (
            var + num_points["parent"] * leaf_mean_prior_scale
        )

        second_term = left_contribution + right_contribution - parent_contribution

        return first_term + (var_mu / (2 * var)) * second_term.item()

    def _get_log_structure_ratio(
        self,
        mutation: Mutation,
        alpha: float,
        beta: float,
        X: torch.Tensor,
    ) -> float:
        """
        Implement expression for log( P(T*) / P(T) ) in as discussed in [1].

        Reference:
            [1] Adam Kapelner & Justin Bleich. "bartMachine: Machine Learning with Bayesian Additive Regression Trees" (2013).
            https://arxiv.org/abs/1312.2171

        Args:
            mutation: Proposed mutation,
            X: Covariate matrix / training data.
            alpha: Hyperparameter used in tree prior.
            beta: Hyperparameter used in tree prior.
        """
        if isinstance(mutation, GrowMutation):
            return self._grow_log_structure_ratio(
                mutation=mutation,
                alpha=alpha,
                beta=beta,
                X=X,
            )
        elif isinstance(mutation, PruneMutation):
            return -self._grow_log_structure_ratio(
                mutation=GrowMutation(
                    old_node=mutation.new_node, new_node=mutation.old_node
                ),
                alpha=alpha,
                beta=beta,
                X=X,
            )
        else:
            raise TreeStructureError("Only grow or prune mutations are allowed")

    def _grow_log_structure_ratio(
        self,
        mutation: GrowMutation,
        alpha: float,
        beta: float,
        X: torch.Tensor,
    ) -> float:
        """
        Implement expression for log( P(T*) / P(T) ) in a GROW step as discussed in section A.1 of [1].

        Reference:
            [1] Adam Kapelner & Justin Bleich. "bartMachine: Machine Learning with Bayesian Additive Regression Trees" (2013).
            https://arxiv.org/abs/1312.2171

        Args:
            mutation: Proposed mutation,
            X: Covariate matrix / training data.
            alpha: Hyperparameter used in tree prior.
            beta: Hyperparameter used in tree prior.
        """

        denominator = _log_probability_node_is_terminal(alpha, beta, mutation.old_node)

        log_probability_left_is_terminal = _log_probability_node_is_terminal(
            alpha, beta, mutation.new_node.left_child
        )
        log_probability_right_is_terminal = _log_probability_node_is_terminal(
            alpha, beta, mutation.new_node.right_child
        )
        log_probability_parent_is_nonterminal = _log_probability_node_is_nonterminal(
            alpha, beta, mutation.old_node
        )
        log_probability_rule = _log_probability_of_growing_node(mutation=mutation, X=X)
        numerator = (
            log_probability_left_is_terminal
            + log_probability_right_is_terminal
            + log_probability_parent_is_nonterminal
            + log_probability_rule
        )
        return numerator - denominator


def _log_probability_node_is_nonterminal(
    alpha: float, beta: float, node: Union[LeafNode, SplitNode]
) -> float:
    """Get log probability of node being non-terminal (internal node) as discussed in Eq. 7 of [1].

    Reference:
        [1] Hugh A. Chipman, Edward I. George, Robert E. McCulloch (2010). "BART: Bayesian additive regression trees"
        https://projecteuclid.org/journals/annals-of-applied-statistics/volume-4/issue-1/BART-Bayesian-additive-regression-trees/10.1214/09-AOAS285.full

    Args:
        alpha: Hyperparameter used in tree prior.
        beta: Hyperparameter used in tree prior.
        node: Node for which probability is being calculated.

    """
    return math.log(alpha * math.pow(1 + node.depth, -beta))


def _log_probability_node_is_terminal(
    alpha: float, beta: float, node: Union[LeafNode, SplitNode]
) -> float:
    """Get log probability of node being terminal (leaf node) as discussed in Eq. 7 of [1].

    Reference:
        [1] Hugh A. Chipman, Edward I. George, Robert E. McCulloch (2010). "BART: Bayesian additive regression trees"
        https://projecteuclid.org/journals/annals-of-applied-statistics/volume-4/issue-1/BART-Bayesian-additive-regression-trees/10.1214/09-AOAS285.full

    Args:
        alpha: Hyperparameter used in tree prior.
        beta: Hyperparameter used in tree prior.
        node: Node for which probability is being calculated.

    """
    return 1 - _log_probability_node_is_nonterminal(alpha=alpha, beta=beta, node=node)


def _log_probability_of_growing_a_tree(
    tree: Tree, mutation: GrowMutation, X: torch.Tensor
) -> float:
    """
    Get probability of choosing a node and growing it as discussed in section A.1 of [1].

    Reference:
        [1] Adam Kapelner & Justin Bleich. "bartMachine: Machine Learning with Bayesian Additive Regression Trees" (2013).
        https://arxiv.org/abs/1312.2171

    Args:
        tree: Previous tree.
        mutation: Growth mutation being applied.
        X: Covariate matrix / training data.
    """

    return -math.log(tree.num_growable_leaf_nodes(X))
    +_log_probability_of_growing_node(mutation=mutation, X=X)


def _log_probability_of_growing_node(mutation: GrowMutation, X: torch.Tensor) -> float:
    """
    Get probability of growing a node as discussed in section A.1 of [1].

    Reference:
        [1] Adam Kapelner & Justin Bleich. "bartMachine: Machine Learning with Bayesian Additive Regression Trees" (2013).
        https://arxiv.org/abs/1312.2171

    Args:
        mutation: Growth mutation being applied.
        X: Covariate matrix / training data.
    """
    log_probability_of_selecting_dim = -math.log(
        mutation.old_node.get_num_growable_dims(X)
    )
    grow_dim = mutation.new_node.most_recent_rule().grow_dim
    grow_val = mutation.new_node.most_recent_rule().grow_val
    log_probability_of_growing_at_val = -math.log(
        mutation.old_node.get_partition_of_split(X, grow_dim, grow_val)
    )
    return log_probability_of_selecting_dim + log_probability_of_growing_at_val
