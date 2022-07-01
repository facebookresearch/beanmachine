# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import torch
from beanmachine.ppl.experimental.causal_inference.models.bart.exceptions import (
    TreeStructureError,
)
from beanmachine.ppl.experimental.causal_inference.models.bart.mutation import (
    GrowMutation,
    PruneMutation,
)

from beanmachine.ppl.experimental.causal_inference.models.bart.node import (
    LeafNode,
    SplitNode,
)


class Tree:
    """
    Encapsulates a tree structure where each node is either a nonterminal `SplitNode` or a terminal `LeafNode`.
    This class consists of methods to track and modify overall tree structure.

    Args:
        nodes: List of nodes comprising the tree.
    """

    def __init__(self, nodes: List[Union[LeafNode, SplitNode]]):
        self._nodes = nodes

    def num_nodes(self) -> int:
        """
        Returns the total number of nodes in the tree.
        """
        return len(self._nodes)

    def leaf_nodes(self) -> List[LeafNode]:
        """
        Returns a list of all of the leaf nodes in the tree.
        """
        return [node for node in self._nodes if isinstance(node, LeafNode)]

    def growable_leaf_nodes(self, X: torch.Tensor) -> List[LeafNode]:
        """
        List of all leaf nodes in the tree which can be grown in a non-degenerate way
        i.e. such that not all values in the column of the covariate matrix are duplicates
        conditioned on the rules of that node.

        Args:
            X: Input / covariate matrix.
        """
        return [node for node in self.leaf_nodes() if node.is_growable(X)]

    def num_growable_leaf_nodes(self, X: torch.Tensor) -> int:
        """
        Returns the number of nodes which can be grown in the tree.
        """
        return len(self.growable_leaf_nodes(X))

    def split_nodes(self) -> List[SplitNode]:
        """
        List of internal `SplitNode`s in the tree.
        """
        return [node for node in self._nodes if isinstance(node, SplitNode)]

    def prunable_split_nodes(self) -> List[SplitNode]:
        """
        List of decision nodes in the tree that are suitable for pruning
        i.e., `SplitNode`s` that have two terminal `LeafNode` children
        """
        return [node for node in self.split_nodes() if node.is_prunable()]

    def num_prunable_split_nodes(self) -> int:
        """
        Number of prunable split nodes in tree.
        """
        return len(self.prunable_split_nodes())

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Generate a set of predictions with the same dimensionality as the target array
        Note that the prediction is from one tree, so represents only (1 / number_of_trees) of the target.
        """
        prediction = torch.zeros((len(X), 1), dtype=torch.float)
        for leaf in self.leaf_nodes():
            prediction[leaf.composite_rules.condition_on_rules(X)] = leaf.predict()
        return prediction

    def mutate(self, mutation: Union[GrowMutation, PruneMutation]) -> None:
        """
        Apply a change to the structure of the tree.
        Args:
            mutation: The mutation to apply to the tree.
                Only grow and prune mutations are accepted.
        """

        if isinstance(mutation, PruneMutation):
            self._remove_node(mutation.old_node.left_child)
            self._remove_node(mutation.old_node.right_child)
            self._remove_node(mutation.old_node)
            self._add_node(mutation.new_node)

        elif isinstance(mutation, GrowMutation):
            self._remove_node(mutation.old_node)
            self._add_node(mutation.new_node)
            self._add_node(mutation.new_node.left_child)
            self._add_node(mutation.new_node.right_child)

        else:
            raise TreeStructureError("Only Grow and Prune mutations are valid.")

        for node in self._nodes:
            if node.right_child == mutation.old_node:
                node._right_child = mutation.new_node
            if node.left_child == mutation.old_node:
                node._left_child = mutation.new_node

    def _remove_node(self, node: Optional[Union[LeafNode, SplitNode]] = None) -> None:
        """
        Remove a single node from the tree non-recursively.
        Only drops the node and not any children.
        """
        if node is not None:
            self._nodes.remove(node)

    def _add_node(self, node: Optional[Union[LeafNode, SplitNode]] = None) -> None:
        """
        Add a node to the tree non-recursively.
        Only adds the node and does not link it to any node.
        """
        if node is not None:
            self._nodes.append(node)
