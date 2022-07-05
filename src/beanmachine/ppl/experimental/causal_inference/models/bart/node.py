# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, overload, Tuple, Union

import torch

from beanmachine.ppl.experimental.causal_inference.models.bart.exceptions import (
    PruneError,
)
from beanmachine.ppl.experimental.causal_inference.models.bart.split_rule import (
    CompositeRules,
    SplitRule,
)


class BaseNode:
    """
    Base class for node structures.
    Contains reference to a left and right child which can be used to traverse the tree.

    Args:
        depth (int): Distance of node from root node.
        composite_rules (CompositeRules): Dimensional rules that are satisfied by this node.
        left_child ("BaseNode"): Left child of the node.
        right_child ("BaseNode"): Right child of the node.
    """

    def __init__(
        self,
        depth: int,
        composite_rules: CompositeRules,
        left_child: Optional["BaseNode"] = None,
        right_child: Optional["BaseNode"] = None,
    ):
        """ """
        self.depth = depth
        self.composite_rules = composite_rules
        self._left_child = left_child
        self._right_child = right_child

    @property
    def left_child(self) -> Optional["BaseNode"]:
        """Returns the left_child of the node."""
        return self._left_child

    @property
    def right_child(self) -> Optional["BaseNode"]:
        """Returns the right_child of the node."""
        return self._right_child

    @overload
    def data_in_node(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    @overload
    def data_in_node(self, X: torch.Tensor) -> torch.Tensor:
        ...

    def data_in_node(
        self, X: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Conditions the covariate matrix and (optionally) response vector to return the
        respective subsets which satisfy the composite rules of this node.
        Note that the conditioning only looks at the input / covariate matrix
        to determine this conditioning.

        Args:
            X: Input / covariate matrix.
            y: (Optional) response vector.
        """
        condition_mask = self.composite_rules.condition_on_rules(X)
        if y is not None:
            return X[condition_mask], y[condition_mask]
        return X[condition_mask]


class LeafNode(BaseNode):
    """
    A representation of a leaf node in the tree. Does not have children.
    In addition to the normal work of a `BaseNode`, a `LeafNode` is responsible for
    making predictions based on its value.

    Args:
        depth (int): Distance of node from root node.
        composite_rules (CompositeRules): Dimensional rules that are satisfied by this node.
        val (float): The prediction value of the node.

    """

    def __init__(
        self,
        depth: int,
        composite_rules: CompositeRules,
        val: float = 0.0,
    ):
        self.val = val
        super().__init__(
            depth=depth,
            composite_rules=composite_rules,
            left_child=None,
            right_child=None,
        )

    def predict(self) -> float:
        """
        Returns the val attribute as a prediction.
        """
        return self.val

    def is_growable(self, X: torch.Tensor) -> bool:
        """
        Returns true if this leaf node can be grown.
        This is checked by ensuring the input covariate matrix
        has atleast more than 1 unique values along any dimension.

        Args:
            X: Input / covariate matrix.
        """
        return len(self.get_growable_dims(X)) > 0

    def get_growable_dims(self, X: torch.Tensor) -> List[int]:
        """
        Returns the list of dimensions along which this leaf node can be gronw.
        This is checked by ensuring the input covariate matrix
        has more than 1 unique values along any dimension.

        Args:
            X: Input / covariate matrix.
        """
        X_conditioned = self.data_in_node(X)
        if len(X_conditioned) == 0:
            return []
        return [
            dim
            for dim in range(X_conditioned.shape[-1])
            if len(torch.unique(self.get_growable_vals(X_conditioned, dim))) > 1
        ]

    def get_num_growable_dims(self, X: torch.Tensor) -> int:
        """
        Returns the number of dimensions along which this leaf node can be grown.
        This is checked by ensuring the input covariate matrix
        has atleast more than 1 unique values along any dimension.

        Args:
            X: Input / covariate matrix.
        """
        return len(self.get_growable_dims(X))

    def get_growable_vals(self, X: torch.Tensor, grow_dim: int) -> torch.Tensor:
        """Returns the values in a feature dimension.
        Args:
            X: Input / covariate matrix.
            grow_dim: Input dimensions along which values are required
        """
        return self.data_in_node(X)[:, grow_dim]

    def get_partition_of_split(
        self, X: torch.Tensor, grow_dim: int, grow_val: float
    ) -> float:
        """
        Get probability that a split value is chosen among possible values in an input dimension defined as
        N(values_in_dimension == split_val) / N(values_in_dimension).
        Args:
            X: Input / covariate matrix.
            grow_dim: Input dimensions along which values are required.
            grow_va;: The value along which the split is being carried out.
        """

        growable_vals = self.get_growable_vals(X, grow_dim)
        return torch.mean(
            (growable_vals == grow_val).to(torch.float), dtype=torch.float
        ).item()

    @staticmethod
    def grow_node(
        node: "LeafNode",
        left_rule: SplitRule,
        right_rule: SplitRule,
    ) -> "SplitNode":
        """
        Converts a LeafNode into an internal SplitNode by applying the split rules for the left and right nodes.
        This returns a copy of the oriingal node.
        Args:
            left_rule: Rule applied to left child of the grown node.
            right_rule: Rule applied to the right child of the grown node.
        """
        left_composite_rules = node.composite_rules.add_rule(left_rule)
        right_composite_rules = node.composite_rules.add_rule(right_rule)

        return SplitNode(
            depth=node.depth,
            composite_rules=node.composite_rules,
            left_child=LeafNode(
                depth=node.depth + 1, composite_rules=left_composite_rules
            ),
            right_child=LeafNode(
                depth=node.depth + 1, composite_rules=right_composite_rules
            ),
        )


class SplitNode(BaseNode):
    """
    Encapsulates internal node in the tree. It has the same attributes as BaseNode.
    It contains the additional logic to determine if this node can be pruned.

    Args:
        depth (int): Distance of node from root node.
        composite_rules (CompositeRules): Dimensional rules that are satisfied by this node.
        left_child ("BaseNode"): Left child of the node.
        right_child ("BaseNode"): Right child of the node.
    """

    def __init__(
        self,
        depth: int,
        composite_rules: CompositeRules,
        left_child: Optional["BaseNode"] = None,
        right_child: Optional["BaseNode"] = None,
    ):
        """
        Args:
           depth: Distance of node from root node.
           composite_rules: Dimensional rules that are satisfied by this node.
           left_child: Left child of the node.
           right_child: Right child of the node.
        """
        super().__init__(
            depth=depth,
            composite_rules=composite_rules,
            left_child=left_child,
            right_child=right_child,
        )

    def is_prunable(self) -> bool:
        """Returns true if this node is prunable. This is decided by the fact if its children are `LeafNodes`."""
        return isinstance(self.left_child, LeafNode) and isinstance(
            self.right_child, LeafNode
        )

    def most_recent_rule(self) -> Optional[SplitRule]:
        """Returns the rule which grew this node from a `LeafNode` and is specifically the rule which created its left child."""
        if self.left_child is None:
            raise AttributeError("This node is not split")
        return self.left_child.composite_rules.most_recent_split_rule()

    @staticmethod
    def prune_node(
        node: "SplitNode",
    ) -> LeafNode:
        """
        Converts a SplitNode to a LeafNode by eliminating its children (if they are leaf nodes). Returns a copy.

        Args:
            node: Node to prune.
        Raises:
            PruneError: If this node is not prunable.
        """
        if not node.is_prunable():
            raise PruneError("Not a valid prunable node")
        return LeafNode(depth=node.depth, composite_rules=node.composite_rules)
