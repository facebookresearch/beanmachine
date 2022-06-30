# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC
from dataclasses import dataclass
from typing import Union

from beanmachine.ppl.experimental.causal_inference.models.bart.exceptions import (
    GrowError,
    PruneError,
)
from beanmachine.ppl.experimental.causal_inference.models.bart.node import (
    LeafNode,
    SplitNode,
)


@dataclass
class Mutation(ABC):
    """
    A data class for storing the nodes before and after a mutation to a tree.
    These mutations are applied to traverse the space of tree structures. The possible mutations considered here are:
    - **Grow**: Where a `LeafNode` of the tree is split based on a decision rule, turning it into an internal `SplitNode`.
    - **Prune**: Where an internal `SplitNode` with only terminal children is converted into a `LeafNode`.
    These steps constitute the Grow-Prune approach of Pratola [1] where the additional steps of
    BART (Change and Swap) are eliminated.

    Reference:
        [1] Pratola MT, Chipman H, Higdon D, McCulloch R, Rust W (2013). “Parallel Bayesian Additive Regression Trees.”
        Technical report, University of Chicago.
        https://arxiv.org/pdf/1309.1906.pdf

    Args:
        old_node: The node before mutation.
        new_node: The node after mutation.

    """

    __slots__ = ["old_node", "new_node"]

    def __init__(
        self,
        old_node: Union[SplitNode, LeafNode],
        new_node: Union[SplitNode, LeafNode],
    ):
        self.old_node = old_node
        self.new_node = new_node


@dataclass
class PruneMutation(Mutation):
    """Encapsulates the prune action where an internal `SplitNode` with only terminal children
    is converted into a `LeafNode`.

    Args:
        old_node: The node before mutation.
        new_node: The node after mutation.
    """

    def __init__(self, old_node: SplitNode, new_node: LeafNode):
        """
        Raises:
            PruneError: if the prune mutation is invalid.
        """

        if not isinstance(old_node, SplitNode) or not old_node.is_prunable():
            raise PruneError("Pruning only valid on prunable SplitNodes")
        if not isinstance(new_node, LeafNode):
            raise PruneError("Pruning can only create a LeafNode")
        super().__init__(old_node, new_node)


@dataclass
class GrowMutation(Mutation):
    """Encapsulates the grow action where a `LeafNode` of the tree is split based on a decision rule,
    turning it into an internal `SplitNode`.

    Args:
        old_node: The node before mutation.
        new_node: The node after mutation.

    """

    def __init__(self, old_node: LeafNode, new_node: SplitNode):
        """
        Raises:
            GrowError: if the grow mutation is invalid.
        """
        if not isinstance(old_node, LeafNode):
            raise GrowError("Can only grow LeafNodes")
        if not isinstance(new_node, SplitNode):
            raise GrowError("Growing a LeafNode turns it into a SplitNode")
        super().__init__(old_node, new_node)
