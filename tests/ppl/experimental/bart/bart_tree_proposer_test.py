# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch

from beanmachine.ppl.experimental.causal_inference.models.bart.exceptions import (
    PruneError,
)
from beanmachine.ppl.experimental.causal_inference.models.bart.grow_prune_tree_proposer import (
    GrowPruneTreeProposer,
    MutationKind,
)

from beanmachine.ppl.experimental.causal_inference.models.bart.node import (
    LeafNode,
    SplitNode,
)
from beanmachine.ppl.experimental.causal_inference.models.bart.split_rule import (
    CompositeRules,
    Operator,
    SplitRule,
)

from beanmachine.ppl.experimental.causal_inference.models.bart.tree import Tree


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(5)


@pytest.fixture
def X():
    return torch.Tensor([[3.0, 1.0], [4.0, 1.0], [1.5, 1.0], [-1.0, 1.0]])


@pytest.fixture
def root_node(X):
    return SplitNode(
        depth=0,
        composite_rules=CompositeRules(all_dims=list(range(X.shape[-1]))),
    )


@pytest.fixture
def single_node_tree(X):
    leaf_root = LeafNode(
        depth=0,
        composite_rules=CompositeRules(all_dims=list(range(X.shape[-1]))),
    )
    tree_ = Tree(nodes=[leaf_root])
    return tree_


@pytest.fixture
def r1_growable(X):
    return LeafNode(
        depth=1,
        composite_rules=CompositeRules(
            all_dims=list(range(X.shape[-1])),
            all_split_rules=[SplitRule(grow_dim=0, grow_val=0, operator=Operator.gt)],
        ),
        val=-10,
    )


@pytest.fixture
def l1_non_growable(X):
    return LeafNode(
        depth=1,
        composite_rules=CompositeRules(
            all_dims=list(range(X.shape[-1])),
            all_split_rules=[SplitRule(grow_dim=0, grow_val=0, operator=Operator.le)],
        ),
        val=-10,
    )


@pytest.fixture
def single_layer_tree(root_node, r1_growable, l1_non_growable):
    """
                root_node
                /\
    (x1 <= 0)l1   r1 (x1 > 0)

    The tree is made such that all positive input gets a positive prediciton and vice-versa.

    """

    root_node._left_child = l1_non_growable
    root_node._right_child = r1_growable
    tree_ = Tree(nodes=[root_node, l1_non_growable, r1_growable])
    return tree_


@pytest.fixture
def l2_non_growable(X):
    return LeafNode(
        depth=2,
        composite_rules=CompositeRules(
            all_dims=list(range(X.shape[-1])),
            all_split_rules=[SplitRule(grow_dim=0, grow_val=3, operator=Operator.le)],
        ),
        val=-10,
    )


@pytest.fixture
def r2_growable(X):
    return LeafNode(
        depth=1,
        composite_rules=CompositeRules(
            all_dims=list(range(X.shape[-1])),
            all_split_rules=[SplitRule(grow_dim=0, grow_val=0, operator=Operator.gt)],
        ),
        val=-10,
    )


@pytest.fixture
def r1_grown(X):
    return SplitNode(
        depth=1,
        composite_rules=CompositeRules(
            all_dims=list(range(X.shape[-1])),
            all_split_rules=[SplitRule(grow_dim=0, grow_val=3, operator=Operator.gt)],
        ),
    )


@pytest.fixture
def double_layer_tree(
    root_node, r1_grown, l1_non_growable, r2_growable, l2_non_growable
):
    """
                root_node
                /\
    (x1 <= 0)l1   r1 (x1 > 0)
                 /\
         (<=3)l2   r2 (>3)

    """

    root_node._left_child = l1_non_growable
    root_node._right_child = r1_grown
    r1_grown._left_child = l2_non_growable
    r1_grown._right_child = r2_growable
    tree_ = Tree(
        nodes=[root_node, l1_non_growable, r1_grown, l2_non_growable, r2_growable]
    )
    return tree_


@pytest.fixture
def proposer():
    return GrowPruneTreeProposer()


def test_new_mutation(proposer, single_node_tree, X):
    assert proposer._get_new_mutation(X=X, tree=single_node_tree) == MutationKind.grow


def test_select_root_to_grow(proposer, single_node_tree, X):
    assert (
        proposer._select_leaf_to_grow(single_node_tree, X) == single_node_tree._nodes[0]
    )


def test_select_leaf_to_grow(proposer, single_layer_tree, X, r1_growable):
    assert proposer._select_leaf_to_grow(single_layer_tree, X) == r1_growable


def test_select_dim_to_grow(proposer, single_node_tree, X):
    assert proposer._select_grow_dim(leaf_to_grow=single_node_tree._nodes[0], X=X) == 0


def test_select_node_to_prune(proposer, single_node_tree, double_layer_tree, r1_grown):
    assert proposer._select_split_node_to_prune(tree=double_layer_tree) == r1_grown
    with pytest.raises(PruneError):
        _ = proposer._select_split_node_to_prune(tree=single_node_tree)


def test_propose(proposer, single_node_tree, X):
    proposed_tree = proposer.propose(
        tree=single_node_tree,
        X=X,
        partial_residual=torch.zeros(X.shape[0], 1),
        alpha=0.5,
        beta=0.5,
        sigma_val=0.01,
        leaf_mean_prior_scale=1,
    )
    assert isinstance(proposed_tree, Tree)
    assert abs(proposed_tree.num_nodes() - single_node_tree.num_nodes()) in [
        0,
        2,
    ]  # 2: grow or prune, 0 for no change
