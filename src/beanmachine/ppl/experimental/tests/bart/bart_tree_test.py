# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch

from beanmachine.ppl.experimental.causal_inference.models.bart.exceptions import (
    GrowError,
    PruneError,
)
from beanmachine.ppl.experimental.causal_inference.models.bart.mutation import (
    GrowMutation,
    PruneMutation,
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


@pytest.fixture
def X():
    return torch.Tensor(
        [[3.0], [4.0], [1.5], [-1.0]]
    )  # only r1 containing all positive entries is growable


@pytest.fixture
def l1_non_growable():
    return LeafNode(
        depth=1,
        composite_rules=CompositeRules(
            all_dims=[0],
            all_split_rules=[SplitRule(grow_dim=0, grow_val=0, operator=Operator.le)],
        ),
        val=-10,
    )


@pytest.fixture
def l2_non_growable():
    return LeafNode(
        depth=2,
        composite_rules=CompositeRules(
            all_dims=[0],
            all_split_rules=[
                SplitRule(grow_dim=0, grow_val=1.5, operator=Operator.le),
                SplitRule(grow_dim=0, grow_val=0, operator=Operator.gt),
            ],
        ),
        val=15,
    )


@pytest.fixture
def r2_growable():
    return LeafNode(
        depth=2,
        composite_rules=CompositeRules(
            all_dims=[0],
            all_split_rules=[
                SplitRule(grow_dim=0, grow_val=1.5, operator=Operator.gt),
                SplitRule(grow_dim=0, grow_val=0, operator=Operator.gt),
            ],
        ),
        val=15,
    )


@pytest.fixture
def r1_grown(r2_growable, l2_non_growable):
    return SplitNode(
        depth=1,
        left_child=l2_non_growable,
        right_child=r2_growable,
        composite_rules=CompositeRules(
            all_dims=[0],
            all_split_rules=[SplitRule(grow_dim=0, grow_val=0, operator=Operator.gt)],
        ),
    )


@pytest.fixture
def root(l1_non_growable, r1_grown):
    return SplitNode(
        depth=0,
        left_child=l1_non_growable,
        right_child=r1_grown,
        composite_rules=CompositeRules(all_dims=[0]),
    )


@pytest.fixture
def tree(root, r1_grown, l1_non_growable, r2_growable, l2_non_growable):
    """
                root_node
                /\
    (x1 <= 0)l1   r1 (x1 > 0)
                /   \
    (x1 <= 1.5) l2  r2 (x1 > 1.5)

    The tree is made such that all positive input gets a positive prediciton and vice-versa.

    """

    tree_ = Tree(nodes=[root, l1_non_growable, r1_grown, l2_non_growable, r2_growable])
    return tree_


def test_num_nodes(tree):
    assert tree.num_nodes() == 5


def test_leaf_split_nodes(tree):
    for node in tree.split_nodes():
        assert isinstance(node, SplitNode)
    for node in tree.split_nodes():
        assert isinstance(node, SplitNode)


def test_prunable_split_nodes(tree):
    for node in tree.prunable_split_nodes():
        assert isinstance(node.left_child, LeafNode)
        assert isinstance(node.left_child, LeafNode)
    assert len(tree.prunable_split_nodes()) == tree.num_prunable_split_nodes()


def test_growable_leaves(tree, r2_growable, l1_non_growable, X):
    assert tree.num_growable_leaf_nodes(X) == 1
    growable_leaves = tree.growable_leaf_nodes(X)
    assert len(tree.growable_leaf_nodes(X)) == len(growable_leaves)
    assert r2_growable in growable_leaves
    assert l1_non_growable not in growable_leaves
    assert l2_non_growable not in growable_leaves


def test_prediction(tree, X):
    for x1 in X:
        x1 = x1.reshape(1, 1)
        assert float(x1) * tree.predict(x1) >= 0


def test_mutate_prune(tree, root, l1_non_growable, r1_grown):
    old_tree_len = tree.num_nodes()
    pruned_r1 = SplitNode.prune_node(r1_grown)

    # pruning an internal node
    with pytest.raises(PruneError):
        _ = PruneMutation(old_node=root, new_node=l1_non_growable)

    mutation = PruneMutation(old_node=r1_grown, new_node=pruned_r1)
    tree.mutate(mutation)
    assert tree.num_nodes() == old_tree_len - 2


def test_mutate_grow(tree, r2_growable):
    old_tree_len = tree.num_nodes()

    l3 = LeafNode(
        depth=3,
        composite_rules=CompositeRules(
            all_dims=[0],
            all_split_rules=[SplitRule(grow_dim=0, grow_val=3, operator=Operator.le)],
        ),
        val=15,
    )
    r3 = LeafNode(
        depth=3,
        composite_rules=CompositeRules(
            all_dims=[0],
            all_split_rules=[SplitRule(grow_dim=0, grow_val=1.5, operator=Operator.gt)],
        ),
        val=15,
    )
    r2_grown = SplitNode(
        depth=2,
        left_child=l3,
        right_child=r3,
        composite_rules=CompositeRules(
            all_dims=[0],
            all_split_rules=[SplitRule(grow_dim=0, grow_val=1.5, operator=Operator.gt)],
        ),
    )
    # growing an internal node
    with pytest.raises(GrowError):
        _ = GrowMutation(old_node=r2_grown, new_node=r2_growable)

    mutation = GrowMutation(old_node=r2_growable, new_node=r2_grown)
    tree.mutate(mutation)
    assert tree.num_nodes() == old_tree_len + 2
