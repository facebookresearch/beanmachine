# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import pytest
import torch
from beanmachine.ppl.experimental.causal_inference.models.bart.exceptions import (
    PruneError,
)
from beanmachine.ppl.experimental.causal_inference.models.bart.node import (
    BaseNode,
    LeafNode,
    SplitNode,
)
from beanmachine.ppl.experimental.causal_inference.models.bart.split_rule import (
    CompositeRules,
    Operator,
    SplitRule,
)


@pytest.fixture
def composite_rule():
    all_rules = []
    all_dims = [0, 1, 2]
    for dim in all_dims:
        all_rules.append(SplitRule(grow_dim=dim, grow_val=0, operator=Operator.le))
    composite_rule = CompositeRules(all_dims=all_dims, all_split_rules=all_rules)
    return composite_rule


@pytest.fixture
def left_rule():
    return SplitRule(grow_dim=0, grow_val=-0.5, operator=Operator.le)


@pytest.fixture
def right_rule():
    return SplitRule(grow_dim=0, grow_val=-0.5, operator=Operator.gt)


@pytest.fixture
def all_pass_composite_rule():
    all_rules = []
    all_dims = [0, 1, 2]
    for dim in all_dims:
        all_rules.append(
            SplitRule(grow_dim=dim, grow_val=float("inf"), operator=Operator.le)
        )
    composite_rule = CompositeRules(all_dims=all_dims, all_split_rules=all_rules)
    return composite_rule


@pytest.fixture
def X():
    return torch.Tensor([[1.0, 3.0, 7.0], [-1.1, -1, -5]])


def test_conditioning(X, composite_rule):
    base_node = BaseNode(depth=0, composite_rules=composite_rule)
    assert torch.all(
        base_node.data_in_node(X) == X[composite_rule.condition_on_rules(X)]
    )


def test_leaf_node_prediction(composite_rule):
    val = 10
    leaf_node = LeafNode(composite_rules=composite_rule, depth=0, val=val)
    assert leaf_node.predict() == val


@pytest.fixture
def leaf_node(composite_rule):
    return LeafNode(composite_rules=composite_rule, depth=0)


@pytest.fixture
def loose_leaf(all_pass_composite_rule):
    return LeafNode(composite_rules=all_pass_composite_rule, depth=0)


def test_growable_dims(leaf_node, loose_leaf, X):
    assert leaf_node.get_num_growable_dims(X) == 0  # only one row of X passes the test
    assert loose_leaf.get_num_growable_dims(X) == X.shape[-1]  # everything passes
    assert len(loose_leaf.get_growable_dims(X)) == loose_leaf.get_num_growable_dims(X)


def test_is_grow(leaf_node, loose_leaf, X):
    assert not leaf_node.is_growable(X)  # no splittable_dims. Cannot grow.
    assert loose_leaf.is_growable(X)


def test_grow_node(leaf_node, left_rule, right_rule, X):
    grown_leaf = LeafNode.grow_node(
        leaf_node, left_rule=left_rule, right_rule=right_rule
    )
    assert isinstance(grown_leaf, SplitNode)
    assert grown_leaf.left_child is not None
    assert grown_leaf.right_child is not None
    assert grown_leaf.most_recent_rule() == left_rule


def test_prune_node(leaf_node, composite_rule):
    split_node = SplitNode(
        left_child=leaf_node,
        right_child=deepcopy(leaf_node),
        depth=1,
        composite_rules=composite_rule,
    )
    grandfather_node = SplitNode(
        left_child=leaf_node,
        right_child=split_node,
        depth=0,
        composite_rules=composite_rule,
    )
    assert split_node.is_prunable()
    assert not grandfather_node.is_prunable()
    assert isinstance(SplitNode.prune_node(split_node), LeafNode)
    with pytest.raises(PruneError):
        SplitNode.prune_node(grandfather_node)

    def test_partition_of_split(loose_leaf, X):
        grow_val = X[0, 0]
        growable_vals = loose_leaf.get_growable_vals(X=X, grow_dim=0)

        assert torch.isclose(
            torch.tensor(
                [loose_leaf.get_partition_of_split(X=X, grow_dim=0, grow_val=grow_val)]
            ),
            torch.mean(growable_vals.eq(grow_val.item()), dtype=torch.float),
        )
