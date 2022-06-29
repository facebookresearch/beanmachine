# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch
from beanmachine.ppl.experimental.causal_inference.models.bart.split_rule import (
    CompositeRules,
    DimensionalRule,
    Operator,
    SplitRule,
)


@pytest.fixture
def grow_dim():
    return 1


@pytest.fixture
def grow_val():
    return 2.1


def test_dimensional_rule_addition(grow_dim, grow_val):
    lax_rule = SplitRule(
        grow_dim=grow_dim, grow_val=grow_val + 10, operator=Operator.le
    )
    existing_dimensional_rule = DimensionalRule(
        grow_dim=grow_dim, min_val=grow_val - 20, max_val=grow_val
    )
    assert (
        existing_dimensional_rule.max_val
        == existing_dimensional_rule.add_rule(lax_rule).max_val
    )
    assert (
        existing_dimensional_rule.min_val
        == existing_dimensional_rule.add_rule(lax_rule).min_val
    )

    restrictive_rule_le = SplitRule(
        grow_dim=grow_dim, grow_val=grow_val - 10, operator=Operator.le
    )
    assert (
        existing_dimensional_rule.max_val
        > existing_dimensional_rule.add_rule(restrictive_rule_le).max_val
    )
    assert (
        existing_dimensional_rule.min_val
        == existing_dimensional_rule.add_rule(restrictive_rule_le).min_val
    )

    restrictive_rule_gt = SplitRule(
        grow_dim=grow_dim, grow_val=grow_val - 10, operator=Operator.gt
    )
    assert (
        existing_dimensional_rule.max_val
        == existing_dimensional_rule.add_rule(restrictive_rule_gt).max_val
    )
    assert (
        existing_dimensional_rule.min_val
        < existing_dimensional_rule.add_rule(restrictive_rule_gt).min_val
    )


@pytest.fixture
def all_dims():
    return [0, 2]


@pytest.fixture
def all_split_rules(all_dims):
    all_rules = []
    for dim in all_dims:
        all_rules.append(SplitRule(grow_dim=dim, grow_val=5, operator=Operator.le))
    return all_rules


@pytest.fixture
def X():
    return torch.Tensor([[1.0, 3.0, 7.0], [-1.1, 100, 5]])


def test_composite_rules(all_dims, all_split_rules, X):
    composite_rule = CompositeRules(all_dims=all_dims, all_split_rules=all_split_rules)
    X_cond = X[composite_rule.condition_on_rules(X)]
    for dim in all_dims:
        assert torch.all(X_cond[:, dim] > composite_rule.dimensional_rules[dim].min_val)
        assert torch.all(
            X_cond[:, dim] <= composite_rule.dimensional_rules[dim].max_val
        )

    invalid_split_rule = SplitRule(
        grow_dim=max(all_dims) + 1, grow_val=12, operator=Operator.le
    )
    with pytest.raises(ValueError):
        _ = composite_rule.add_rule(invalid_split_rule)

    valid_split_rule = SplitRule(
        grow_dim=max(all_dims), grow_val=1000.0, operator=Operator.gt
    )
    valid_new_composite_rule = composite_rule.add_rule(valid_split_rule)
    assert valid_new_composite_rule.most_recent_split_rule() == valid_split_rule
