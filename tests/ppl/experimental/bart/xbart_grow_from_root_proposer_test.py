# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from beanmachine.ppl.experimental.causal_inference.models.bart.bart_model import (
    LeafMean,
)

from beanmachine.ppl.experimental.causal_inference.models.bart.grow_from_root_tree_proposer import (
    GrowFromRootTreeProposer,
    SortedInvariants,
)
from beanmachine.ppl.experimental.causal_inference.models.bart.node import LeafNode
from beanmachine.ppl.experimental.causal_inference.models.bart.split_rule import (
    CompositeRules,
)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(5)


@pytest.fixture
def gfr_proposer():
    gfr = GrowFromRootTreeProposer()
    gfr.num_cuts = 2
    gfr.num_null_cuts = 1
    return gfr


@pytest.fixture
def X():
    return torch.Tensor([[3.0, 1.0], [4.0, 1.0], [1.5, 1.0], [-1.0, 1.0]])


@pytest.fixture
def w(X):
    num_vars = X.shape[-1]
    weights = torch.Tensor([1 / num_vars for _ in range(num_vars - 1)])
    return weights


def test_sample_variables(gfr_proposer, w):
    num_vars_to_sample = max(len(w) - 1, 1)
    assert (
        len(gfr_proposer._sample_variables(num_vars_to_sample, w)) == num_vars_to_sample
    )
    impossible_num_vars_to_sample = len(w) + 1
    assert len(gfr_proposer._sample_variables(impossible_num_vars_to_sample, w)) == len(
        w
    )


def test_presort(gfr_proposer, X):
    O_ = gfr_proposer._presort(X)
    num_observations, num_dims = X.shape
    for inp_dim in range(num_dims):
        for obs in range(1, num_observations):
            assert X[O_[inp_dim, obs - 1], inp_dim] <= X[O_[inp_dim, obs], inp_dim]


def test_get_uniq_elems(gfr_proposer, X):
    O_ = gfr_proposer._presort(X)
    uniq_vals, val_counts = gfr_proposer._get_uniq_elems(X=X, O_=O_)
    num_observations, num_dims = X.shape
    for inp_dim in range(num_dims):
        dim_val_counts = val_counts[inp_dim]
        assert sum(dim_val_counts.values()) == num_observations
        for id_, uniq_val in enumerate(uniq_vals[inp_dim]):
            assert dim_val_counts[uniq_val] > 0
            if id_ > 0:
                assert uniq_val >= uniq_vals[inp_dim][id_ - 1]
        assert set(uniq_vals[inp_dim]) == {_.item() for _ in X[:, inp_dim]}


@pytest.fixture
def invariants(gfr_proposer, X):
    O_ = gfr_proposer._presort(X)
    uniq_vals, val_counts = gfr_proposer._get_uniq_elems(X=X, O_=O_)
    return SortedInvariants(O_=O_, uniq_vals=uniq_vals, val_counts=val_counts)


def test_select_cutpoints(gfr_proposer, X, invariants):
    num_observations, num_dims = X.shape
    cutpoints = gfr_proposer._select_cutpoints(
        candidate_dims=list(range(num_dims)), uniq_vals=invariants.uniq_vals
    )

    num_dim_cuts = 0
    for point_id, point in enumerate(cutpoints):
        assert (
            point.cut_val < invariants.uniq_vals[point.dim][-1]
        )  # no degenerate splits
        if point_id > 0 and cutpoints[point_id - 1].dim == point.dim:
            assert cutpoints[point_id - 1].cut_val < point.cut_val
            num_dim_cuts += 1
        elif point_id > 0 and cutpoints[point_id - 1].dim != point.dim:
            assert num_dim_cuts <= gfr_proposer.num_cuts
            num_dim_cuts = 0
        else:
            num_dim_cuts += 1


@pytest.fixture
def partial_residual(X):
    return torch.ones((len(X), 1)) * 0.2


@pytest.fixture
def sigma_val():
    return 0.1


@pytest.fixture
def leaf_sampler():
    return LeafMean(prior_loc=0.0, prior_scale=0.1)


@pytest.fixture
def current_node(X):
    return LeafNode(
        depth=0,
        val=0.1,
        composite_rules=CompositeRules(all_dims=list(range(X.shape[-1]))),
    )


@pytest.fixture
def alpha():
    return 0.95


@pytest.fixture
def beta():
    return 1.25


@pytest.fixture
def cut_points(gfr_proposer, invariants):
    num_dims = invariants.O_.shape[0]
    return gfr_proposer._select_cutpoints(
        candidate_dims=list(range(num_dims)), uniq_vals=invariants.uniq_vals
    )


def test_sample_cut_point(
    gfr_proposer,
    X,
    invariants,
    cut_points,
    partial_residual,
    sigma_val,
    leaf_sampler,
    current_node,
    alpha,
    beta,
):

    num_observations, num_dims = X.shape

    num_trials = 10
    all_sampled_cutpoints = []
    for _ in range(num_trials):
        all_sampled_cutpoints.append(
            gfr_proposer._sample_cut_point(
                candidate_cut_points=cut_points,
                partial_residual=partial_residual,
                invariants=invariants,
                sigma_val=sigma_val,
                leaf_sampler=leaf_sampler,
                current_node=current_node,
                alpha=alpha,
                beta=beta,
            )
        )
    for point in all_sampled_cutpoints:
        if point is not None:
            assert point in cut_points


def test_sift(
    gfr_proposer,
    X,
    invariants,
    cut_points,
    partial_residual,
    sigma_val,
    leaf_sampler,
    current_node,
    alpha,
    beta,
):
    cut_point = gfr_proposer._sample_cut_point(
        candidate_cut_points=cut_points,
        partial_residual=partial_residual,
        invariants=invariants,
        sigma_val=sigma_val,
        leaf_sampler=leaf_sampler,
        current_node=current_node,
        alpha=alpha,
        beta=beta,
    )
    left_invariants, right_invariants = gfr_proposer._sift(
        X=X, cut_point=cut_point, invariants=invariants
    )
    assert (
        invariants.O_.shape[0] == left_invariants.O_.shape[0]
        and invariants.O_.shape[0] == right_invariants.O_.shape[0]
    )  # num dims shouldnt change
    assert (
        invariants.O_.shape[1]
        == left_invariants.O_.shape[1] + right_invariants.O_.shape[1]
    )

    for dim in range(invariants.O_.shape[0]):
        assert set(invariants.uniq_vals[dim]) == set(
            left_invariants.uniq_vals[dim]
        ).union(set(right_invariants.uniq_vals[dim]))
        for val in invariants.uniq_vals[dim]:
            assert (
                invariants.val_counts[dim][val]
                == left_invariants.val_counts[dim][val]
                + right_invariants.val_counts[dim][val]
            )


def test_propose(
    X,
    invariants,
    cut_points,
    partial_residual,
    sigma_val,
    leaf_sampler,
    current_node,
    alpha,
    beta,
    w,
):
    proposer = GrowFromRootTreeProposer()
    tree_, variable_counts = proposer.propose(
        X=X,
        partial_residual=partial_residual,
        m=X.shape[-1],
        w=w,
        sigma_val=sigma_val,
        leaf_sampler=leaf_sampler,
        alpha=alpha,
        beta=beta,
        root_node=current_node,
        num_cuts=2,
        num_null_cuts=1,
    )

    all_leaves = tree_.leaf_nodes()
    assert len(all_leaves) > 0
    if len(all_leaves) > 0:
        assert sum(variable_counts) > 0

    assert tree_.predict(X).shape == partial_residual.shape
