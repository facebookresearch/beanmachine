import beanmachine.ppl as bm
import pytest
import torch
import torch.distributions as dist
from beanmachine.ppl.experimental.global_inference.proposer.nuts_proposer import (
    NUTSProposer,
    _Tree,
    _TreeArgs,
    _TreeNode,
)
from beanmachine.ppl.experimental.global_inference.simple_world import SimpleWorld


@bm.random_variable
def foo():
    return dist.Beta(2.0, 2.0)


@bm.random_variable
def bar():
    return dist.Bernoulli(foo())


@pytest.fixture
def nuts():
    world = SimpleWorld(observations={bar(): torch.tensor(0.8)})
    world.call(bar())
    nuts_proposer = NUTSProposer(world)
    return nuts_proposer


@pytest.fixture
def tree_node(nuts):
    momentums = nuts._initialize_momentums(nuts._positions)
    return _TreeNode(
        positions=nuts._positions, momentums=momentums, pe_grad=nuts._pe_grad
    )


@pytest.fixture
def tree_args(tree_node, nuts):
    initial_energy = nuts._hamiltonian(
        nuts._positions, tree_node.momentums, nuts._mass_inv, nuts._pe
    )
    return _TreeArgs(
        log_slice=-initial_energy,
        direction=1,
        step_size=nuts.step_size,
        initial_energy=initial_energy,
        mass_inv=nuts._mass_inv,
    )


def test_base_tree(tree_node, tree_args, nuts):
    nuts._multinomial_sampling = False
    tree_args = tree_args._replace(
        log_slice=torch.log1p(-torch.rand(())) - tree_args.initial_energy
    )
    tree = nuts._build_tree_base_case(root=tree_node, args=tree_args)
    assert isinstance(tree, _Tree)
    assert torch.isclose(tree.log_weight, torch.tensor(float("-inf"))) or torch.isclose(
        tree.log_weight, torch.tensor(0.0)
    )
    assert tree.left == tree.right


def test_base_tree_multinomial(tree_node, tree_args, nuts):
    tree = nuts._build_tree_base_case(root=tree_node, args=tree_args)
    assert isinstance(tree, _Tree)
    # in multinomial sampling, trees are weighted by their accept prob
    assert torch.isclose(
        torch.clamp(tree.log_weight.exp(), max=1.0), tree.sum_accept_prob
    )


def test_build_tree(tree_node, tree_args, nuts):
    tree_depth = 3
    tree = nuts._build_tree(root=tree_node, tree_depth=tree_depth, args=tree_args)
    assert isinstance(tree, _Tree)
    assert tree.turned_or_diverged or (tree.left is not tree.right)
    assert tree.turned_or_diverged or tree.num_proposals == 2 ** tree_depth
