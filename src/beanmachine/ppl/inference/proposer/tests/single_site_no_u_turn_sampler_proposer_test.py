# Copyright (c) Facebook, Inc. and its affiliates
import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.inference.proposer.single_site_no_u_turn_sampler_proposer import (
    SingleSiteNoUTurnSamplerProposer,
)
from beanmachine.ppl.world import Variable, World
from torch import tensor
from torch.autograd import grad


class SingleSiteNoUTurnSamplerProposerTest(unittest.TestCase):
    def test_nuts_compute_kinetic_energy(self):
        proposer = SingleSiteNoUTurnSamplerProposer()
        proposer.covariance = tensor([[0.95, 0.05], [0.05, 0.60]])
        k = proposer._compute_kinetic_energy(tensor([1.0, 2.0]))
        self.assertAlmostEqual(k.item(), 1.775)

    def test_nuts_leapfrog_step(self):
        world = World()
        proposer = SingleSiteNoUTurnSamplerProposer()
        distribution = dist.Normal(0, 1)
        val = tensor(1.0)
        val.requires_grad_(True)

        @bm.random_variable
        def foo():
            return dist.Normal(0, 1)

        node = foo()
        node_var = Variable(
            distribution=distribution,
            value=val,
            log_prob=distribution.log_prob(val).sum(),
            parent=set(),
            children=set(),
            proposal_distribution=None,
            is_discrete=False,
            transforms=[],
            transformed_value=val,
            jacobian=tensor(0.0),
        )
        world.variables_.data_[node] = node_var
        theta = tensor(1.0)
        theta.requires_grad_(True)
        r = tensor(0.2)
        step_size = tensor(0.1)

        is_valid, proposer_theta, proposer_r = proposer._leapfrog_step(
            node, world, theta, r, step_size
        )

        d = dist.Normal(0, 1)
        grad_U = grad(-(d.log_prob(theta)), theta)[0]
        r = r - step_size * grad_U / 2
        handwritten_theta = theta + step_size * r
        grad_U = grad(-(d.log_prob(handwritten_theta)), handwritten_theta)[0]
        handwritten_r = r - step_size * grad_U / 2

        self.assertAlmostEqual(proposer_theta, handwritten_theta)
        self.assertAlmostEqual(proposer_r, handwritten_r)

    def test_nuts_build_tree_base_case_invalid(self):
        world = World()
        proposer = SingleSiteNoUTurnSamplerProposer()
        distribution = dist.Normal(0, 1)
        val = tensor(1.0)
        val.requires_grad_(True)

        @bm.random_variable
        def foo():
            return dist.Normal(0, 1)

        node = foo()
        node_var = Variable(
            distribution=distribution,
            value=val,
            log_prob=distribution.log_prob(val).sum(),
            parent=set(),
            children=set(),
            proposal_distribution=None,
            is_discrete=False,
            transforms=[],
            transformed_value=val,
            jacobian=tensor(0.0),
        )
        world.variables_.data_[node] = node_var

        theta = tensor(5.0)
        r = tensor(0.2)
        build_tree_input = (
            theta,
            r,
            tensor(-1.0),
            tensor(1.0),
            0,
            tensor(1.0),
            tensor(0.2),
        )
        proposer.step_size = tensor(1e30)
        output = proposer._build_tree_base_case(node, world, build_tree_input)
        self.assertAlmostEqual(output[0], theta)
        self.assertAlmostEqual(output[1], r)
        self.assertAlmostEqual(output[2], theta)
        self.assertAlmostEqual(output[3], r)
        self.assertAlmostEqual(output[4], theta)
        self.assertAlmostEqual(output[5], tensor(1.0))
        self.assertAlmostEqual(output[6], tensor(0.0))
        self.assertAlmostEqual(output[7], tensor(0.0))
        self.assertAlmostEqual(output[8], tensor(1.0))

    def test_nuts_build_tree_base_case_valid(self):
        world = World()
        proposer = SingleSiteNoUTurnSamplerProposer()
        distribution = dist.Normal(0, 1)
        val = tensor(1.0)
        val.requires_grad_(True)

        @bm.random_variable
        def foo():
            return dist.Normal(0, 1)

        node = foo()
        node_var = Variable(
            distribution=distribution,
            value=val,
            log_prob=distribution.log_prob(val).sum(),
            parent=set(),
            children=set(),
            proposal_distribution=None,
            is_discrete=False,
            transforms=[],
            transformed_value=val,
            jacobian=tensor(0.0),
        )
        world.variables_.data_[node] = node_var

        theta = tensor(1.0)
        theta.requires_grad_(True)
        r = tensor(0.2)
        u = tensor(0.2)
        v = tensor(1.0)
        j = 0
        theta0 = tensor(0.8)
        r0 = tensor(0.18)
        step_size = tensor(0.1)
        build_tree_input = (theta, r, u, v, j, theta0, r0)

        proposer.step_size = step_size

        output = proposer._build_tree_base_case(node, world, build_tree_input)

        d = dist.Normal(0, 1)
        grad_U = grad(-(d.log_prob(theta)), theta)[0]
        r = r - step_size * grad_U / 2
        theta1 = theta + step_size * r
        grad_U = grad(-(d.log_prob(theta1)), theta1)[0]
        r1 = r - step_size * grad_U / 2

        h1 = d.log_prob(theta1) - (r1 * r1).sum() / 2
        h0 = d.log_prob(theta0) - (r0 * r0).sum() / 2
        dH = h1 - h0

        n1 = (u <= dH).detach().clone()
        s1 = (u < proposer.delta_max + dH).detach().clone()
        accept_ratio = torch.min(tensor(1.0), torch.exp(dH))

        self.assertAlmostEqual(output[0], theta1)
        self.assertAlmostEqual(output[1], r1)
        self.assertAlmostEqual(output[2], theta1)
        self.assertAlmostEqual(output[3], r1)
        self.assertAlmostEqual(output[4], theta1)
        self.assertEqual(output[5], n1)
        self.assertEqual(output[6], s1)
        self.assertAlmostEqual(output[7], accept_ratio)
        self.assertAlmostEqual(output[8], tensor(1.0))

    def test_nuts_build_tree(self):
        world = World()
        proposer = SingleSiteNoUTurnSamplerProposer(use_dense_mass_matrix=False)
        distribution = dist.Normal(0, 1)
        val = tensor(1.0)
        val.requires_grad_(True)

        @bm.random_variable
        def foo():
            return dist.Normal(0, 1)

        node = foo()
        node_var = Variable(
            distribution=distribution,
            value=val,
            log_prob=distribution.log_prob(val).sum(),
            parent=set(),
            children=set(),
            proposal_distribution=None,
            is_discrete=False,
            transforms=[],
            transformed_value=val,
            jacobian=tensor(0.0),
        )
        world.variables_.data_[node] = node_var

        step_size = tensor(0.1)
        theta = tensor(1.0)
        theta.requires_grad_(True)
        r = tensor(0.2)
        u = tensor(0.2)
        v = tensor(1.0)
        j = 2
        theta0 = tensor(0.8)
        r0 = tensor(0.18)
        proposer.step_size = step_size
        build_tree_input = (theta, r, u, v, j, theta0, r0)
        torch.manual_seed(17)
        output = proposer._build_tree(node, world, build_tree_input)

        # subtree
        d = dist.Normal(0, 1)
        grad_U = grad(-(d.log_prob(theta)), theta)[0]
        r = r - step_size * grad_U / 2
        theta1 = theta + step_size * r
        grad_U = grad(-(d.log_prob(theta1)), theta1)[0]
        r1 = r - step_size * grad_U / 2
        h1 = d.log_prob(theta1) - (r1 * r1).sum() / 2
        h0 = d.log_prob(theta0) - (r0 * r0).sum() / 2
        dH = h1 - h0
        n1 = (u <= dH).detach().clone()
        accept_ratio_1 = torch.min(tensor(1.0), torch.exp(dH)).detach()

        # right subtree
        d = dist.Normal(0, 1)
        grad_U = grad(-(d.log_prob(theta1)), theta1)[0]
        r1_next = r1 - step_size * grad_U / 2
        theta2 = theta1 + step_size * r1_next
        grad_U = grad(-(d.log_prob(theta2)), theta2)[0]
        r2 = r1_next - step_size * grad_U / 2
        h1 = d.log_prob(theta2) - (r2 * r2).sum() / 2
        dH = h1 - h0
        n2 = (u <= dH).detach().clone()
        s2 = (u < proposer.delta_max + dH).detach().clone()
        accept_ratio_2 = torch.min(tensor(1.0), torch.exp(dH)).detach()

        neg_turn = ((theta2 - theta1) * r1).sum() >= 0
        pos_turn = ((theta2 - theta1) * r2).sum() >= 0
        stop = s2 * neg_turn * pos_turn

        self.assertAlmostEqual(output[0], theta1)
        self.assertAlmostEqual(output[1], r1)
        self.assertAlmostEqual(output[2], theta2)
        self.assertAlmostEqual(output[3], r2)
        self.assertAlmostEqual(output[4], theta1)
        self.assertEqual(output[5], n1 + n2)
        self.assertEqual(output[6], stop)
        self.assertAlmostEqual(output[7].detach(), accept_ratio_1 + accept_ratio_2)
        self.assertAlmostEqual(output[8], tensor(2.0))

    def test_nuts_compute_hamiltonian(self):
        world = World()
        proposer = SingleSiteNoUTurnSamplerProposer()
        distribution = dist.Normal(0, 1)
        val = tensor(1.0)
        val.requires_grad_(True)

        @bm.random_variable
        def foo():
            return dist.Normal(0, 1)

        node = foo()
        node_var = Variable(
            distribution=distribution,
            value=val,
            log_prob=distribution.log_prob(val).sum(),
            parent=set(),
            children=set(),
            proposal_distribution=None,
            is_discrete=False,
            transforms=[],
            transformed_value=val,
            jacobian=tensor(0.0),
        )
        world.variables_.data_[node] = node_var

        theta = tensor(2.0)
        r = tensor(0.5)
        proposer.covariance = tensor([[1.0]])
        hamiltonian = proposer._compute_hamiltonian(node, world, theta, r)
        handwritten_hamiltonian = dist.Normal(0, 1).log_prob(theta) - (r * r).sum() / 2
        self.assertAlmostEqual(hamiltonian, handwritten_hamiltonian)
