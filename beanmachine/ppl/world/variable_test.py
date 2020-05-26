# Copyright (c) Facebook, Inc. and its affiliates.
import unittest
from collections import namedtuple

import torch
import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.world.variable import Variable


class VariableTest(unittest.TestCase):
    def test_variable_types(self):
        with self.assertRaises(ValueError):
            Variable(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

    def test_variable_assignments(self):
        distribution = dist.Normal(0, 1)
        val = distribution.sample()
        log_prob = distribution.log_prob(val)
        var = Variable(
            distribution=distribution,
            value=val,
            log_prob=log_prob,
            parent=set(),
            children=set(),
            proposal_distribution=None,
            extended_val=None,
            is_discrete=False,
            transforms=[],
            unconstrained_value=val,
            jacobian=tensor(0.0),
        )
        self.assertEqual(var.distribution, distribution)
        self.assertEqual(var.value, val)
        self.assertEqual(var.log_prob, log_prob)
        self.assertEqual(var.parent, set())
        self.assertEqual(var.children, set())

    def test_copy(self):
        tmp = namedtuple("tmp", "name")
        distribution = dist.Normal(0, 1)
        val = distribution.sample()
        log_prob = distribution.log_prob(val)
        var = Variable(
            distribution=distribution,
            value=val,
            log_prob=log_prob,
            parent=set({tmp(name="name")}),
            children=set({tmp(name="name")}),
            proposal_distribution=None,
            extended_val=None,
            is_discrete=False,
            transforms=[],
            unconstrained_value=val,
            jacobian=tensor(0.0),
        )
        var_copy = var.copy()
        self.assertEqual(var_copy.distribution, distribution)
        self.assertEqual(var_copy.value, val)
        self.assertEqual(var_copy.log_prob, log_prob)
        self.assertEqual(var_copy.parent, set({tmp(name="name")}))
        self.assertEqual(var_copy.children, set({tmp(name="name")}))
        self.assertEqual(var_copy.parent is var.parent, False)
        self.assertEqual(var_copy.children is var.children, False)

    def test_transform_log_prob(self):
        distribution = dist.Gamma(2, 2)
        val = distribution.sample()
        log_prob = distribution.log_prob(val)
        var = Variable(
            distribution=distribution,
            value=None,
            log_prob=log_prob,
            parent=set(),
            children=set(),
            proposal_distribution=None,
            extended_val=None,
            is_discrete=False,
            transforms=[dist.ExpTransform()],
            unconstrained_value=None,
            jacobian=tensor(0.0),
        )

        var.update_fields(val, None, True)
        unconstrained_sample = var.unconstrained_value
        jacobian = var.jacobian
        log = var.log_prob
        log_prob = log + jacobian
        transform = dist.ExpTransform()
        expected_unconstrained_sample = transform._inverse(val)
        expected_constrained_sample = transform._call(expected_unconstrained_sample)
        expected_log_prob = distribution.log_prob(
            expected_constrained_sample
        ) + transform.log_abs_det_jacobian(
            expected_unconstrained_sample, expected_constrained_sample
        )
        self.assertAlmostEqual(
            expected_unconstrained_sample.item(),
            unconstrained_sample.item(),
            delta=0.01,
        )

        self.assertAlmostEqual(expected_log_prob.item(), log_prob.item(), delta=0.01)

    def test_str_nonscalar(self):
        distribution = dist.MultivariateNormal(tensor([0.0, 0.0]), torch.eye(2))
        var = Variable(
            distribution=distribution,
            value=None,
            log_prob=None,
            parent=set(),
            children=set(),
            proposal_distribution=None,
            extended_val=None,
            is_discrete=None,
            transforms=[],
            unconstrained_value=None,
            jacobian=None,
        )
        value = var.initialize_value(None)
        var.update_value(value)
        try:
            str(var)
        except Exception:
            self.fail("str(Variable) raised an Exception!")
