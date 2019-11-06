# Copyright (c) Facebook, Inc. and its affiliates.
import unittest
from collections import namedtuple

import torch.distributions as dist
from beanmachine.ppl.world.variable import Variable


class VariableTest(unittest.TestCase):
    def test_variable_types(self):
        with self.assertRaises(ValueError):
            Variable(1, 1, 1, 1, 1, 1, 1)

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
        )
        var_copy = var.copy()
        self.assertEqual(var_copy.distribution, distribution)
        self.assertEqual(var_copy.value, val)
        self.assertEqual(var_copy.log_prob, log_prob)
        self.assertEqual(var_copy.parent, set({tmp(name="name")}))
        self.assertEqual(var_copy.children, set({tmp(name="name")}))
        self.assertEqual(var_copy.parent is var.parent, False)
        self.assertEqual(var_copy.children is var.children, False)
