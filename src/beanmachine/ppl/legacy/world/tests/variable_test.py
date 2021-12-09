# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from collections import namedtuple

import torch
import torch.distributions as dist
from beanmachine.ppl.legacy.world.variable import TransformData, TransformType, Variable
from beanmachine.ppl.world.utils import get_default_transforms, initialize_value
from torch import tensor


class VariableTest(unittest.TestCase):
    def test_variable_types(self):
        with self.assertRaises(ValueError):
            Variable(1, 1, 1, 1, 1, 1, 1, 1, 1)

    def test_variable_assignments(self):
        distribution = dist.Normal(0, 1)
        val = distribution.sample()
        log_prob = distribution.log_prob(val)
        var = Variable(
            distribution=distribution,
            value=val,
            log_prob=log_prob,
            transformed_value=val,
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
            transformed_value=val,
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

    def test_transform_gamma_log_prob(self):
        distribution = dist.Gamma(2, 2)
        val = distribution.sample()
        expected_log_prob = distribution.log_prob(val)
        var = Variable(
            distribution=distribution,
            value=None,
            log_prob=expected_log_prob,
            transform=dist.ExpTransform(),
            transformed_value=None,
            jacobian=tensor(0.0),
        )

        var.update_fields(val, None, TransformData(TransformType.NONE, []), None)
        self.assertAlmostEqual(val.item(), var.transformed_value.item(), delta=0.01)
        log_prob = var.log_prob + var.jacobian
        self.assertAlmostEqual(expected_log_prob.item(), log_prob.item(), delta=0.01)

        var.update_fields(val, None, TransformData(TransformType.DEFAULT, []), None)
        unconstrained_sample = var.transformed_value
        jacobian = var.jacobian
        log = var.log_prob
        log_prob = log + jacobian
        transform = get_default_transforms(var.distribution)
        expected_unconstrained_sample = transform(val)
        expected_constrained_sample = transform.inv(expected_unconstrained_sample)
        expected_log_prob = distribution.log_prob(
            expected_constrained_sample
        ) - transform.log_abs_det_jacobian(
            expected_constrained_sample, expected_unconstrained_sample
        )
        self.assertAlmostEqual(
            expected_unconstrained_sample.item(),
            unconstrained_sample.item(),
            delta=0.01,
        )
        log_prob = var.log_prob + var.jacobian
        self.assertAlmostEqual(expected_log_prob.item(), log_prob.item(), delta=0.01)

        var.update_fields(
            val,
            None,
            TransformData(
                TransformType.CUSTOM,
                [transform],
            ),
            None,
        )
        self.assertAlmostEqual(
            expected_unconstrained_sample.item(),
            var.transformed_value.item(),
            delta=0.01,
        )
        log_prob = var.log_prob + var.jacobian
        self.assertAlmostEqual(expected_log_prob.item(), log_prob.item(), delta=0.01)

    def test_transform_beta_log_prob(self):
        distribution = dist.Beta(2.0, 2.0)
        val = distribution.sample()
        expected_log_prob = distribution.log_prob(val)
        var = Variable(
            distribution=distribution,
            value=None,
            log_prob=expected_log_prob,
            transform=dist.ExpTransform(),
            transformed_value=None,
            jacobian=tensor(0.0),
        )

        var.update_fields(val, None, TransformData(TransformType.NONE, []), None)
        self.assertAlmostEqual(val.item(), var.transformed_value.item(), delta=0.01)
        log_prob = var.log_prob + var.jacobian
        self.assertAlmostEqual(expected_log_prob.item(), log_prob.item(), delta=0.01)

        var.update_fields(val, None, TransformData(TransformType.DEFAULT, []), None)
        unconstrained_sample = var.transformed_value
        jacobian = var.jacobian
        log = var.log_prob
        log_prob = log + jacobian
        transform = get_default_transforms(var.distribution)
        expected_unconstrained_sample = transform(val)
        expected_constrained_sample = transform.inv(expected_unconstrained_sample)
        expected_log_prob = distribution.log_prob(
            expected_constrained_sample
        ) - transform.log_abs_det_jacobian(
            expected_constrained_sample, expected_unconstrained_sample
        )
        self.assertAlmostEqual(
            expected_unconstrained_sample.item(),
            unconstrained_sample.item(),
            delta=0.01,
        )
        log_prob = var.log_prob + var.jacobian
        self.assertAlmostEqual(expected_log_prob.item(), log_prob.item(), delta=0.01)

        var.update_fields(
            val,
            None,
            TransformData(
                TransformType.CUSTOM,
                [transform],
            ),
            None,
        )
        self.assertAlmostEqual(
            expected_unconstrained_sample.item(),
            var.transformed_value.item(),
            delta=0.01,
        )
        log_prob = var.log_prob + var.jacobian
        self.assertAlmostEqual(expected_log_prob.item(), log_prob.item(), delta=0.01)

    def test_transform_dirichlet_log_prob(self):
        distribution = dist.Dirichlet(tensor([0.5, 0.5]))
        val = distribution.sample()
        expected_log_prob = distribution.log_prob(val)
        var = Variable(
            distribution=distribution,
            value=None,
            log_prob=expected_log_prob,
            transform=dist.ExpTransform(),
            transformed_value=None,
            jacobian=tensor(0.0),
        )

        var.update_fields(val, None, TransformData(TransformType.NONE, []), None)
        self.assertAlmostEqual(val[0], var.transformed_value[0], delta=0.01)
        self.assertAlmostEqual(val[1], var.transformed_value[1], delta=0.01)
        log_prob = var.log_prob + var.jacobian
        self.assertAlmostEqual(expected_log_prob.item(), log_prob.item(), delta=0.01)

        var.update_fields(val, None, TransformData(TransformType.DEFAULT, []), None)
        jacobian = var.jacobian
        log = var.log_prob
        log_prob = log + jacobian
        transform = get_default_transforms(var.distribution)
        expected_unconstrained_sample = transform(val)
        expected_constrained_sample = transform.inv(expected_unconstrained_sample)
        expected_log_prob = distribution.log_prob(
            expected_constrained_sample
        ) - transform.log_abs_det_jacobian(
            expected_constrained_sample, expected_unconstrained_sample
        )
        self.assertAlmostEqual(
            expected_unconstrained_sample.item(),
            var.transformed_value.item(),
            delta=0.01,
        )
        log_prob = var.log_prob + var.jacobian
        self.assertAlmostEqual(expected_log_prob.item(), log_prob.item(), delta=0.01)

        var.update_fields(
            val,
            None,
            TransformData(TransformType.CUSTOM, [transform]),
            None,
        )
        self.assertAlmostEqual(
            expected_unconstrained_sample.item(),
            var.transformed_value.item(),
            delta=0.01,
        )
        log_prob = var.log_prob + var.jacobian
        self.assertAlmostEqual(expected_log_prob.item(), log_prob.item(), delta=0.01)

    def test_str_nonscalar(self):
        distribution = dist.MultivariateNormal(tensor([0.0, 0.0]), torch.eye(2))
        var = Variable(
            distribution=distribution,
            value=None,
            log_prob=None,
            transformed_value=None,
            jacobian=None,
        )
        value = initialize_value(distribution)
        var.update_value(value)
        try:
            str(var)
        except Exception:
            self.fail("str(Variable) raised an Exception!")
