# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torch.distributions as dist
from beanmachine.ppl.world.utils import get_default_transforms
from torch import tensor


class InferenceUtilsTest(unittest.TestCase):
    def test_get_default_transforms(self):
        bernoulli = dist.Bernoulli(0.1)
        transforms = get_default_transforms(bernoulli)
        self.assertListEqual([], transforms)

        normal = dist.Normal(0, 1)
        transforms = get_default_transforms(normal)
        self.assertListEqual([], transforms)

        gamma = dist.Gamma(1, 1)
        transforms = get_default_transforms(gamma)
        value = tensor(0.5)
        transformed_value = value
        for f in transforms:
            transformed_value = f(transformed_value)
        expected_transform = dist.ExpTransform().inv
        self.assertAlmostEqual(transformed_value, expected_transform(value))

        log_normal = dist.LogNormal(1, 1)
        value = tensor(0.5)
        transformed_value = value
        transforms = get_default_transforms(log_normal)
        for f in transforms:
            transformed_value = f(transformed_value)
        expected_transform = dist.ExpTransform().inv
        self.assertAlmostEqual(transformed_value, expected_transform(value))

        dirichlet = dist.Dirichlet(tensor([0.5, 0.5]))
        value = tensor([0.2, 0.8])
        transformed_value = value
        transforms = get_default_transforms(dirichlet)
        for f in transforms:
            transformed_value = f(transformed_value)
        expected_transform = dist.StickBreakingTransform().inv
        self.assertAlmostEqual(transformed_value, expected_transform(value))

        beta = dist.Beta(1, 1)
        value = tensor(0.5)
        transformed_value = value
        transforms = get_default_transforms(beta)
        for f in transforms:
            transformed_value = f(transformed_value)
        expected_transform = dist.StickBreakingTransform().inv
        self.assertAlmostEqual(
            transformed_value, expected_transform(tensor([value, 1 - value]))
        )

        uniform = dist.Uniform(0, 10)
        value = tensor(5.0)
        transformed_value = value
        transforms = get_default_transforms(uniform)
        for f in transforms:
            transformed_value = f(transformed_value)
        expected_transform = dist.StickBreakingTransform().inv
        pass
