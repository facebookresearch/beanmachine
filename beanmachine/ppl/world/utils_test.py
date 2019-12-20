# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.world.utils import get_transforms, is_discrete


class InferenceUtilsTest(unittest.TestCase):
    def test_is_discrete(self):
        self.assertTrue(is_discrete(dist.Bernoulli(0.1)))
        self.assertTrue(is_discrete(dist.Categorical(tensor([0.5, 0.5]))))
        self.assertTrue(is_discrete(dist.Multinomial(10, tensor([1.0, 1.0, 1.0, 1.0]))))
        self.assertTrue(is_discrete(dist.Geometric(tensor([0.3]))))
        self.assertFalse(is_discrete(dist.Normal(0, 1)))

    def test_get_transform(self):
        bernoulli = dist.Bernoulli(0.1)
        transforms = get_transforms(bernoulli)
        self.assertListEqual([], transforms)

        normal = dist.Normal(0, 1)
        transforms = get_transforms(normal)
        self.assertListEqual([], transforms)

        gamma = dist.Gamma(1, 1)
        transforms = get_transforms(gamma)
        self.assertListEqual(transforms, [dist.ExpTransform()])

        log_normal = dist.LogNormal(1, 1)
        transforms = get_transforms(log_normal)
        self.assertListEqual(transforms, [dist.ExpTransform()])

        dirichlet = dist.Dirichlet(tensor([0.5, 0.5]))
        transforms = get_transforms(dirichlet)
        self.assertListEqual(transforms, [dist.StickBreakingTransform()])

        beta = dist.Beta(1, 1)
        transforms = get_transforms(beta)
        self.assertListEqual(transforms, [dist.StickBreakingTransform()])

        uniform = dist.Uniform(0, 10)
        with self.assertRaises(ValueError):
            transforms = get_transforms(uniform)
