# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torch.distributions as dist
from beanmachine.ppl.world.utils import get_default_transforms


class InferenceUtilsTest(unittest.TestCase):
    def test_get_default_transforms(self):
        bernoulli = dist.Bernoulli(0.1)
        transforms = get_default_transforms(bernoulli)
        self.assertEqual(dist.transforms.identity_transform, transforms)

        normal = dist.Normal(0, 1)
        transforms = get_default_transforms(normal)
        self.assertEqual(dist.transforms.identity_transform, transforms)

        gamma = dist.Gamma(1, 1)
        transforms = get_default_transforms(gamma)
        self.assertTrue(transforms.bijective)
