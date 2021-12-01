# Copyright (c) Facebook, Inc. and its affiliates.
import pytest
import torch.distributions as dist
from beanmachine.ppl.world.utils import get_default_transforms, initialize_value


def test_get_default_transforms():
    bernoulli = dist.Bernoulli(0.1)
    transforms = get_default_transforms(bernoulli)
    assert dist.transforms.identity_transform == transforms

    normal = dist.Normal(0, 1)
    transforms = get_default_transforms(normal)
    assert dist.transforms.identity_transform == transforms

    gamma = dist.Gamma(1, 1)
    transforms = get_default_transforms(gamma)
    assert transforms.bijective


def test_initialize_value():
    distribution = dist.Normal(0, 1)
    value = initialize_value(distribution)
    assert value.item() == pytest.approx(0, abs=1e-5)
    first_sample = initialize_value(distribution, True)
    second_sample = initialize_value(distribution, True)
    assert first_sample.item() != second_sample.item()
