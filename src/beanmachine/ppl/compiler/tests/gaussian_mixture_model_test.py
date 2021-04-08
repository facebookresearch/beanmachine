#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.


# Suppose we have a mixture of k normal distributions each with standard
# deviation equal to 1, but different means. Our prior on means is that
# mean(0), ... mean(k) are normally distributed.
#
# To make samples mixed(0), ... from this distribution we first choose which
# mean we want with category(0), ..., use that to sample mean(category(0))
# to get the mean, and then use that mean to sample from a normal distribution.
#

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference import BMGInference
from torch import tensor
from torch.distributions import Categorical, Normal


@bm.random_variable
def mean(k):
    # Means of the components are normally distributed
    return Normal(0, 1)


@bm.random_variable
def category(item):
    # Choose a category, 0, 1 or 2 with ratio 1:3:4.
    return Categorical(tensor([1.0, 3.0, 4.0]))


@bm.random_variable
def mixed(item):
    return Normal(mean(category(item)), 1)


class GaussianMixtureModelTest(unittest.TestCase):
    def test_gmm_to_dot(self) -> None:
        self.maxDiff = None
        queries = [mixed(0)]
        observations = {}

        # TODO: Since this model uses a categorical distribution to choose another
        # random variable, which is not yet supported in BMG, we cannot compile
        # it to BMG. Once we can, update this test to use after_transform=True
        # and actually run inference.

        observed = BMGInference().to_dot(queries, observations, after_transform=False)
        expected = """
digraph "graph" {
  N00[label=0];
  N01[label=1.0];
  N02[label=Normal];
  N03[label=Sample];
  N04[label=Sample];
  N05[label=2];
  N06[label=Sample];
  N07[label=map];
  N08[label="[0.125,0.375,0.5]"];
  N09[label=Categorical];
  N10[label=Sample];
  N11[label=index];
  N12[label=1];
  N13[label=Normal];
  N14[label=Sample];
  N15[label=Query];
  N00 -> N02;
  N00 -> N07;
  N01 -> N02;
  N01 -> N07;
  N02 -> N03;
  N02 -> N04;
  N02 -> N06;
  N03 -> N07;
  N04 -> N07;
  N05 -> N07;
  N06 -> N07;
  N07 -> N11;
  N08 -> N09;
  N09 -> N10;
  N10 -> N11;
  N11 -> N13;
  N12 -> N13;
  N13 -> N14;
  N14 -> N15;
}"""
        self.assertEqual(expected.strip(), observed.strip())
