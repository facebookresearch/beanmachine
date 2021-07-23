#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

# Categorical compiler tests

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference import BMGInference
from torch import tensor
from torch.distributions import Categorical, Dirichlet, HalfCauchy


t = tensor([0.125, 0.125, 0.25, 0.5])


@bm.random_variable
def c_const_simplex():
    return Categorical(t)


@bm.random_variable
def c_const_unnormalized():
    # If we have samples of both the normalized and unnormalized distributions
    # the deduplicator should merge them into the same distribution, since
    # 2:2:4:8 :: 1/8:1/8:1/4:1/2
    return Categorical(t * 16.0)


@bm.random_variable
def c_const_logit_simplex():
    # Note that logits here means log probabilities, not log odds.
    # Since the argument is just a constant, the runtime should detect
    # that it can simply reuse the [0.125, 0.125, 0.25, 0.5] node
    # in the generated graph.
    return Categorical(logits=t.log())


@bm.random_variable
def hc():
    return HalfCauchy(0.0)


@bm.random_variable
def c_random_logit():
    return Categorical(logits=tensor([0.0, 0.0, 0.0, -hc()]))


@bm.random_variable
def d4():
    return Dirichlet(tensor([1.0, 1.0, 1.0, 1.0]))


@bm.random_variable
def cd4():
    return Categorical(d4())


# TODO: random variable indexed by categorical
# TODO: multidimensional categorical


class CategoricalTest(unittest.TestCase):
    # TODO: Categorical is not yet marked as supported in BMG;
    # Update these tests to after_transform=True when it is,
    # to verify that the type checker is doing the proper transformations.
    def test_categorical(self) -> None:
        self.maxDiff = None
        queries = [
            c_const_simplex(),
            c_const_unnormalized(),
            c_const_logit_simplex(),
            cd4(),
            c_random_logit(),
        ]
        observations = {}
        observed = BMGInference().to_dot(queries, observations, after_transform=False)
        expected = """
digraph "graph" {
  N00[label="[0.125,0.125,0.25,0.5]"];
  N01[label=Categorical];
  N02[label=Sample];
  N03[label=Query];
  N04[label=Sample];
  N05[label=Query];
  N06[label=Sample];
  N07[label=Query];
  N08[label="[1.0,1.0,1.0,1.0]"];
  N09[label=Dirichlet];
  N10[label=Sample];
  N11[label=Categorical];
  N12[label=Sample];
  N13[label=Query];
  N14[label=0.0];
  N15[label=0.0];
  N16[label=HalfCauchy];
  N17[label=Sample];
  N18[label="-"];
  N19[label=Tensor];
  N20[label="Categorical(logits)"];
  N21[label=Sample];
  N22[label=Query];
  N00 -> N01;
  N01 -> N02;
  N01 -> N04;
  N01 -> N06;
  N02 -> N03;
  N04 -> N05;
  N06 -> N07;
  N08 -> N09;
  N09 -> N10;
  N10 -> N11;
  N11 -> N12;
  N12 -> N13;
  N14 -> N19;
  N14 -> N19;
  N14 -> N19;
  N15 -> N16;
  N16 -> N17;
  N17 -> N18;
  N18 -> N19;
  N19 -> N20;
  N20 -> N21;
  N21 -> N22;
}
        """
        self.assertEqual(expected.strip(), observed.strip())
