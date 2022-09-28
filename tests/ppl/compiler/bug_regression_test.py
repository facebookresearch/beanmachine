#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.inference import BMGInference

# https://github.com/facebookresearch/beanmachine/issues/1312


@bm.random_variable
def unif():
    return dist.Uniform(0, 1)


@bm.random_variable
def beta():
    return dist.Beta(unif() + 0.1, unif() + 0.1)


@bm.random_variable
def flip():
    return dist.Bernoulli(1 - beta())


class BugRegressionTest(unittest.TestCase):
    def test_regress_1312(self) -> None:
        self.maxDiff = None

        # There were two problems exposed by this user-supplied repro. Both are
        # now fixed.
        #
        # The first was that a typo in the code which propagated type mutations
        # through the graph during the problem-fixing phase was causing some types
        # to not be updated correctly, which was then causing internal compiler
        # errors down the line.
        #
        # The second was that due to the order in which the problem fixers ran,
        # the 1-beta operation was generated as:
        #
        # ToProb(Add(1.0, ToReal(Negate(ToPosReal(Sample(Beta(...))))))
        #
        # Which is not wrong but is quite inefficient. It is now generated as
        # you would expect:
        #
        # Complement(Sample(Beta(...)))

        queries = [unif()]
        observations = {flip(): torch.tensor(1)}

        observed = BMGInference().to_dot(queries, observations)
        expected = """
digraph "graph" {
  N00[label=Flat];
  N01[label=Sample];
  N02[label=ToPosReal];
  N03[label=0.1];
  N04[label="+"];
  N05[label=Beta];
  N06[label=Sample];
  N07[label=complement];
  N08[label=Bernoulli];
  N09[label=Sample];
  N10[label="Observation True"];
  N11[label=Query];
  N00 -> N01;
  N01 -> N02;
  N01 -> N11;
  N02 -> N04;
  N03 -> N04;
  N04 -> N05;
  N04 -> N05;
  N05 -> N06;
  N06 -> N07;
  N07 -> N08;
  N08 -> N09;
  N09 -> N10;
}
        """
        self.assertEqual(expected.strip(), observed.strip())
