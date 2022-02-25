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

        # TODO: There are two problems exposed by this user-supplied repro.
        # The first, which is fixed, is that a typo in the code which propagated
        # type mutations through the graph during the problem-fixing phase was
        # causing some types to not be updated correctly, which was then causing
        # internal compiler errors down the line.
        #
        # The second, which is not yet fixed, is that this repro demonstrates that
        # the problem fixers need to run repeatedly until a fixpoint is reached.
        # Notice the very strange graph generation here: the operation 1-beta should
        # be generated as Complement(Sample(Beta(...))) but instead is generated as
        # ToProb(Add(ToReal(Negate(ToPosReal(Sample(Beta(...)))), 1.0))! The generated
        # code will work but is way more nodes than it needs to be.
        #
        # What is happening here is: the AdditionFixer is running first but skipping
        # the rewrite of 1-beta into a complement because beta has an untypable
        # Uniform(0, 1) node in its ancestors. We rewrite that into a Flat node in
        # the UnsupportedNodeFixer, but we do not then re-run the AdditionFixer.
        #
        # What we should do is have each fixer behave like the parse tree rewriters.
        # That is, we should have them report on whether they made progress. We can
        # then write combinators that keep running rewriters until they stop making
        # progress and we've either reported errors or attained a fixpoint.

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
  N07[label=1.0];
  N08[label=ToPosReal];
  N09[label="-"];
  N10[label=ToReal];
  N11[label="+"];
  N12[label=ToProb];
  N13[label=Bernoulli];
  N14[label=Sample];
  N15[label="Observation True"];
  N16[label=Query];
  N00 -> N01;
  N01 -> N02;
  N01 -> N16;
  N02 -> N04;
  N03 -> N04;
  N04 -> N05;
  N04 -> N05;
  N05 -> N06;
  N06 -> N08;
  N07 -> N11;
  N08 -> N09;
  N09 -> N10;
  N10 -> N11;
  N11 -> N12;
  N12 -> N13;
  N13 -> N14;
  N14 -> N15;
}
        """
        self.assertEqual(expected.strip(), observed.strip())
