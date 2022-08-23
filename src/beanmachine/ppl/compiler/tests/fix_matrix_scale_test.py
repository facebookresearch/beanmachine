# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
import torch
from beanmachine.ppl.inference import BMGInference
from torch.distributions import Normal


@bm.random_variable
def scalar():
    return Normal(0.0, 1.0)


matrix = torch.tensor([20, 40])


@bm.functional
def scaled():
    return scalar() * matrix


@bm.functional
def scaled_sym():
    return matrix * scalar()


@bm.functional
def scaled2():
    return scalar() * torch.tensor([scalar(), scalar()])


@bm.functional
def scaled2_sym():
    return (torch.tensor([scalar(), scalar()])) * scalar()


@bm.functional
def multiple_scalars():
    return scalar() * scalar() * matrix * scalar() * scalar()


class FixMatrixScaleTest(unittest.TestCase):
    def test_fix_matrix_scale_1(self) -> None:
        self.maxDiff = None
        observations = {}
        queries = [scaled()]
        num_samples = 1000
        num_chains = 1

        # Sanity check to make sure the model is valid
        nmc = bm.SingleSiteNewtonianMonteCarlo()
        _ = nmc.infer(
            queries=queries,
            observations=observations,
            num_samples=num_samples,
            num_chains=num_chains,
        )

        observed = BMGInference().to_dot(queries, observations, after_transform=False)

        # The model before optimization

        expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label="[20,40]"];
  N5[label="*"];
  N6[label=Query];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N5;
  N4 -> N5;
  N5 -> N6;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # After optimization:

        observed = BMGInference().to_dot(queries, observations, after_transform=True)
        expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label="[20,40]"];
  N5[label=MatrixScale];
  N6[label=Query];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N5;
  N4 -> N5;
  N5 -> N6;
}"""
        self.assertEqual(expected.strip(), observed.strip())

        # The model runs on Bean Machine Graph

        _ = BMGInference().infer(queries, observations, num_samples=num_samples)

    def test_fix_matrix_scale_1_sym(self) -> None:
        self.maxDiff = None
        observations = {}
        queries = [scaled_sym()]
        num_samples = 1000
        num_chains = 1

        # Sanity check to make sure the model is valid
        nmc = bm.SingleSiteNewtonianMonteCarlo()
        _ = nmc.infer(
            queries=queries,
            observations=observations,
            num_samples=num_samples,
            num_chains=num_chains,
        )

        observed = BMGInference().to_dot(queries, observations, after_transform=False)

        # The model before optimization

        expected = """
digraph "graph" {
  N0[label="[20,40]"];
  N1[label=0.0];
  N2[label=1.0];
  N3[label=Normal];
  N4[label=Sample];
  N5[label="*"];
  N6[label=Query];
  N0 -> N5;
  N1 -> N3;
  N2 -> N3;
  N3 -> N4;
  N4 -> N5;
  N5 -> N6;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # After optimization:

        observed = BMGInference().to_dot(queries, observations, after_transform=True)
        expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label="[20,40]"];
  N5[label=MatrixScale];
  N6[label=Query];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N5;
  N4 -> N5;
  N5 -> N6;
}"""
        self.assertEqual(expected.strip(), observed.strip())

        # The model runs on Bean Machine Graph

        _ = BMGInference().infer(queries, observations, num_samples=num_samples)

    def test_fix_matrix_scale_2(self) -> None:
        self.maxDiff = None
        observations = {}
        queries = [scaled2()]
        num_samples = 1000
        num_chains = 1

        # Sanity check to make sure the model is valid
        nmc = bm.SingleSiteNewtonianMonteCarlo()
        _ = nmc.infer(
            queries=queries,
            observations=observations,
            num_samples=num_samples,
            num_chains=num_chains,
        )

        observed = BMGInference().to_dot(queries, observations, after_transform=False)

        # The model before optimization
        expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label=Tensor];
  N5[label="*"];
  N6[label=Query];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
  N3 -> N4;
  N3 -> N5;
  N4 -> N5;
  N5 -> N6;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # After optimization:

        observed = BMGInference().to_dot(queries, observations, after_transform=True)
        expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label=2];
  N5[label=1];
  N6[label=ToMatrix];
  N7[label=MatrixScale];
  N8[label=Query];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N6;
  N3 -> N6;
  N3 -> N7;
  N4 -> N6;
  N5 -> N6;
  N6 -> N7;
  N7 -> N8;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # The model runs on Bean Machine Graph

        _ = BMGInference().infer(queries, observations, num_samples=num_samples)

    def test_fix_matrix_scale_2_sym(self) -> None:
        self.maxDiff = None
        observations = {}
        queries = [scaled2_sym()]
        num_samples = 1000
        num_chains = 1

        # Sanity check to make sure the model is valid
        nmc = bm.SingleSiteNewtonianMonteCarlo()
        _ = nmc.infer(
            queries=queries,
            observations=observations,
            num_samples=num_samples,
            num_chains=num_chains,
        )

        observed = BMGInference().to_dot(queries, observations, after_transform=False)

        # The model before optimization
        expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label=Tensor];
  N5[label="*"];
  N6[label=Query];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
  N3 -> N4;
  N3 -> N5;
  N4 -> N5;
  N5 -> N6;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # After optimization:

        observed = BMGInference().to_dot(queries, observations, after_transform=True)
        expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label=2];
  N5[label=1];
  N6[label=ToMatrix];
  N7[label=MatrixScale];
  N8[label=Query];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N6;
  N3 -> N6;
  N3 -> N7;
  N4 -> N6;
  N5 -> N6;
  N6 -> N7;
  N7 -> N8;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # The model runs on Bean Machine Graph

        _ = BMGInference().infer(queries, observations, num_samples=num_samples)

    def test_fix_matrix_scale_3(self) -> None:
        # TODO: The matrix scale optimizer correctly removes the extra matrix scale
        # but the multiary multiplication optimizer does not optimize to a single
        # multiplication node. That optimizer does not optimize nodes where the
        # outgoing edge count is more than one, but in this case the outgoing
        # edges are to orphaned nodes, illustrating a flaw in this design.
        # We might consider always doing the optimization even if there are multiple
        # outgoing edges -- that risks making a suboptimal graph but that scenario
        # is likely rare. Or we could write an orphan-trimming pass.
        self.maxDiff = None
        observations = {}
        queries = [multiple_scalars()]
        observed = BMGInference().to_dot(queries, observations)
        expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label="*"];
  N5[label="*"];
  N6[label="*"];
  N7[label="[20,40]"];
  N8[label=MatrixScale];
  N9[label=Query];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
  N3 -> N4;
  N3 -> N5;
  N3 -> N6;
  N4 -> N5;
  N5 -> N6;
  N6 -> N8;
  N7 -> N8;
  N8 -> N9;
}
"""
        self.assertEqual(expected.strip(), observed.strip())
