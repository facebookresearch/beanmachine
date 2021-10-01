# Copyright (c) Facebook, Inc. and its affiliates.
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
