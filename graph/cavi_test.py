# Copyright (c) Facebook, Inc. and its affiliates.
import math
import unittest

import torch
from beanmachine import graph


class TestCAVI(unittest.TestCase):
    def test_interface(self):
        g = graph.Graph()
        c1 = g.add_constant(0.1)
        d1 = g.add_distribution(
            graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, [c1]
        )
        o1 = g.add_operator(graph.OperatorType.SAMPLE, [d1])
        g.query(o1)
        mean_vars = g.variational(100, 10)
        self.assertEqual(len(mean_vars), 1, "number of queries")
        self.assertEqual(len(mean_vars[0]), 1, "each parameter must be mean")

    def build_graph1(self):
        g = graph.Graph()
        c1 = g.add_constant(0.1)
        d1 = g.add_distribution(
            graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, [c1]
        )
        o1 = g.add_operator(graph.OperatorType.SAMPLE, [d1])
        o2 = g.add_operator(graph.OperatorType.TO_REAL, [o1])
        o3 = g.add_operator(graph.OperatorType.NEGATE, [o2])
        o4 = g.add_operator(graph.OperatorType.EXP, [o3])
        d2 = g.add_distribution(
            graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, [o4]
        )
        o5 = g.add_operator(graph.OperatorType.SAMPLE, [d2])
        g.observe(o5, True)
        g.query(o1)
        return g

    def test_cavi1(self):
        g = self.build_graph1()
        means = g.variational(1, 1)
        self.assertAlmostEqual(0.039, means[0][0], 2, "posterior mean")

    def test_gibbs1(self):
        g = self.build_graph1()
        samples = g.infer(1000, graph.InferenceType.GIBBS)
        means = torch.tensor(
            [[x.bool for x in sample] for sample in samples], dtype=float
        ).mean(axis=0)
        self.assertGreater(means[0].item(), 0.03)
        self.assertLess(means[0].item(), 0.05)

    def build_graph2(self):
        g = graph.Graph()
        # This is a simplified noisy-or model
        # X ~ Bernoulli(0.01)
        # Y ~ Bernoulli(0.01)
        # Z ~ Bernoulli(1 - exp( log(0.99) + log(0.01)*X + log(0.01)*Y ))
        # query (X, Y) observe Z = True
        # Correct answer: either X or Y are True with probability 0.33 and
        # 0.33 prob of neither being true and 0.0033 odds of both being true
        c_one = g.add_constant(1.0)
        c_prior = g.add_constant(0.01)
        d_prior = g.add_distribution(
            graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, [c_prior]
        )
        x = g.add_operator(graph.OperatorType.SAMPLE, [d_prior])
        y = g.add_operator(graph.OperatorType.SAMPLE, [d_prior])
        real_x = g.add_operator(graph.OperatorType.TO_REAL, [x])
        real_y = g.add_operator(graph.OperatorType.TO_REAL, [y])
        c_log_pt01 = g.add_constant(math.log(0.01))
        c_log_pt99 = g.add_constant(math.log(0.99))
        logprob = g.add_operator(
            graph.OperatorType.ADD,
            [
                c_log_pt99,
                g.add_operator(graph.OperatorType.MULTIPLY, [c_log_pt01, real_x]),
                g.add_operator(graph.OperatorType.MULTIPLY, [c_log_pt01, real_y]),
            ],
        )
        prob = g.add_operator(graph.OperatorType.EXP, [logprob])
        prob = g.add_operator(graph.OperatorType.NEGATE, [prob])
        prob = g.add_operator(graph.OperatorType.ADD, [c_one, prob])
        d_like = g.add_distribution(
            graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, [prob]
        )
        z = g.add_operator(graph.OperatorType.SAMPLE, [d_like])
        g.observe(z, True)
        g.query(x)
        g.query(y)
        return g

    def test_gibbs2(self):
        g = self.build_graph2()
        samples = torch.tensor(
            [
                [x.bool for x in sample]
                for sample in g.infer(10000, graph.InferenceType.GIBBS)
            ],
            dtype=float,
        )
        x_marginal = samples.mean(axis=0)[0].item()
        y_marginal = samples.mean(axis=0)[1].item()
        x_y_joint = (samples[:, 0] * samples[:, 1]).mean().item()
        self.assertAlmostEqual(
            x_marginal, y_marginal, 1, "posterior marginal of x and y are nearly equal"
        )
        self.assertAlmostEqual(x_marginal, 0.33, 1, "posterior x is 0.33")
        self.assertLess(x_y_joint, 0.01, "joint posterior of x and y < 0.01")

    def test_cavi2(self):
        g = self.build_graph2()
        means = g.variational(100, 100)
        self.assertAlmostEqual(
            means[0][0], means[1][0], 1, "X and Y have same variational posterior"
        )
        self.assertAlmostEqual(means[0][0], 0.245, 1, "X posterior is ?")
