# Copyright (c) Facebook, Inc. and its affiliates.
import math
import unittest

import numpy as np
import torch
from beanmachine import graph


class TestCAVI(unittest.TestCase):
    def test_interface(self):
        g = graph.Graph()
        c1 = g.add_constant_probability(0.1)
        d1 = g.add_distribution(
            graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, [c1]
        )
        o1 = g.add_operator(graph.OperatorType.SAMPLE, [d1])
        g.query(o1)
        mean_vars = g.variational(100, 10, elbo_samples=100)
        self.assertEqual(len(mean_vars), 1, "number of queries")
        self.assertEqual(len(mean_vars[0]), 1, "each parameter must be mean")
        elbo = g.get_elbo()
        self.assertEqual(len(elbo), 100, "one ELBO value per iteration")
        mean_vars = g.variational(100, 10)  # elbo_samples=0 by default
        elbo = g.get_elbo()
        self.assertEqual(len(elbo), 0, "ELBO not computed unless requested")

    def build_graph1(self):
        """
        o1 ~ Bernoulli( 0.1 )
        o2 ~ Bernoulli( exp( - o1 ) )
        infer P(o1 | o2 = True)
        now, P(o1 = T, o2 = T) = 0.1 * exp(-1) = 0.036787944117144235
        and, P(o1 = F, o2 = T) = 0.9 * exp(0) = 0.9
        => P(o1 = True | o2 = True) = 0.03927030055005057
        also P(o2 = True) = 0.9367879441171443 >= ELBO
        """
        g = graph.Graph()
        c1 = g.add_constant_probability(0.1)
        d1 = g.add_distribution(
            graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, [c1]
        )
        o1 = g.add_operator(graph.OperatorType.SAMPLE, [d1])
        c2 = g.add_constant_row_simplex_matrix(
            np.array([[0.0, 1.0], [1 - math.exp(-1), math.exp(-1)]])
        )
        d2 = g.add_distribution(
            graph.DistributionType.TABULAR, graph.AtomicType.BOOLEAN, [c2, o1]
        )
        o2 = g.add_operator(graph.OperatorType.SAMPLE, [d2])
        g.observe(o2, True)
        g.query(o1)
        return g

    def test_cavi1(self):
        g = self.build_graph1()
        means = g.variational(1, 1, elbo_samples=100)
        self.assertAlmostEqual(0.039, means[0][0], 2, "posterior mean")
        elbo = g.get_elbo()
        self.assertGreater(math.log(0.94), elbo[0], "ELBO")

    def test_gibbs1(self):
        g = self.build_graph1()
        samples = g.infer(1000, graph.InferenceType.GIBBS)
        means = torch.tensor(samples, dtype=float).mean(axis=0)
        self.assertGreater(means[0].item(), 0.03)
        self.assertLess(means[0].item(), 0.05)

    def build_graph2(self):
        """
        This is a simplified noisy-or model
        X ~ Bernoulli(0.01)
        Y ~ Bernoulli(0.01)
        Z ~ Bernoulli(1 - exp( log(0.99) + log(0.01)*X + log(0.01)*Y ))
        Note: the last line is equivalent to:
        Z ~ BernoulliNoisyOr( - ( log(0.99) + log(0.01)*X + log(0.01)*Y ) )
        OR
        Z ~ BernoulliNoisyOr( -log(0.99) + (-log(0.01))*X + (-log(0.01))*Y ) )
        query (X, Y) observe Z = True

        X  Y  P(X, Y, Z=T)  P(X, Y | Z=T)
        ---------------------------------
        F  F  0.009801        0.3322
        F  T  0.009802        0.3322
        T  F  0.009802        0.3322
        T  T  0.0000999901    0.0034

        P(Z=T) = 0.029505, this ELBO <= log(.029505) = -3.5232

        Let Q(X) = Q(Y) = Bernoulli(q); The KL-Divergence as a function of q is:
        kl = lambda q: (1-q)**2 * (2*log(1-q)-log(.3322))
         + 2*q*(1-q)*(log(q)+log(1-q)-log(.3322)) + q**2 * (2*log(q)-log(.0034))

        KL Divergence is minimized at q=0.245, and kl(.245) = .2635

        And max ELBO = log P(Z=T) - kl(.245) = -3.7867
        """
        g = graph.Graph()
        c_prior = g.add_constant_probability(0.01)
        d_prior = g.add_distribution(
            graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, [c_prior]
        )
        x = g.add_operator(graph.OperatorType.SAMPLE, [d_prior])
        y = g.add_operator(graph.OperatorType.SAMPLE, [d_prior])
        pos_x = g.add_operator(graph.OperatorType.TO_POS_REAL, [x])
        pos_y = g.add_operator(graph.OperatorType.TO_POS_REAL, [y])
        c_m_log_pt01 = g.add_constant_pos_real(-(math.log(0.01)))
        c_m_log_pt99 = g.add_constant_pos_real(-(math.log(0.99)))
        param = g.add_operator(
            graph.OperatorType.ADD,
            [
                c_m_log_pt99,
                g.add_operator(graph.OperatorType.MULTIPLY, [c_m_log_pt01, pos_x]),
                g.add_operator(graph.OperatorType.MULTIPLY, [c_m_log_pt01, pos_y]),
            ],
        )
        d_like = g.add_distribution(
            graph.DistributionType.BERNOULLI_NOISY_OR, graph.AtomicType.BOOLEAN, [param]
        )
        z = g.add_operator(graph.OperatorType.SAMPLE, [d_like])
        g.observe(z, True)
        g.query(x)
        g.query(y)
        return g

    def test_gibbs2(self):
        g = self.build_graph2()
        samples = torch.tensor(g.infer(10000, graph.InferenceType.GIBBS), dtype=float)
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
        means = g.variational(100, 1000, elbo_samples=1000)
        self.assertAlmostEqual(
            means[0][0], means[1][0], 1, "X and Y have same variational posterior"
        )
        self.assertAlmostEqual(means[0][0], 0.245, 1, "X posterior is ?")
        elbo = g.get_elbo()
        self.assertAlmostEqual(elbo[-1], -3.7867, 1, "ELBO converged")
